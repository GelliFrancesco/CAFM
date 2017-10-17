'''
@author:
Francesco Gelli (francesco.gelli@u.nus.edu)
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)
'''

import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
import LoadData as DATA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from utils import build_itemlist, build_userlist, read_traits
import h5py
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from random import shuffle
import shutil


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run CAFM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='context',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=8,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.0001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer,'
                             ' MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--item_ft', type=int, default=0,
                        help='Whether to use item dense ft or not')
    parser.add_argument('--user_ft', type=int, default=1,
                        help='Whether to use user dense ft or not')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                    help='Keep probability (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to use batchnorm or not')

    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, hidden_factor, epoch, batch_size, learning_rate,
                 lamda_bilinear, optimizer_type, verbose, data_folder, num_users, num_items, item_ft,
                 user_ft, keep, batch_norm, random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.keep = keep
        self.batch_norm = batch_norm
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.verbose = verbose
        # performance of each epoch
        self.data_folder = data_folder
        self.item_ft = item_ft
        self.user_ft = user_ft

        self._init_graph()
        if item_ft:
            print 'Reading User Features...'
            self.item_tr = read_dense_data(data_folder+'/training/item_dense.h5')
            self.item_ts = read_dense_data(data_folder + '/testing/item_dense.h5')
            print '...Done Reading User Features'
            self.num_users = num_users
            self.num_items = num_items
        if user_ft:
            print 'Reading User Features...'
            self.user_features = read_traits(data_folder + '/users/traits.csv')
            print '...Done Reading User Features'

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            tf.set_random_seed(self.random_seed)  # Set graph level random seed
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            if self.item_ft:
                self.train_features_item_ft = tf.placeholder(tf.float32, shape=[None, None])
                self.item_transform = tf.Variable(initial_value=tf.truncated_normal([4342, self.hidden_factor],
                                                                                         -0.005, 0.004))
                self.item_features_bias = tf.Variable(initial_value=tf.truncated_normal([self.hidden_factor], 0.04, 0.063))
                self.item_features_embedding = tf.matmul(self.train_features_item_ft, self.item_transform) +\
                                           self.item_features_bias
                self.variable_summaries(self.item_transform, 'item_transform')
                self.variable_summaries(self.item_features_bias, 'item_features_bias')

            if self.user_ft:
                self.train_features_user_ft = tf.placeholder(tf.float32, shape=[None, None])
                self.user_transform = tf.Variable(initial_value=tf.truncated_normal([5, self.hidden_factor], -0.6, 0.24))
                self.user_features_bias = tf.Variable(initial_value=tf.truncated_normal([self.hidden_factor], -0.75, 0.27))
                self.user_features_embedding = tf.matmul(self.train_features_user_ft, self.user_transform) + self.user_features_bias
                self.variable_summaries(self.user_features_bias, 'user_features_bias')
                self.variable_summaries(self.user_transform, 'user_transform')

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            # tf.summary.scalar('shape_of_nonzero_embeddings', self.sh1)
            self.summed_features_emb = tf.reduce_sum(self.nonzero_embeddings, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(self.nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square,
                                   self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep) # dropout at the FM layer

            # sparse-dense interaction
            if self.item_ft:
                self.denseFM_item = tf.reduce_sum(tf.matmul(self.nonzero_embeddings, tf.expand_dims(
                    self.item_features_embedding, 2)), 1)
            if self.user_ft:
                self.denseFM_user = tf.reduce_sum(tf.matmul(self.nonzero_embeddings, tf.expand_dims(
                    self.user_features_embedding, 2)), 1)

            # dense-dense interaction
            if self.item_ft & self.user_ft:
                self.user_item_interaction = tf.reduce_sum(tf.reduce_sum(tf.matmul(tf.expand_dims(
                    self.item_features_embedding, 2), tf.transpose(tf.expand_dims(self.user_features_embedding, 2),
                                                               perm=[0,2,1])), 1), 1, keep_dims=True)

            # _________out _________
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                                              1)  # None * 1
            ones = tf.ones_like(self.train_labels, dtype=tf.float32)
            Bias = tf.multiply(self.weights['bias'], ones)  # None * 1

            if self.item_ft & self.user_ft:

                self.out = tf.add_n([Bilinear, self.Feature_bias, Bias,
                                     self.denseFM_item,
                                     self.denseFM_user,
                                     self.user_item_interaction])
            elif self.item_ft & (not self.user_ft):
                self.out = tf.add_n([Bilinear, self.Feature_bias, Bias, self.denseFM_item])
            elif (not self.item_ft) & self.user_ft:
                self.out = tf.add_n([Bilinear, self.Feature_bias, Bias, self.denseFM_user])
            else:
                self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])

            self.out_sig = tf.sigmoid(self.out)
            tf.summary.histogram('output', self.out_sig)
            self.non_reg_loss = tf.losses.log_loss(self.train_labels, self.out_sig)
            if self.lamda_bilinear > 0:
                loss = self.non_reg_loss + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings']) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_bias'])
                if self.item_ft & self.user_ft:
                    self.loss = loss + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.item_transform)+ tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.user_transform) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.item_features_bias) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.user_features_bias)
                elif self.item_ft & (not self.user_ft):
                    self.loss = loss  + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.item_transform)  + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.item_features_bias)
                elif (not self.item_ft) & self.user_ft:
                    self.loss = loss + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.user_transform) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.user_features_bias)
                else:
                    self.loss = loss
            else:
                self.loss = tf.losses.log_loss(self.train_labels, self.out_sig)

            tf.summary.scalar('loss', self.loss)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('logDir/init/', self.sess.graph)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()

        c1 = tf.Variable(tf.truncated_normal([self.features_M, self.hidden_factor], 0.0002, 0.0125), name='feature_embeddings')
        c2 = tf.constant(0.0, tf.float32, [1, self.hidden_factor])
        all_weights['feature_embeddings'] = tf.concat([c1, c2],0)
        self.variable_summaries(all_weights['feature_embeddings'], 'feature_embeddings')

        all_weights['feature_bias'] = tf.concat([tf.Variable(tf.truncated_normal([self.features_M, 1], -0.00015,
                                                                                 0.0022, dtype=tf.float32),
                                                             name='feature_bias', dtype=tf.float32),
                                                 tf.constant(0.0, tf.float32, [1, 1])],0) # features_M * 1
        self.variable_summaries(all_weights['feature_bias'], 'feature_bias')
        all_weights['bias'] = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='bias', dtype=tf.float32)  # 1 * 1
        tf.summary.scalar('bias', all_weights['bias'])

        return all_weights

    def variable_summaries(self, var, name):
        """Attach summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name+'_mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(name+'_stddev', stddev)
            tf.summary.scalar(name+'_max', tf.reduce_max(var))
            tf.summary.scalar(name+'_min', tf.reduce_min(var))
            tf.summary.histogram(name+'_histogram', var)
        return

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data, i):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'],
                     self.dropout_keep: self.keep, self.train_phase: True}
        if self.item_ft:
            feed_dict[self.train_features_item_ft] = self.item_tr[data['Item'], :]
        if self.user_ft:
            feed_dict[self.train_features_user_ft] = self.user_features[data['User'], :]

        summary, loss, non_reg_loss, opt = self.sess.run((self.merged, self.loss, self.non_reg_loss, self.optimizer),
                                                         feed_dict=feed_dict)
        self.train_writer.add_summary(summary, i)
        return loss

    def get_random_block_from_data(self, data, batch_size):
        '''
        sample a batch randomly. Sample is constrained to a 3/4 proportion between positive and negative instances
        '''

        ones_indexes = [i for i, v in enumerate(data['Y']) if v == 1.0]
        shuffle(ones_indexes)
        ones_indexes = ones_indexes[:batch_size/4]
        zeros_indexes = [i for i, v in enumerate(data['Y']) if v == 0.0]
        shuffle(zeros_indexes)
        zeros_indexes = zeros_indexes[:batch_size*3/4]

        perm = list(np.random.permutation(len(ones_indexes+zeros_indexes)))

        res = {'X': [x for (y, x) in sorted(zip(perm, [data['X'][i] for i in ones_indexes+zeros_indexes]))],
               'Y': [x for (y, x) in sorted(zip(perm, [[data['Y'][i]] for i in ones_indexes+zeros_indexes]))]}

        if self.item_ft:
            res['Item'] = [x for (y, x) in sorted(zip(perm, [data['Item'][i] for i in ones_indexes+zeros_indexes]))]
        if self.user_ft:
            res['User'] = [x for (y, x) in sorted(zip(perm, [data['User'][i] for i in ones_indexes+zeros_indexes]))]

        return res

    def get_blocks_from_data(self, data, batch_size, random=False):
        '''
        splits data in sequential or randomly sorted blocks
        '''
        perm = list(np.random.permutation(len(data['X'])))

        if ~random:
            X = data['X']
            Y = data['Y']
            if self.item_ft:
                Item = data['Item']
            if self.user_ft:
                User = data['User']
        else:
            X = [x for (y, x) in sorted(zip(perm, data['X']))]
            Y = [x for (y, x) in sorted(zip(perm, data['Y']))]
            if self.item_ft:
                Item = [x for (y, x) in sorted(zip(perm, data['Item']))]
            if self.user_ft:
                User = [x for (y, x) in sorted(zip(perm, data['User']))]

        chunked_X = [X[i:i + batch_size] for i in xrange(0, len(X), batch_size)]
        chunked_Y = [Y[i:i + batch_size] for i in xrange(0, len(Y), batch_size)]
        if self.item_ft:
            chunked_Items = [Item[i:i + batch_size] for i in xrange(0, len(Item), batch_size)]
        if self.user_ft:
            chunked_Users = [User[i:i + batch_size] for i in xrange(0, len(User), batch_size)]

        chunked_data = []
        for k in range(len(chunked_Y)):
            chunked_dict = {'X': chunked_X[k], 'Y': [[e] for e in chunked_Y[k]]}
            if self.item_ft:
                chunked_dict['Item'] = chunked_Items[k]
            if self.user_ft:
                chunked_dict['User'] = chunked_Users[k]
            chunked_data.append(chunked_dict)

        return chunked_data

    def train(self, Train_data, Test_data):
        with open('logDir/loss.csv', 'w') as w:
            for e, epoch in enumerate(xrange(self.epoch)):
                t1 = time()
                cost = []
                total_batch = int(len(Train_data['Y']) / self.batch_size)
                print 'iterating on', total_batch, 'blocks...'
                for i in xrange(total_batch):
                    batch_xs = self.get_random_block_from_data(Train_data, self.batch_size) # generate a batch data
                    cost.append(self.partial_fit(batch_xs, i)) # Fit training using batch data

                t2 = time()
                print 'computing results...' # output validation
                p_tr, r_tr, f_tr, ll_tr, acc_tr = self.evaluate(self.get_blocks_from_data(
                    Train_data, self.batch_size), 'training', e)
                p_ts, r_ts, f_ts, ll_ts, acc_ts = self.evaluate(self.get_blocks_from_data(
                    Test_data, self.batch_size), 'testing', e)
                w.write(str(ll_tr)+','+str(ll_ts)+'\n')

                if self.verbose > 0:
                    print("Epoch %d [%.1f s] Precision: \ttrain=%.4f,  test=%.4f [%.1f s]"
                          % (epoch + 1, t2 - t1, p_tr, p_ts, time() - t2))
                    print("Epoch %d [%.1f s] Recall: \ttrain=%.4f, test=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, r_tr, r_ts, time() - t2))
                    print("Epoch %d [%.1f s] FScore: \ttrain=%.4f, test=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, f_tr, f_ts, time() - t2))
                    print("Epoch %d [%.1f s] LogLoss: \ttrain=%.4f, test=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, ll_tr, ll_ts, time() - t2))
                    print("Epoch %d [%.1f s] Accuracy: \ttrain=%.4f, test=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, acc_tr, acc_ts, time() - t2))
                print 'Done computing results'
                if (e % 10 == 0) & (e == 99):
                    self.saver.save(self.sess, 'logDir/init/checkpoint', global_step=e)
            print "Test performance: ", self.evaluate(self.get_blocks_from_data(Test_data, self.batch_size), 'testing', e)

    def evaluate(self, data_blocks_to_evaluate, mode, e):  # evaluate the results for an input set
        y_pred = np.array([])
        y_true = np.array([])

        print 'evaluate', mode, 'num blocks:', len(data_blocks_to_evaluate)
        for block in data_blocks_to_evaluate:
            feed_dict = {self.train_features: block['X'], self.train_labels: block['Y'],
                         self.dropout_keep: 1.0, self.train_phase: False}
            if self.item_ft:
                if mode == 'training':
                    feed_dict[self.train_features_item_ft] = self.item_tr[block['Item'], :]
                else:
                    feed_dict[self.train_features_item_ft] = self.item_ts[block['Item'], :]

            if self.user_ft:
                feed_dict[self.train_features_user_ft] = self.user_features[block['User'], :]

            block_predictions = self.sess.run(self.out_sig, feed_dict=feed_dict)
            block_true = block['Y']
            y_pred = np.concatenate((y_pred, np.reshape(block_predictions, (len(block_true),))), axis=0)
            y_true = np.concatenate((y_true, np.reshape(block_true, (len(block_true),))), axis=0)

        # # print useful stats
        # print mode, 'Min:', np.min(y_pred), '| Max:', np.max(y_pred), '| Len:', len(y_pred),\
        #     '| <0.5:', round(float((y_pred < 0.5).sum())/len(y_pred)*100, 2), '%', \
        #     '| Ones:', round(float((y_pred == 1.0).sum())/len(y_pred)*100, 2), \
        #     '| Zeros:', round(float((y_pred == 0.0).sum())/len(y_pred)*100, 2)

        predictions_binary = []
        for item in y_pred:
            if item > 0.5:
                predictions_binary.append(1.0)
            else:
                predictions_binary.append(0.0)
        Performance = precision_recall_fscore_support(y_true, predictions_binary, pos_label=1, average='binary')

        g = tf.Graph()
        with g.as_default():
            l = tf.losses.log_loss(tf.constant(y_true), tf.constant(y_pred))
            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as ll_sess:
                ll = ll_sess.run(l)

        if (e % 10 == 0) & (mode == 'testing'):
            with open('logDir/init/y_true-'+str(e)+'.txt', 'w') as w:
                for item in y_true:
                    w.write("%s\n" % item)
            with open('logDir/init/y_pred-'+str(e)+'.txt', 'w') as w:
                for item in y_pred:
                    w.write("%s\n" % item)
        return Performance[0], Performance[1], Performance[2], ll, accuracy_score(y_true, predictions_binary)


def read_dense_data(path):
    with h5py.File(path, "r") as f:
        item_ft = np.array(f['output'], dtype='float32')
    return item_ft


if __name__ == '__main__':
    '''
    This main function is specific to the experiments performed in the paper.
    '''

    args = parse_args()

    if os.path.isdir('logDir'):
        shutil.rmtree('logDir/')

    items = build_itemlist('training/ratings.txt', 'testing/ratings.txt', 'error_imgs.txt',  args.path) # ordered list of image tweets is created

    users_discard = [u.lower() for u in
                     ['ConnorRyan90', 'grierrxnash', 'austio311', 'Nodays_off__', 'cutiestylespie', 'eazzz_e',
                      'eminemlights', 'Im_Wierdd1027', 'shay1498', 'urchkin', 'miley23isHot']] # for these users the tweet history was not long enough for accurate personality extraction

    users_filtered = build_userlist('users/traits.csv', users_discard,  args.path) #  the 862 users in the dataset
    users_all = build_userlist('users/traits.csv', [],  args.path) # all the users in the csv

    data = DATA.LoadData(args.path, args.dataset, items, users_filtered, args.item_ft, args.user_ft, users_all) # instance of class LoadData

    if args.verbose > 0:
        print("FM: dataset=%s, factors=%d, num_epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, optimizer=%s"
              % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda,
                 args.optimizer))

    model = FM(data.features_M, args.hidden_factor, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose, args.path, len(users_filtered),
               len(items), args.item_ft, args.user_ft, args.keep_prob, args.batch_norm)

    model.train(data.Train_data, data.Test_data)
