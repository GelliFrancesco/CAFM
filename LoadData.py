import numpy as np
from utils import read_image_index
import pickle


class LoadData( object ):
    '''given the path of data, return the required data format
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors
    with 0 or 1 entries
    Test_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, item_indeces, user_indeces, item_ft, personality_ft, traits_index):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset +".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.trainfile_index = self.path + "index_tr.csv"
        self.testfile_index = self.path + "index_ts.csv"
        self.item_indeces = item_indeces
        self.user_indeces = user_indeces
        self.traits_index = traits_index # all the users in the csv, for indexing
        self.data_folder = path
        self.features_M = self.map_features()
        self.item_ft = item_ft
        self.personality_ft = personality_ft
        self.Train_data, self.Test_data = self.construct_data()

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        return len(self.features)

    def read_features(self, file): # read a feature file
        f = open( file )
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.features:
                    self.features[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self):
        X_, Y_ , Y_for_logloss = self.read_data(self.trainfile, self.trainfile_index)
        Train_data = self.construct_dataset(X_, Y_for_logloss, 'training')

        X_, Y_ , Y_for_logloss = self.read_data(self.testfile,  self.testfile_index)
        Test_data = self.construct_dataset(X_, Y_for_logloss, 'testing')

        return Train_data,  Test_data

    def read_data(self, file, file_index):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        with open(file_index, 'r') as f_ind:
            ind = f_ind.readlines()
        f = open( file )
        X_ = []
        Y_ = []
        Y_for_logloss = []
        indexes = []

        line = f.readline()
        i = 0
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0: # > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )
            indexes.append((ind[i].split(',')[0],ind[i].split(',')[1].replace('\n', '')))
            line = f.readline()
            i += 1
        f.close()
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_, mode, load=False):
        if load == False:
            Data_Dic = {}
            X_lens = [len(line) for line in X_]
            indexs = np.argsort(X_lens)
            maxsize = max(X_lens)
            Data_Dic['Y'] = [ Y_[i] for i in indexs]
            Data_Dic['X'] = [ X_[i] for i in indexs]

            inv_dic = {v: k for k, v in self.features.iteritems()}
            users = [int(inv_dic[el[0]].split(':')[0]) for el in Data_Dic['X']]
            num_users = len(set(users))
            Data_Dic['X'] = []
            Data_Dic['Y'] = []

            # need to get the tweet id, then look into image_index.txt and find the index for the dense features
            im_dic = {k: v for v, k in enumerate(read_image_index(self.data_folder+mode+'/image_index.txt'))}
            inv_item_indeces = {v: k for k, v in self.item_indeces.iteritems()}
            inv_user_indeces = {v: k for k, v in self.user_indeces.iteritems()}
            Data_Dic['Item'] = []
            Data_Dic['User'] = []
            for i in range(len(X_)):
                el = X_[i]
                user = inv_user_indeces[int(inv_dic[el[0]].split(':')[0])]
                tweet = inv_item_indeces[int(inv_dic[el[1]].split(':')[0]) - num_users]

                if tweet in im_dic:
                    Data_Dic['X'].append(X_[i] + ([self.features_M]*(maxsize-len(X_[i]))))
                    Data_Dic['Y'].append(Y_[i])
                    if self.item_ft:
                        Data_Dic['Item'].append(im_dic[tweet])
                    if self.personality_ft:
                        Data_Dic['User'].append(self.traits_index[user])

        else:
            print 'Starting loading data:', mode
            with open('filename_'+mode+'.pickle', 'rb') as handle:
                Data_Dic = pickle.load(handle)
            print 'End loading data:', mode
        return Data_Dic
