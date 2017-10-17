import numpy as np
from numpy import genfromtxt


def read_traits(file):
    traits = genfromtxt(file, delimiter=',', dtype='float16')
    traits = np.delete(traits, 0, axis=1)
    traits = np.delete(traits, 0, axis=0)
    return traits[:, :5]


def read_items(file):
    items = set()
    with open(file, 'r') as f:
        lines = f.readlines()
    for l in lines:
        for i in [el.split()[0] for el in l.split(',')[1:]]:
            items.add(i)
    return items


def build_itemlist(ratings1, ratings2, error_imgs_path, data_folder):
    with open(data_folder + error_imgs_path, 'r') as f:
        error_imgs = f.readlines()
    total_set = read_items(data_folder + ratings1).union(read_items(data_folder + ratings2))
    to_remove_set = set([e.replace('\n', '') for e in error_imgs])
    item_list = list(total_set.difference(to_remove_set))
    item_index = {}
    with open('data/item_index.txt', 'w') as w:
        for i, u in enumerate(item_list):
            item_index[u] = i
            w.write(str(i) + ' ' + u + '\n')
    return item_index


def build_userlist(file, users_discard, data_folder):
    with open(data_folder + file, 'r') as f:
        lines = f.readlines()
    user_list = [u for u in [l.split(',')[0].lower() for l in lines][1:] if u not in users_discard]
    user_index = {}
    with open(data_folder + 'user_index.txt', 'w') as w:
        for i, u in enumerate(user_list):
            user_index[u] = i
            w.write(str(i) + ' ' + u + '\n')
    return user_index

def read_image_index(file):
    with open(file, 'r') as f:
        index = f.readlines()
    return [i.split('.')[0] for i in index]