import numpy as np
import cPickle as pickle
from random import shuffle
import math


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']


def cifar_generator(filenames, batch_size, data_dir, classes_to_load=None, mode_frac=None):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if classes_to_load is not None:
        index_list = None
        for cl_ind, cl in enumerate(classes_to_load):
            class_ind = np.squeeze(np.nonzero(labels == cl))
            shuffle(class_ind)
            nsamples = len(class_ind)
            nsamples_to_load = int(math.ceil(mode_frac[cl_ind]*nsamples))
            if index_list is None:
                index_list = class_ind[0:nsamples_to_load]
            else:
                index_list = np.concatenate((index_list, class_ind[0:nsamples_to_load]), axis=0)
        images = images[index_list, ::]
        labels = labels[index_list]
    
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir, classes_to_load=None, mode_frac=None):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'],
                        batch_size, data_dir, classes_to_load, mode_frac),
        cifar_generator(['test_batch'], batch_size, data_dir, classes_to_load, mode_frac)
    )
