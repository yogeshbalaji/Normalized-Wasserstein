import numpy as np
from random import shuffle
import os
import cPickle as pickle
import numpy 
import cv2


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']


def cifar_generator(filenames, celeba_data, batch_size, data_dir, mode_frac=None):
    
    # reshaping CelebA
    celeba_images_reshaped = []
    for i in range(len(celeba_data)):
        img = celeba_data[i]
        img = img.astype(np.float32)
        img = img[:, :, ::-1]
        img = numpy.transpose(img, (2, 0, 1)) 
        celeba_images_reshaped.append(img)
    celeba_images_reshaped = np.array(celeba_images_reshaped)
    celeba_images_reshaped = celeba_images_reshaped.reshape(celeba_images_reshaped.shape[0], -1)

    if mode_frac!=None:
        num_celeba = int(celeba_images_reshaped.shape[0]*mode_frac[1])
        celeba_images_reshaped = celeba_images_reshaped[0:num_celeba, ::]
    celeba_targets = np.zeros((celeba_images_reshaped.shape[0],)).astype(np.int)

    print(celeba_images_reshaped.shape)
    print('Loading {} CelebA images'.format(celeba_images_reshaped.shape[0]))
    
    
    # reading CIFAR
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if mode_frac!=None:
        shuffle(images)
        shuffle(labels)
        num_cifar = int(images.shape[0]*mode_frac[0])
        images = images[0:num_cifar, ::]
        labels = labels[0:num_cifar]
    
    print('Loading {} CIFAR-10 images'.format(images.shape[0]))
    
    print('Ranges')
    print(celeba_images_reshaped.min())
    print(celeba_images_reshaped.max())
    print(images.min())
    print(images.max())
    # concatenating MNIST and CIFAR
    images = np.concatenate((images, celeba_images_reshaped), axis=0)
    labels = np.concatenate((labels, celeba_targets), axis=0)
    
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load_celeba(celeba_path, num_files_to_load):

    filelist = os.listdir(celeba_path)
    shuffle(filelist)
    filelist = filelist[0:num_files_to_load]

    img_all = []
    for f in filelist:
        fpath = os.path.join(celeba_path, f)
        img = cv2.imread(fpath)
        img_all.append(img)

    return img_all


def load(batch_size, cifar_dir, celeba_dir, mode_frac=None):

    # loading CelebA in RAM
    celeba_imgs = load_celeba(celeba_dir, 100000)
    celeba_test = load_celeba(celeba_dir, 20000)

    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'],
                        celeba_imgs, batch_size, cifar_dir, mode_frac),
        cifar_generator(['test_batch'], celeba_test, batch_size, cifar_dir, mode_frac)
    )
