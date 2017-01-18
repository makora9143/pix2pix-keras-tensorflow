# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt

def mkdir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    return



def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    return scipy.misc.imread(path).astype(np.float)
    
    
def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]
    return img_A, img_B


def show_image(img):
    plt.imshow(np.asarray(np.clip((img + 1.)*127.5, 0., 255.), dtype=np.uint8))
    plt.show()


def save_image(img, filedir, i):
    plt.imsave(filedir + '/epoch-%d.jpg' % i, np.asarray(np.clip((img + 1.)*127.5, 0., 255.), dtype=np.uint8))


def img_preprocess(img, label, fine_size, load_size, is_test=False):
    if is_test:
        img = scipy.misc.imresize(img, [fine_size, fine_size])
        label = scipy.misc.imresize(label, [fine_size, fine_size])
    else:
        img = scipy.misc.imresize(img, [load_size, load_size])
        label = scipy.misc.imresize(label, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img = img[h1: h1 + fine_size, w1: w1 + fine_size]
        label = label[h1: h1 + fine_size, w1: w1 + fine_size]
        if np.random.random() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
    img = img_shift(img)
    label = img_shift(label)
    return img, label
        

def img_shift(img):
    return img / 127.5 - 1.

def concat(x, y):
    return tf.concat(3, [x, y])
