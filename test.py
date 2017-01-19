# coding: utf-8


import threading
import argparse
import progressbar
import time

from progressbar import Bar, ETA, Percentage, ProgressBar, SimpleProgress

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import backend as K

from dataset import Dataset
from ops import *
from model import create_netD, create_netG, my_init


np.random.seed(1234)
tf.set_random_seed(1234)

sess = tf.Session()
K.set_session(sess)


# Parameters
parser = argparse.ArgumentParser(description='Training Pix2Pix Model')
parser.add_argument('--dataset', '-d', default='facades', help='Select the datasets from facades')
parser.add_argument('--out', '-o', default='./output_imgs', help='Directory path for generated images')
parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
parser.add_argument('--learningrate', '-l', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1; momentum')
parser.add_argument('--epoch', '-e', type=int, default=200, help='Epoch')
parser.add_argument('--thread', '-t', type=int, default=1, help='num of thread')
parser.add_argument('--filter', '-f', type=int, default=4, help='kernel/filter size')

args = parser.parse_args()

mkdir(args.out)

ngf = 64
ndf = 64
batch_size = args.batchsize
nb_epochs = args.epoch

data = Dataset(dataset=args.dataset, batch_size=batch_size, thread_num=args.thread)

train_X, train_y = data.get_inputs()

img_shape, label_shape = data.get_shape()

test_img, test_label = load_image('./datasets/%s/val/%d.jpg' % (args.dataset, 1))

image_width = img_shape[0]
image_height = img_shape[1]
input_channel = label_shape[2]
output_channel = img_shape[2]

tmp_x = tf.placeholder(tf.float32, [batch_size, image_width, image_height, input_channel])

D = create_netD(image_width, image_height, input_channel+output_channel, ndf, args.filter)
dec_output, generated_img, encoder_decoder = create_netG(train_X, tmp_x, ngf, args.filter, image_width, image_height, input_channel, output_channel, batch_size)


# ## Initialize

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
data.start_threads(sess)
saver = tf.train.Saver()
# # Training 

print 'start training'
widgets = ['Train: ', Percentage(), '(', SimpleProgress(), ') ',Bar(marker='#', left='[', right=']'), ' ', ETA()]

for i in range(nb_epochs):
    ave_d = []
    ave_g = []

    pbar = ProgressBar(widgets=widgets, maxval=data.get_size() - 1 )
    pbar.start()

    for j in range(data.get_size() - 1):
    generated_image = sess.run(generated_img, feed_dict={tmp_x: [test_label], K.learning_phase(): 1})
    save_image(generated_image[0], args.out + '/' + args.dataset , i+1)
        time.sleep(0.001)
        pbar.update(j)
    pbar.finish()



