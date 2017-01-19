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

test_img, test_label = load_image('./datasets/%s/val/%d.jpg' % (args.dataset, 1))
test_img = img_shift(test_img)
test_label = img_shift(test_label)

img_shape, label_shape = data.get_shape()

image_width = img_shape[0]
image_height = img_shape[1]
input_channel = label_shape[2]
output_channel = img_shape[2]
##############################################
# Generator
# U-NET
# 
# CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
# 
##############################################

tmp_x = tf.placeholder(tf.float32, [batch_size, image_width, image_height, input_channel])

D = create_netD(image_width, image_height, input_channel+output_channel, ndf, args.filter)
dec_output, generated_img, encoder_decoder = create_netG(train_X, tmp_x, ngf, args.filter, image_width, image_height, input_channel, output_channel, batch_size)

# ## Objective function

loss_d = tf.reduce_mean(tf.log(D(concat(train_X, train_y)) + 1e-12)) + tf.reduce_mean(tf.log(1 - D(concat(train_X, dec_output)) + 1e-12))

loss_g_1 = tf.reduce_mean(tf.log(1 - D(concat(train_X, dec_output)) + 1e-12))
loss_g_2 = tf.reduce_mean(tf.abs(train_y - dec_output))
loss_g = loss_g_1 + 100. * loss_g_2


# ## Optimizer

train_d = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(-loss_d, var_list=D.trainable_weights)
train_g = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_g, var_list=[op for l in map(lambda x: x.trainable_weights, encoder_decoder) for op in l])


# ## Initialize

sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
data.start_threads(sess)
saver = tf.train.Saver()
mkdir('./model')
# # Training 

print 'start training'
widgets = ['Train: ', Percentage(), '(', SimpleProgress(), ') ',Bar(marker='#', left='[', right=']'), ' ', ETA()]

for i in range(nb_epochs):
    ave_d = []
    ave_g = []

    pbar = ProgressBar(widgets=widgets, maxval=data.get_size() - 1 )
    pbar.start()

    for j in range(data.get_size() - 1):
        sess.run(train_d, feed_dict={K.learning_phase(): 1})
        sess.run(train_g, feed_dict={K.learning_phase(): 1})
        
        loss_d_val = sess.run(loss_d, feed_dict={K.learning_phase(): 1})
        ave_d.append(loss_d_val)
        ave_g.append(sess.run(loss_g, feed_dict={K.learning_phase(): 1}))
        time.sleep(0.001)
        pbar.update(j)
    pbar.finish()

    print "Epoch %d/%d - dis_loss: %g - gen_loss: %g" % (i+1, nb_epochs, np.mean(ave_d), np.mean(ave_g))
    generated_image = sess.run(generated_img, feed_dict={tmp_x: [test_label], K.learning_phase(): 1})
    save_image(generated_image[0], args.out + '/' + args.dataset , i+1)
    saver.save(sess, './model/{}/model.ckpt'.format(args.dataset), global_step=i+1)


