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
from keras import initializations
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dropout, merge
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from dataset import Dataset
from ops import *


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


image_width = 256

image_height = 256
input_channel = 3
output_channel = 3
ngf = 64
batch_size = args.batchsize
nb_epochs = args.epoch

data = Dataset(dataset=args.dataset, batch_size=batch_size, thread_num=args.thread)

train_X, train_y = data.get_inputs()

test_img, test_label = load_image('./datasets/%s/val/%d.jpg' % (args.dataset, 1))
test_img = img_shift(test_img)
test_label = img_shift(test_label)

##############################################
# Generator
# U-NET
# 
# CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
# 
##############################################


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name)

tmp_x = tf.placeholder(tf.float32, [batch_size, image_width, image_height, input_channel])

encoder_decoder = []
# encoder
# C64 256=>128
enc_conv0 = Convolution2D(ngf, args.filter, args.filter, 
                  subsample=(2, 2), 
                  border_mode='same', 
                  init=my_init,  
                  input_shape=(image_width, image_height, input_channel)
                 )
enc_output0 = enc_conv0(train_X)
enc_output0_ = enc_conv0(tmp_x)

encoder_decoder.append(enc_conv0)

# C128 128=>64
enc_conv1 = Convolution2D(ngf * 2, args.filter, args.filter, 
                  subsample=(2, 2), 
                     init=my_init,
                  border_mode='same'
                 )
enc_bn1 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu1 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv1, enc_bn1]

enc_output1 = enc_bn1(enc_conv1(leaky_relu1(enc_output0)))
enc_output1_ = enc_bn1(enc_conv1(leaky_relu1(enc_output0_)))

# C256 64=>32
enc_conv2 = Convolution2D(ngf * 4, args.filter, args.filter, 
                  subsample=(2, 2), 
                          init=my_init,
                  border_mode='same'
                 )
enc_bn2 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu2 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv2, enc_bn2]

enc_output2 = enc_bn2(enc_conv2(leaky_relu2(enc_output1)))
enc_output2_ = enc_bn2(enc_conv2(leaky_relu2(enc_output1_)))

# C512 32=>16
enc_conv3 = Convolution2D(ngf * 8, args.filter, args.filter, 
                  subsample=(2, 2), 
                          init=my_init,
                  border_mode='same'
                  )
enc_bn3 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu3 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv3, enc_bn3]
enc_output3 = enc_bn3(enc_conv3(leaky_relu3(enc_output2)))
enc_output3_ = enc_bn3(enc_conv3(leaky_relu3(enc_output2_)))

# C512 16=>8
enc_conv4 = Convolution2D(ngf * 8, args.filter, args.filter, 
                  subsample=(2, 2), 
                          init=my_init,
                  border_mode='same'
                  )
enc_bn4 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu4 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv4, enc_bn4]
enc_output4 = enc_bn4(enc_conv4(leaky_relu4(enc_output3)))
enc_output4_ = enc_bn4(enc_conv4(leaky_relu4(enc_output3_)))
# C512 8=>4
enc_conv5 = Convolution2D(ngf * 8, args.filter, args.filter, 
                  subsample=(2, 2), 
                          init=my_init,
                  border_mode='same'
                  )
enc_bn5 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu5 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv5, enc_bn5]
enc_output5 = enc_bn5(enc_conv5(leaky_relu5(enc_output4)))
enc_output5_ = enc_bn5(enc_conv5(leaky_relu5(enc_output4_)))

# C512 4=>2
enc_conv6 = Convolution2D(ngf * 8, args.filter, args.filter, 
                  subsample=(2, 2), 
                          init=my_init,
                  border_mode='same'
                  )
enc_bn6 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu6 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv6, enc_bn6]

enc_output6 = enc_bn6(enc_conv6(leaky_relu6(enc_output5)))
enc_output6_ = enc_bn6(enc_conv6(leaky_relu6(enc_output5_)))



# C512 2=>1
enc_conv7 = Convolution2D(ngf * 8, args.filter, args.filter, 
                  subsample=(2, 2), 
                          init=my_init,
                  border_mode='same'
                  )
enc_bn7 = BatchNormalization(epsilon=1e-5, momentum=0.9)
leaky_relu7 = LeakyReLU(alpha=0.2)

encoder_decoder += [enc_conv7, enc_bn7]

enc_output7 = enc_bn7(enc_conv7(leaky_relu7(enc_output6)))
enc_output7_ = enc_bn7(enc_conv7(leaky_relu7(enc_output6_)))

# decoder
#CD512 1=>2
dec_conv0 = Deconvolution2D(ngf * 8, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 128, image_height / 128, ngf * 8),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn0 = BatchNormalization(epsilon=1e-5, momentum=0.9)
dropout0 = Dropout(0.5)
relu0 = Activation('relu')

encoder_decoder += [dec_conv0, dec_bn0]
dec_output0 = dropout0(dec_bn0(dec_conv0(relu0(enc_output7))))

dec_output0 = merge([dec_output0, enc_output6], mode='concat')

dec_output0_ = dropout0(dec_bn0(dec_conv0(relu0(enc_output7_))))

dec_output0_ = merge([dec_output0_, enc_output6_], mode='concat')


#CD512 2=>4
dec_conv1 = Deconvolution2D(ngf * 8, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 64, image_height / 64, ngf * 8),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn1 = BatchNormalization(epsilon=1e-5, momentum=0.9)
dropout1 = Dropout(0.5)
relu1 = Activation('relu')

encoder_decoder += [dec_conv1, dec_bn1]
dec_output1 = dropout1(dec_bn1(dec_conv1(relu1(dec_output0))))

dec_output1 = merge([dec_output1, enc_output5], mode='concat')

dec_output1_ = dropout1(dec_bn1(dec_conv1(relu1(dec_output0_))))

dec_output1_ = merge([dec_output1_, enc_output5_], mode='concat')

#CD512 4=>8
dec_conv2 = Deconvolution2D(ngf * 8, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 32, image_height / 32, ngf * 8),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn2 = BatchNormalization(epsilon=1e-5, momentum=0.9)
dropout2 = Dropout(0.5)
relu2 = Activation('relu')

encoder_decoder += [dec_conv2, dec_bn2]
dec_output2 = dropout2(dec_bn2(dec_conv2(relu2(dec_output1))))

dec_output2 = merge([dec_output2, enc_output4], mode='concat')

dec_output2_ = dropout2(dec_bn2(dec_conv2(relu2(dec_output1_))))

dec_output2_ = merge([dec_output2_, enc_output4_], mode='concat')

#C512 8=>16
dec_conv3 = Deconvolution2D(ngf * 8, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 16, image_height / 16, ngf * 8),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn3 = BatchNormalization(epsilon=1e-5, momentum=0.9)
relu3 = Activation('relu')

encoder_decoder += [dec_conv3, dec_bn3]
dec_output3 = dec_bn3(dec_conv3(relu3(dec_output2)))

dec_output3 = merge([dec_output3, enc_output3], mode='concat')

dec_output3_ = dec_bn3(dec_conv3(relu3(dec_output2_)))

dec_output3_ = merge([dec_output3_, enc_output3_], mode='concat')

#C256 16=>32
dec_conv4 = Deconvolution2D(ngf * 4, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 8, image_height / 8, ngf * 4),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn4 = BatchNormalization(epsilon=1e-5, momentum=0.9)
relu4 = Activation('relu')

encoder_decoder += [dec_conv4, dec_bn4]
dec_output4 = dec_bn4(dec_conv4(relu4(dec_output3)))

dec_output4 = merge([dec_output4, enc_output2], mode='concat')

dec_output4_ = dec_bn4(dec_conv4(relu4(dec_output3_)))

dec_output4_ = merge([dec_output4_, enc_output2_], mode='concat')



#C128 32=>64
dec_conv5 = Deconvolution2D(ngf * 2, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 4, image_height / 4, ngf * 2),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn5 = BatchNormalization(epsilon=1e-5, momentum=0.9)
relu5 = Activation('relu')
encoder_decoder += [dec_conv5, dec_bn5]
dec_output5 = dec_bn5(dec_conv5(relu5(dec_output4)))


dec_output5 = merge([dec_output5, enc_output1], mode='concat')

dec_output5_ = dec_bn5(dec_conv5(relu5(dec_output4_)))

dec_output5_ = merge([dec_output5_, enc_output1_], mode='concat')

#C64 64=>128
dec_conv6 = Deconvolution2D(ngf, args.filter, args.filter, 
                    output_shape=(batch_size, image_width / 2, image_height / 2, ngf),
                    subsample=(2, 2), 
                            init=my_init,
                    border_mode='same'
                    )
dec_bn6 = BatchNormalization(epsilon=1e-5, momentum=0.9)
relu6 = Activation('relu')

encoder_decoder += [dec_conv6, dec_bn6]
dec_output6 = dec_bn6(dec_conv6(relu6(dec_output5)))
dec_output6 = merge([dec_output6, enc_output0], mode='concat')

dec_output6_ = dec_bn6(dec_conv6(relu6(dec_output5_)))
dec_output6_ = merge([dec_output6_, enc_output0_], mode='concat')

#C3 128=>256 last layer tanh
dec_conv7 = Deconvolution2D(output_channel, args.filter, args.filter, 
                            output_shape=(batch_size, image_width, image_height, output_channel),
                            subsample=(2, 2),
                            init=my_init,
                            border_mode='same'
                    )
dec_tanh = Activation('tanh')
encoder_decoder += [dec_conv7]

dec_output = dec_tanh(dec_conv7(relu6(dec_output6)))
generated_img = dec_tanh(dec_conv7(relu6(dec_output6_)))



#########################################
# Discriminator
#
# PatchGAN (Sequential)
# 
# C64-C128-C256-C512
# 
#########################################


patchGAN = Sequential()

# C64 256=>128
patchGAN.add(Convolution2D(ngf, args.filter, args.filter, 
                           subsample=(2, 2),
                           border_mode='same',
                           init=my_init,
                           input_shape=(image_width, image_height, input_channel+output_channel)))
patchGAN.add(LeakyReLU(alpha=0.2))

# C128 128=>64
patchGAN.add(Convolution2D(ngf * 2, args.filter, args.filter, 
                           subsample=(2, 2),
                           init=my_init,
                           border_mode='same',
                          ))
patchGAN.add(BatchNormalization())
patchGAN.add(LeakyReLU(alpha=0.2))

# C256 64=>32
patchGAN.add(Convolution2D(ngf * 4, args.filter, args.filter, 
                           subsample=(2, 2),
                           init=my_init,
                           border_mode='same',
                          ))
patchGAN.add(BatchNormalization())
patchGAN.add(LeakyReLU(alpha=0.2))

# C512 32=>16
patchGAN.add(Convolution2D(ngf * 8, args.filter, args.filter, 
                           subsample=(1, 1),
                           init=my_init,
                           border_mode='same',
                          ))
patchGAN.add(BatchNormalization())
patchGAN.add(LeakyReLU(alpha=0.2))

patchGAN.add(Convolution2D(1, args.filter, args.filter, 
                           subsample=(1, 1),
                           init=my_init,
                           border_mode='same',
                          ))
patchGAN.add(Activation('sigmoid'))
patchGAN.add(Flatten())


D = patchGAN


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
        sess.run(train_g, feed_dict={K.learning_phase(): 1})
        
        loss_d_val = sess.run(loss_d, feed_dict={K.learning_phase(): 1})
        ave_d.append(loss_d_val)
        ave_g.append(sess.run(loss_g, feed_dict={K.learning_phase(): 1}))
        time.sleep(0.001)
        pbar.update(j)
    pbar.finish()

    print "Epoch %d/%d - dis_loss: %g - gen_loss: %g" % (i+1, nb_epochs, np.mean(ave_d), np.mean(ave_g))
    generated_image = sess.run(generated_img, feed_dict={tmp_x: [test_label], K.learning_phase(): 1})
    save_image(generated_image[0], args.out, i+1)
    saver.save(sess, './model/model.ckpt', global_step=i+1)


