# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras import initializations
from keras.layers import Activation, Flatten, Dropout, merge
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name)

def create_netD(image_width, image_height, input_channel, ndf, filter_size):
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
    patchGAN.add(Convolution2D(ndf, filter_size, filter_size, 
                               subsample=(2, 2),
                               border_mode='same',
                               init=my_init,
                               input_shape=(image_width, image_height, input_channel)))
    patchGAN.add(LeakyReLU(alpha=0.2))

    # C128 128=>64
    patchGAN.add(Convolution2D(ndf * 2, filter_size, filter_size, 
                               subsample=(2, 2),
                               init=my_init,
                               border_mode='same',
                              ))
    patchGAN.add(BatchNormalization())
    patchGAN.add(LeakyReLU(alpha=0.2))

    # C256 64=>32
    patchGAN.add(Convolution2D(ndf * 4, filter_size, filter_size, 
                               subsample=(2, 2),
                               init=my_init,
                               border_mode='same',
                              ))
    patchGAN.add(BatchNormalization())
    patchGAN.add(LeakyReLU(alpha=0.2))

    # C512 32=>16
    patchGAN.add(Convolution2D(ndf * 8, filter_size, filter_size, 
                               subsample=(1, 1),
                               init=my_init,
                               border_mode='same',
                              ))
    patchGAN.add(BatchNormalization())
    patchGAN.add(LeakyReLU(alpha=0.2))

    patchGAN.add(Convolution2D(1, filter_size, filter_size, 
                               subsample=(1, 1),
                               init=my_init,
                               border_mode='same',
                              ))
    patchGAN.add(Activation('sigmoid'))
    patchGAN.add(Flatten())
    return patchGAN

def create_netG(train_X, tmp_x, ngf, filter_size, image_width, image_height, input_channel, output_channel, batch_size):
    encoder_decoder = []
    # encoder
    # C64 256=>128
    enc_conv0 = Convolution2D(ngf, filter_size, filter_size, 
                      subsample=(2, 2), 
                      border_mode='same', 
                      init=my_init,  
                      input_shape=(image_width, image_height, input_channel)
                     )
    enc_output0 = enc_conv0(train_X)
    enc_output0_ = enc_conv0(tmp_x)

    encoder_decoder.append(enc_conv0)

    # C128 128=>64
    enc_conv1 = Convolution2D(ngf * 2, filter_size, filter_size, 
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
    enc_conv2 = Convolution2D(ngf * 4, filter_size, filter_size, 
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
    enc_conv3 = Convolution2D(ngf * 8, filter_size, filter_size, 
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
    enc_conv4 = Convolution2D(ngf * 8, filter_size, filter_size, 
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
    enc_conv5 = Convolution2D(ngf * 8, filter_size, filter_size, 
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
    enc_conv6 = Convolution2D(ngf * 8, filter_size, filter_size, 
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
    enc_conv7 = Convolution2D(ngf * 8, filter_size, filter_size, 
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
    dec_conv0 = Deconvolution2D(ngf * 8, filter_size, filter_size, 
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
    dec_conv1 = Deconvolution2D(ngf * 8, filter_size, filter_size, 
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
    dec_conv2 = Deconvolution2D(ngf * 8, filter_size, filter_size, 
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
    dec_conv3 = Deconvolution2D(ngf * 8, filter_size, filter_size, 
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
    dec_conv4 = Deconvolution2D(ngf * 4, filter_size, filter_size, 
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
    dec_conv5 = Deconvolution2D(ngf * 2, filter_size, filter_size, 
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
    dec_conv6 = Deconvolution2D(ngf, filter_size, filter_size, 
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
    dec_conv7 = Deconvolution2D(output_channel, filter_size, filter_size, 
                                output_shape=(batch_size, image_width, image_height, output_channel),
                                subsample=(2, 2),
                                init=my_init,
                                border_mode='same'
                        )
    dec_tanh = Activation('tanh')
    encoder_decoder += [dec_conv7]

    dec_output = dec_tanh(dec_conv7(relu6(dec_output6)))
    generated_img = dec_tanh(dec_conv7(relu6(dec_output6_)))
    return dec_output, generated_img, encoder_decoder
