# -*- coding: utf-8 -*-

import threading


from ops import *
import os
from glob import glob

import numpy as np
import tensorflow as tf

def download(dataset_name):
    datasets_dir = './datasets/'
    mkdir(datasets_dir)
    URL='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz' % (dataset_name)
    TAR_FILE='./datasets/%s.tar.gz' % (dataset_name)
    TARGET_DIR='./datasets/%s/' % (dataset_name)
    os.system('wget -N %s -O %s' % (URL, TAR_FILE))
    os.mkdir(TARGET_DIR)
    os.system('tar -zxf %s -C ./datasets/' % (TAR_FILE))
    os.remove(TAR_FILE)


class Dataset(object):
    def __init__(self, dataset, is_test=False, batch_size=4, crop_width=256, thread_num=1):
        self.batch_size = batch_size
        self.thread_num = thread_num
        print "Batch size: %d, Thread num: %d" % (batch_size, thread_num)
        datasetDir = './datasets/{}'.format(dataset)
        if not os.path.isdir(datasetDir):
            download(dataset)
        dataDir = datasetDir + '/train'
        data = glob((dataDir + '/*.jpg').format(dataset))
        self.data_size = min(400, len(data))
        self.data_indice = range(self.data_size - 1)
        self.dataDir = dataDir
        self.is_test = is_test
        self.dataset = []
        for i in range(1, self.data_size):
            img, label = load_image(self.dataDir + '/%d.jpg' % i)

            self.dataset.append((img, label))
        print "load dataset done"
        print 'data size: %d' % len(self.dataset)
        img_shape = list(self.dataset[0][0].shape)
        label_shape = list(self.dataset[0][1].shape)
        self.fine_size = img_shape[0]
        self.crop_width = self.fine_size
        self.load_size = self.fine_size + 30
        
        self.img_data = tf.placeholder(tf.float32, shape=[None] + img_shape)
        self.label_data = tf.placeholder(tf.float32, shape=[None] + label_shape)
        self.queue = tf.FIFOQueue(shapes=[label_shape, img_shape],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=2000)
        self.enqueue_ops = self.queue.enqueue_many([self.label_data, self.img_data])

        
    def batch_iterator(self):
        while True:
            shuffle_indices = np.random.permutation(self.data_indice)
            for i in range(len(self.data_indice) / self.batch_size):
                img_batch = []
                label_batch = []
                for j in range(i*self.batch_size, (i+1)*self.batch_size):
                    label = self.dataset[shuffle_indices[j]][1]
                    img = self.dataset[shuffle_indices[j]][0]
                    img, label = img_preprocess(img, label, self.fine_size, self.load_size )
                    label_batch.append(label)
                    img_batch.append(img)
                yield np.array(label_batch), np.array(img_batch)
    
    def get_inputs(self):
        labels, imgs = self.queue.dequeue_many(self.batch_size)
        return labels, imgs
    
    def thread_main(self, sess):
        for labels, imgs in self.batch_iterator():
            sess.run(self.enqueue_ops, feed_dict={self.label_data: labels , self.img_data: imgs})
            sess.run(self.enqueue_ops, feed_dict={self.label_data: labels , self.img_data: imgs})
            sess.run(self.enqueue_ops, feed_dict={self.label_data: labels , self.img_data: imgs})
    
    def start_threads(self, sess):
        threads = []
        for n in range(self.thread_num):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads
    
    def get_size(self):
        return self.data_size
    
