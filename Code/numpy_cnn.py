# -*- coding: utf-8 -*-
"""
@author: yohager
Using numpy to realize cnn neuron network
data_set: MNIST
"""

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time
from tensorflow.examples.tutorials.mnist import input_data

class numpy_cnn(object):
    #conv_shape:存储卷积核的尺寸和卷积核的个数
    def __init__(self,conv_shape):
        self.conv_shape = conv_shape[0]
        self.conv_num = conv_shape[1]

    #load data
    def load_data(self, filepath):
        data_set = input_data.read_data_sets(filepath, one_hot=True)
        train_data, train_label, test_data, test_label = data_set.train.images, data_set.train.labels, data_set.test.images, data_set.test.labels
        #print(train_data.shape)
        return train_data, train_label, test_data, test_label

    #构造卷积层
    def conv_layer(self,image,conv_shape):
        conv_kernel = np.zeros((conv_shape[1],conv_shape[0],conv_shape[0]))
        print(conv_kernel)
        img_shape = image.shape
        #书写体28*28




if __name__ == '__main__':
    cnn_test = numpy_cnn([3,3])
    #train_data,train_label,test_data,test_label = cnn_test.load_data('MNIST.data')
    #print(train_data.shape)
    cnn_test.conv_layer([3,3])