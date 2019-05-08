# --*-- coding:utf-8 --*--
#author:yohager

#this file is the main function of CNN

import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import CNN_forward
import CNN_class
import CNN_backward
from tensorflow.examples.tutorials.mnist import input_data
import activation_function

def load_data(filename):
    data_set = input_data.read_data_sets(filename,one_hot=True)
    train_data,train_label,test_data,test_label = data_set.train.images,data_set.train.labels,data_set.test.images,data_set.test.labels
    return train_data,train_label,test_data,test_label

def mini_batch_generate(data,size):
    mini_batches = []
    total_number = len(data)
    for i in range(0,total_number,size):
        mini_batches.append(data[i:i+size])
    return mini_batches

def train_process(model,data,label,alpha):
    #先进行前向传播计算当前的误差
    CNN_forward.CNN_forward_function(model,data)
    cost = np.sum((model[-1].a_value - label)**2)
    CNN_backward.CNN_backward_function(model,label,0.01,'RSE')
    return cost

def test_process(model, test_data, test_label):
    total_num = 10000
    true_number = 0
    CNN_forward.CNN_forward_function(model, test_data)
    result = np.argmax(model[-1].a_value, axis=0).reshape(total_num, 1)
    true_label = np.array([np.argmax(y) for y in test_label.T]).reshape(total_num, 1)
    for i in range(total_num):
        if result[i] == true_label[i]:
            true_number += 1
    accuracy = true_number / total_num
    return accuracy



    #留出空间用于数据的预处理
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    train_data, train_label, test_data, test_label = load_data('MNIST')
    training_set = list(zip(train_data,train_label))
    total_number = len(training_set)
    input_number = train_data.shape[1]
    batch_size = 100
    #print(mini_batches_data[0].shape)
    #上面的函数生成了batch的数据
    layer_0 = CNN_class.input_layer(28, batch_size)
    layer_1 = CNN_class.conv_2d_layer(input=layer_0, filter_size=5, depth=10, zero_padding=0, stride=1,activation='Sigmoid')
    layer_2 = CNN_class.pooling_layer(input=layer_1, pooling_size=2, pooling_type='max')
    layer_3 = CNN_class.full_connected_layer(input=layer_2, layer_size=50, activation='Sigmoid')
    layer_4 = CNN_class.full_connected_layer(input=layer_3, layer_size=10, activation='Sigmoid')
    model = [layer_0, layer_1, layer_2, layer_3,layer_4]
    accuracy_list = []
    for iteration in range(10):
        np.random.shuffle(training_set)
        train_data_shuffle = np.array(list(zip(*training_set))[0])
        train_label_shuffle = np.array(list(zip(*training_set))[1])
        mini_batches_data = mini_batch_generate(train_data_shuffle, batch_size)
        mini_batches_label = mini_batch_generate(train_label_shuffle, batch_size)
        for i in range(50):
            data = mini_batches_data[i].reshape(batch_size,28,28)
            label = mini_batches_label[i].T
            cost = train_process(model,data,label,0.1)
            print('cost is %f'%(cost))
        print('accuracy:',test_process(model,test_data.reshape(10000,28,28),test_label.T))
        accuracy_list.append(test_process(model,test_data.reshape(10000,28,28),test_label.T))