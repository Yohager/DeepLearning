# --*-- coding:utf-8 --*--
#author:yohager

#this file is the main function process of NN

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import NN_class
import activation_function
import NN_forward
import NN_propagation


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
    #这里的model传入的是网络模型model[0]:输入层，model[1]:隐藏层1，model[2]:输出层
    #先走一次前向传播
    NN_forward.full_connected(model,data)
    cost = np.sum((model[-1].a_value - label)**2)
    #print('误差值为：',cost)
    NN_propagation.last_layer_error(model[-1],label,'RSE')
    model[-1].weights -= alpha * model[-1].delta_weights / model[-1].input.shape[1]
    #print(np.average(model[-1].error,axis=1).shape,model[-1].bias.shape)
    model[-1].bias -= alpha * np.average(model[-1].error,axis=1).reshape(len(model[-1].error),1)
    for i in range(2,len(model)):
        NN_propagation.full_connected_back(model[-i],model[-i+1])
    for j in range(1,len(model)):
        model[j].weights -= alpha * model[j].delta_weights /model[j].input.shape[1]
        model[j].bias -= alpha * np.average(model[j].error,axis=1).reshape(len(model[j].error),1)
        #print(model[j].bias.shape)
    #return cost



def test_process(model,test_data,test_label):
    total_num = 10000
    true_number = 0
    NN_forward.full_connected(model,test_data)
    result = np.argmax(model[-1].a_value,axis=0).reshape(total_num,1)
    true_label = np.array([np.argmax(y) for y in test_label.T]).reshape(total_num,1)
    for i in range(total_num):
        if result[i] == true_label[i]:
            true_number += 1
    accuracy = true_number / total_num
    return accuracy


if __name__ == '__main__':
    train_data,train_label,test_data,test_label = load_data('MINST')
    '''
    data_test = train_data[:5].T
    #print(data_test.shape)
    data_test_label = train_label[:5].T
    #print(data_test_label)
    layer_0 = NN_class.input_layer(data_size=784,batch_size=5)
    layer_1 = NN_class.full_connected_layer(input=layer_0,layer_size=15,activation='Sigmoid')
    layer_2 = NN_class.full_connected_layer(input=layer_1,layer_size=10,activation='Sigmoid')
    model = [layer_0,layer_1,layer_2]
    for i in range(100):
        train_process(model,data_test,data_test_label,0.01)
        print(layer_2.z_value)
    '''
    training_set = list(zip(train_data,train_label))
    total_number = len(training_set)
    input_number = train_data.shape[1]
    batch_size = 100
    np.random.shuffle(training_set)
    train_data_shuffle = np.array(list(zip(*training_set))[0])
    train_label_shuffle = np.array(list(zip(*training_set))[1])
    mini_batches_data = mini_batch_generate(train_data_shuffle,batch_size)
    mini_batches_label = mini_batch_generate(train_label_shuffle,batch_size)
    #给出网络的基本结构
    layer_0 = NN_class.input_layer(data_size=784,batch_size=batch_size)
    layer_1 = NN_class.full_connected_layer(input=layer_0,layer_size=100,activation='Sigmoid')
    layer_2 = NN_class.full_connected_layer(input=layer_1,layer_size=50,activation='Sigmoid')
    layer_3 = NN_class.full_connected_layer(input=layer_2,layer_size=10,activation='Sigmoid')
    model = [layer_0,layer_1,layer_2,layer_3]
    error = []
    accuracy_list = []
    for iteration in range(100):
        for i in range(len(mini_batches_data)):
            data = mini_batches_data[i].T
            label = mini_batches_label[i].T
            error_1 = train_process(model,data,label,0.5)
            error.append(error_1)
        print(test_process(model,test_data.T,test_label.T))

















