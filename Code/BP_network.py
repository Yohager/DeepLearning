# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11
@author: yohager
"""
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# 首先写一个bp网络
class BackPropagation_NN(object):
    # 使用size定义神经网络每一层的神经元数量
    def __init__(self, size):
        #size的长度表示层数，size每一个元素的值表示为每一层的神经元数个数
        self.layer_num = len(size)
        self.sizes = size
        #bias表示偏置，即每一层神经元多少个偏置多少个,全部为列向量存放在一个list中
        self.bias = [np.random.randn(y,1) for y in size[1:]]
        #weights表示权重，第i层（n个神经元）与第i+1层（m个神经元）之间的权重可以表示为一个矩阵。维度:m*n（相当于表示为列向量的形式进行组合）
        self.weights = [np.random.randn(y,x) for y,x in zip(size[1:],size[:-1])]


    # 加载数据
    def load_data(self, filepath):
        data_set = input_data.read_data_sets(filepath, one_hot=True)
        train_data, train_label, test_data, test_label = data_set.train.images, data_set.train.labels, data_set.test.images, data_set.test.labels
        #print(train_data.shape)
        return train_data, train_label, test_data, test_label

    def activation_function(self,H_x):
        #这里的激活函数可以任意选取，可以使用sigmoid或者relu
        return 1.0/(1.0+np.exp(-H_x))

    def activation_function_derivative(self,H_x):
        return self.activation_function(H_x) * (1 - self.activation_function(H_x))
    #前向传播
    #验证结果的时候使用
    def feedforward(self,input_data):
        #输入函数为784*1的数据向量，从输入到输出通过weights和bias形成全连接
        for weight,bia in zip(self.weights,self.bias):
            #input data: 784*1,第一层weight:w_1 * 784,dot后w_1 * 1，bias:w_1 * 1
            input_data = self.activation_function(np.dot(weight,input_data).reshape(weight.shape[0],1) + bia)
        return input_data

    #后向传播
    #核心代码：输入是单个的784*1的数据样本，输出是一次
    def back_propagation(self,input_data_x,input_data_y):
        #初始化两个列表用于存放所有权重和偏置的梯度的值的矩阵，以这里的为例，第一层15*784的权重，第二层10*15个权重，这些值用于记录后面的cost derivative of weights
        delta_weights = []
        for i in self.weights:
            delta_weights.append(np.zeros(i.shape))
        #print('第一层的权重shape',delta_weights[0].shape)
        delta_biases = []
        for j in self.bias:
            delta_biases.append(np.zeros(j.shape))
        #print(delta_biases[0].shape)
        #这里的z_list用于存放带权输入，a_list用于存放每一层的激活值
        a_input = input_data_x #中间变量进行运算
        z_list = []
        #这个地方之所以把输入层的值也放入激活输出中是因为在求第2层的梯度的时候需要使用输入层的数据
        a_list = [a_input.reshape(len(a_input),1)]
        for weight,bia in zip(self.weights,self.bias):
            #print(weight.shape)
            z_input = np.dot(weight,a_input).reshape(weight.shape[0],1) + bia
            a_input = self.activation_function(z_input)
            z_list.append(z_input)
            a_list.append(a_input)
        #print(len(z_list),len(a_list))
        #print(z_list[0].shape,z_list[1].shape)
        #print('激活输出的值shape:',a_list[-3].shape)
        #print(input_data_y.shape)
        #进行后向传播的过程，从输出结果层进行倒退，一层一层求解梯度值
        #首先计算最后一层的梯度
        last_layer_error_derivative = sum(a_list[-1] - input_data_y.reshape(len(input_data_y),1)) * self.activation_function_derivative(z_list[-1])
        #last_layer_error_derivative = sum(a_list[-1] - input_data_y.reshape(len(input_data_y),1))
        #print('last_layer_error_shape:',last_layer_error_derivative.shape)
        #分析这里的结果：以本数据为例，最后一层10个神经元，有10*1个激活值，与input_data_y做差乘以2为loss关于最后一层激活值得导数，再乘以激活函数的导数
        delta_biases[-1] = last_layer_error_derivative
        delta_weights[-1] = last_layer_error_derivative.dot(a_list[-2].T)
        #delta_biases[-1].shape = 10*1; delta_weights[-1].shape = 10*15 这里完成了最后一层输出层（L层）的梯度的计算
        #print(delta_biases[-1].shape)
        #print(delta_weights[-1].shape)
        #下面进行之前的从（L-1）层到第一层的梯度的计算
        #print('层数',self.layer_num)
        for layer in range(2,self.layer_num):
            z_layer = z_list[-layer]
            #这一步表示为公式BP2：前一层的误差等于后一层的权重乘以后一层的误差再与前一层的带权输入做Hadamard积
            last_layer_error_derivative = np.dot(self.weights[-layer+1].T,last_layer_error_derivative) * self.activation_function_derivative(z_layer)
            #偏置等于这个误差值，权重等于这个误差值乘以再前一层的
            #print('shape_error:',last_layer_error_derivative.shape)
            delta_biases[-layer] = last_layer_error_derivative
            #print(delta_biases[-layer].shape)
            #print(a_list[-layer-1].shape)
            delta_weights[-layer] = np.dot(last_layer_error_derivative,np.array(a_list[-layer-1]).T)
            #last_layer_error:15*1;a_list[-layer-1]:784*1 => 第一层的权重矩阵：784*15
        #print(len(delta_weights))
        return delta_weights,delta_biases
    #更新函数
    def update_gradient(self,mini_batch,learning_rate):
        #这里的mini_batch是矩阵：mini_batch_size
        delta_weights = []
        for i in self.weights:
            delta_weights.append(np.zeros(i.shape))
        #print('第一层的权重shape',delta_weights[0].shape)
        delta_biases = []
        for j in self.bias:
            delta_biases.append(np.zeros(j.shape))

        for x,y in mini_batch:
            delta_w,delta_b = self.back_propagation(x,y)
            delta_weights = [i + j for i,j in zip(delta_w,delta_weights)]
            delta_biases = [m + n for m,n in zip(delta_b,delta_biases)]
        self.weights = [weight - (learning_rate/len(mini_batch)) * weight_part for weight,weight_part in zip(self.weights,delta_weights)]
        self.bias = [bia -(learning_rate/len(mini_batch)) * bia_part for bia,bia_part in zip(self.bias,delta_biases)]

    #实现小批量随机梯度下降法
    def mini_batch_gradient_descent(self,training_x,training_labels,iterations,mini_batch_size,epsilon,learning_rate):
        #小批量随机梯度MSGD：首先给出batch的size，然后打乱数据，选择从第一个数据到到1+size为一个batch……
        #对于每一个batch来说，运算各自的对梯度的改变的平均
        width_x = training_x.shape[1]
        #
        training_data = [(train_x,train_l)]
        #print(training_data[0].shape)
        length = len(training_data)
        #print(training_data.shape)
        counter = 0
        while counter <= iterations:
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,length,mini_batch_size)]
            #print(mini_batches[0])
            for mini_batch in mini_batches:
                self.update_gradient(mini_batch,learning_rate)




if __name__ == '__main__':
    network_test = BackPropagation_NN([784,15,10])
    # train_data,train_label,test_data,test_label = FNN_test.load_data("MNIST_data")
    #data = input_data.read_data_sets("MNIST_data", one_hot=True)
    #print(data)
    train_x, train_l, test_x, test_l = network_test.load_data('MNIST_data')
    print('训练数据的尺寸：',train_x.shape)
    print('训练标签的尺寸：',train_l.shape)
    test_x = np.array(train_x[0]).T
    test_y = train_l[0]
    #print(max(test_x))
    #print(result.shape)
    # print(test_y)
    #delta_w,delta_b = network_test.back_propagation(test_x,test_y)
    network_test.mini_batch_gradient_descent(train_x,train_l,100,100,0.005,0.01)
    result = network_test.feedforward(test_x)
    print(result)