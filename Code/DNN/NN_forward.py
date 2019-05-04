# --*-- coding:utf-8 --*--
#author:yohager

#this file is the forward process of NN

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import NN_class
import activation_function


def full_connected(model,data):
    '''
    :param input_data: 行表示数据的特征数，列表示数据的个数
    :param layer_size: 一个隐藏层的神经元个数
    :param weights: 初始化权重
    :param bias: 全连接层偏置
    '''
    model[0].input = data
    model[0].a_value = data
    for i in range(1,len(model)):
        model[i].input = model[i-1].a_value
        model[i].z_value = np.dot(model[i].weights,model[i].input) + model[i].bias
        if model[i].activation == 'Sigmoid':
            model[i].a_value = activation_function.Sigmoid_forward(model[i].z_value)
        elif model[i].activation == 'RELU':
            model[i].a_value = activation_function.RELU_forward(model[i].z_value)



