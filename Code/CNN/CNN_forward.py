# --*-- coding:utf-8 --*--
#author:yohager

#this file is the forward process of CNN

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import CNN_class
import activation_function
import copy




def conv_2d(conv_layer):
    '''
    :param input_image: 输入图片
    :param filter_size: 卷积核的尺寸
    :param depth: 卷积核的个数
    :param zero_padding: 是否在周围补零
    :param stride: 卷积步幅
    :param activation: 激活函数的类型
    '''
    #input_data的维度是三维：第一维表示图片的个数，第二维表示图片的length，第三维表示图片的width
    conv_length = int((conv_layer.input.shape[1] - conv_layer.size + 2*conv_layer.zero_padding) / conv_layer.stride + 1)
    conv_width = int((conv_layer.input.shape[2] - conv_layer.size + 2*conv_layer.zero_padding) / conv_layer.stride + 1)
    feature_map = np.zeros((conv_layer.input.shape[0],conv_layer.depth,conv_length,conv_width))
    #第二层循环对每一个卷积通道
    for num in range(conv_layer.input.shape[0]):
        for j in range(conv_layer.depth):
            for k in range(conv_length):
                for m in range(conv_width):
                    feature_map[num,j,k,m] = np.sum(np.multiply(conv_layer.input[num,k:k+conv_layer.size,m:m+conv_layer.size],conv_layer.weights[j])) + conv_layer.bias[j,0]
    #这里还需要加一个激活层！！！！
    conv_layer.z_value = feature_map
    if conv_layer.activation == 'Sigmoid':
        conv_layer.a_value = activation_function.Sigmoid_forward(conv_layer.z_value)
    elif conv_layer.activation == 'RELU':
        conv_layer.a_value = activation_function.RELU_forward(conv_layer.z_value)


def pooling(pooling_layer):
    '''
    :param feature_map: 经过卷积层后的feature map
    :param pooling_size: 池化的大小
    :param pooling_type: 池化的类别：max or average
    '''
    #输入的feature_map是一个四维的矩阵:第一维是图片个数，第二维是卷积核的个数，第三，四维是一张feature map的大小
    output_feature = np.zeros(pooling_layer.output_size)
    output_feature_index = np.zeros(pooling_layer.output_size)
    #这里的output_feature也是一个四维的矩阵
    if pooling_layer.type == 'max':
        for k in range(pooling_layer.output_size[0]):
            for n in range(pooling_layer.output_size[1]):
                for i in range(pooling_layer.output_size[2]):
                    for j in range(pooling_layer.output_size[3]):
                        output_feature[k,n,i,j] = np.max(pooling_layer.input[k,n,2*i:2*i+pooling_layer.size,2*j:2*j+pooling_layer.size])
                        output_feature_index[k,n,i,j] = np.argmax(pooling_layer.input[k,n,2*i:2*i+pooling_layer.size,2*j:2*j+pooling_layer.size])
    elif pooling_layer.type == 'average':
        for k in range(pooling_layer.output_size[0]):
            for n in range(pooling_layer.output_size[1]):
                for i in range(pooling_layer.output_size[2]):
                    for j in range(pooling_layer.output_size[3]):
                        output_feature[k,n,i,j] = np.average(pooling_layer.input[k,n,2*i:2*i+pooling_layer.size,2*j:2*j+pooling_layer.size])
        output_feature_index = None
    pooling_layer.z_value = output_feature
    pooling_layer.a_value = output_feature
    pooling_layer.up_sampling = output_feature_index!=0


def full_connected(full_connected_layer):
    '''
    :param input_data: 行表示数据的特征数，列表示数据的个数
    :param layer_size: 一个隐藏层的神经元个数
    :param weights: 初始化权重
    :param bias: 全连接层偏置
    '''
    full_connected_layer.z_value = np.dot(full_connected_layer.weights,full_connected_layer.input) + full_connected_layer.bias
    if full_connected_layer.activation == 'Sigmoid':
        full_connected_layer.a_value = activation_function.Sigmoid_forward(full_connected_layer.z_value)
    elif full_connected_layer.activation == 'RELU':
        full_connected_layer.a_value = activation_function.RELU_forward(full_connected_layer.z_value)


def CNN_forward_function(model,input_data):
    model[0].input = input_data
    model[0].a_value = input_data
    #print(model[0].a_value.shape)
    for i in range(1,len(model)):
        model[i].input = model[i-1].a_value
        #print(model[i].input.shape)
        if model[i].layer_kind == 'conv':
            conv_2d(model[i])
        if model[i].layer_kind == 'pool':
            pooling(model[i])
        if model[i].layer_kind == 'full':
            if model[i-1].layer_kind == 'conv' or model[i-1].layer_kind == 'pool':
                model[i].input = model[i].input.reshape(model[i].input.shape[0],model[i].input.shape[1]*model[i].input.shape[2]*model[i].input.shape[3]).T
            full_connected(model[i])






