# --*-- coding:utf-8 --*--
#author:yohager

#this file is the backward process of CNN

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import activation_function


def conv_weights_delta(input,filters_error):
    #print('in',input.shape)
    #print('filter',filters_error.shape)
    #input为输入的图片（三维：图片数*尺寸*尺寸），filter_error为从pooling层传过来的error：四维：图片数 * depth * 卷积核尺寸*卷积核尺寸
    filters_using = np.average(filters_error,axis=0)
    filter_size_out = input.shape[1] - filters_using.shape[1] + 1
    #先将filter_error压缩为三维，将图片数压掉，以手写体为例:input：5 * 28 *28， filter_error：3 * 24 *24
    feature_map_size = [input.shape[0],filters_using.shape[0],filter_size_out,filter_size_out]
    #print('featureshape',feature_map_size)
    weights_delta = np.zeros(feature_map_size)
    #print('input',input[-1].shape)
    #print('filter_using_size',filters_using.shape)
    for i in range(feature_map_size[0]):
        for j in range(feature_map_size[1]):
            for k in range(feature_map_size[2]):
                for n in range(feature_map_size[3]):
                    #print(filters_using[j].shape)
                    weights_delta[i,j,k,n] = np.sum(input[i,k:k+filters_using.shape[1],n:n+filters_using.shape[1]] * filters_using[j])
    weights_delta = np.average(weights_delta,axis=0)
    #返回的weights_delta尺寸与初始化的卷积核的尺寸一致，三维：卷积核数量*卷积尺寸*卷积尺寸
    return weights_delta

def test_index(index):
    data = np.zeros(shape=(index.shape[0],index.shape[1],index.shape[2]*2,index.shape[3]*2))
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                for n in range(index.shape[3]):
                    data[i,j,2*k:2*k+2,2*n:2*n+2][int(index[i,j,k,n]//2),int(index[i,j,k,n]%2)] =1
    #print(data.shape)
    return data

#用于返回池化层反向传播上采样的结果，这个函数目前还有一些问题
def pooling_upsampling(data_error,data_index,pooling_size):
    data_size = [data_error.shape[0],data_error.shape[1],data_error.shape[2]*pooling_size,data_error.shape[3]*pooling_size]
    data_index_upsampling = test_index(data_index)
    data_upsampling = np.zeros(data_size)
    for i in range(data_error.shape[0]):
        for j in range(data_error.shape[1]):
            for k in range(data_error.shape[2]):
                for n in range(data_error.shape[3]):
                    data_upsampling[i,j,2*k:2*k+pooling_size,2*n:2*n+pooling_size] = data_error[i,j,k,n]
    data_upsampling = data_upsampling * data_index_upsampling
    return data_upsampling

#卷积反向传播的主函数
def conv_back(conv_layer,next_layer):
    #print('next_layer_size',next_layer.error.shape)
    if next_layer.layer_kind == 'pool':
        data_upsampling = pooling_upsampling(next_layer.error,next_layer.up_sampling,next_layer.size)
        if conv_layer.activation == 'Sigmoid':
            conv_layer.error = data_upsampling * activation_function.Sigmoid_backward(conv_layer.a_value)
        elif conv_layer.activation == 'RELU':
            conv_layer.error = data_upsampling * activation_function.RELU_backward(conv_layer.z_value)
        conv_layer.delta_weights = conv_weights_delta(conv_layer.input,conv_layer.error)
    else:
        pass

#池化层反向传播的主函数
def pooling_back(pooling_layer,next_layer):
    if next_layer.layer_kind == 'full':
        pooling_layer.error = np.dot(next_layer.weights.T,next_layer.error).reshape(pooling_layer.output_size)
        #这里需要进行up_sampling，具体进行的方式是通过对这个error进行扩展
        #pooling_layer.error = pooling_upsampling(error,pooling_layer.up_sampling,pooling_layer.size)
    elif next_layer.layer_kind == 'conv':
        pass

#全连接层的反向传播主函数
def full_connected_back(last_layer,next_layer):
    if last_layer.layer_kind == 'full':
        if last_layer.activation == 'Sigmoid':
            last_layer.error = np.multiply(np.dot(next_layer.weights.T,next_layer.error),activation_function.Sigmoid_backward(last_layer.a_value))
        elif last_layer.activation == 'RELU':
            last_layer.error = np.multiply(np.dot(next_layer.weights.T,next_layer.error),activation_function.RELU_backward(last_layer.z_value))
    last_layer.delta_weights = np.dot(last_layer.error,last_layer.input.T)


#最后一层反向传播的函数
def last_layer_error(last_layer,real_data,loss_function_type):
    #输入的pred_data结果也为小批量的结果，即维度是二维，行为特征，列为批量个数
    #计算正向的误差和反向的误差
    if loss_function_type == 'RSE':
        if last_layer.activation == 'Sigmoid':
            last_layer.error = (last_layer.a_value - real_data) * activation_function.Sigmoid_backward(last_layer.a_value)
        #error = np.sum((last_layer.a_value - real_data)**2)
        elif last_layer.activation == 'RELU':
            last_layer.error = (last_layer.a_value - real_data) * activation_function.RELU_backward(last_layer.z_value)
    last_layer.delta_weights = last_layer.error.dot(last_layer.input.T)
    #return delta_weights,delta_bias


#封装好的CNN反向传播的主函数
def CNN_backward_function(model,label,alpha,error_type):
    last_layer_error(model[-1],label,error_type)
    model[-1].weights -= alpha * model[-1].delta_weights / model[-1].input.shape[1]
    model[-1].bias -= alpha * np.average(model[-1].error,axis=1).reshape(len(model[-1].error),1)
    for l in range(2,len(model)):
        if model[-l].layer_kind == 'full':
            full_connected_back(model[-l],model[-l+1])
        elif model[-l].layer_kind == 'pool':
            pooling_back(model[-l],model[-l+1])
        elif model[-l].layer_kind == 'conv':
            conv_back(model[-l],model[-l+1])
    for k in range(1,len(model)-1):
        if model[k].layer_kind == 'conv':
            model[k].weights -= alpha * model[k].delta_weights
            model[k].bias -= alpha * np.average(model[k].delta_weights,axis=(1,2)).reshape(len(np.average(model[k].delta_weights,axis=(1,2))),1)#有问题！！！！！
        elif model[k].layer_kind == 'pool':
            pass
        elif model[k].layer_kind == 'full':
            model[k].weights -= alpha * model[k].delta_weights / model[k].input.shape[1]
            model[k].bias -= alpha * np.average(model[k].error, axis=1).reshape(len(model[k].error), 1)


