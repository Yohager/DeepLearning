# --*-- coding:utf-8 --*--
#author:yohager

#this file is the backward process of NN

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import NN_class
import activation_function
import NN_forward


def full_connected_back(last_layer,next_layer):
    if last_layer.layer_kind == 'full':
        if last_layer.activation == 'Sigmoid':
            last_layer.error = np.multiply(np.dot(next_layer.weights.T,next_layer.error),activation_function.Sigmoid_backward(last_layer.a_value))
        elif last_layer.activation == 'RELU':
            last_layer.error = np.multiply(np.dot(next_layer.weights.T,next_layer.error),activation_function.RELU_backward(last_layer.z_value))
    last_layer.delta_weights = np.dot(last_layer.error,last_layer.input.T)



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

