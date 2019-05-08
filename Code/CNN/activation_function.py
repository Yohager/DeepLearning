# --*-- coding:utf-8 --*--
#author:yohager

#this file is the activation process of CNN

import numpy as np


def Sigmoid_forward(weighted_input):
    return 1.0 / (1.0 + np.exp(-weighted_input))

def Sigmoid_backward(output_value):
    return np.multiply(output_value,(1-output_value))

def RELU_forward(weighted_input):
    return weighted_input * (weighted_input >= 0)

def RELU_backward(output_value):
    output_value[output_value>0] = 1
    output_value[output_value<=0] = 0
    return output_value

def softmax_forward(H_x):
    for i in H_x:
        i = np.exp(i) / np.sum(np.exp(H_x))
    return H_x
