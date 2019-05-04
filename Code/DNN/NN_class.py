# --*-- coding:utf-8 --*--
#author:yohager
'''
this file is the layer for NN(basic Neuron Network)
full_connected_layer
'''

import numpy as np



class input_layer:
    def __init__(self,data_size,batch_size):
        #假设输入的数据为所有的数据:784 * 6w，则通过batch_size控制每一次传入的数据量:784 * batch_size
        self.size = [data_size,batch_size]
        self.input = None
        self.a_value = None
        self.layer_kind = 'input'


class full_connected_layer:
    def __init__(self,input,layer_size,activation):
        '''
        :param input_data: 行表示数据的特征数，列表示数据的个数  m*n：n个m维的数据
        :param layer_size: 一个隐藏层的神经元个数设为k
        :param weights: 初始化权重
        :param bias: 全连接层偏置
        '''
        #行为属性数，列为样本数
        #self.input = input_data.reshape(input_data.shape[1]*input_data.shape[2]*input_data.shape[3],input_data.shape[0])
        self.size = [layer_size,input.size[0]]
        #前一层的输入数 * 下一层神经元数
        self.weights =  np.random.randn(self.size[0],self.size[1])/10
        self.bias = np.random.randn(layer_size,1)/10
        self.input = None
        self.activation = activation
        self.z_value = None
        self.a_value = None
        self.error = None  #这个既表示error也表示delta_bias
        self.delta_weights = None
        self.layer_kind = 'full'

