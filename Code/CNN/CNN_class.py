# --*-- coding:utf-8 --*--
#author:yohager
'''
this file is the different layers for cnn
conv_layer
pooling_layer
full_connected_layer
'''
import numpy as np


class input_layer:
    def __init__(self,data_size,batch_size):
        #假设输入的数据为所有的数据:784 * 6w，则通过batch_size控制每一次传入的数据量:784 * batch_size
        self.output_size = [batch_size,data_size,data_size]
        self.input = None
        self.a_value = None
        self.layer_kind = 'input'


class conv_2d_layer:
    def __init__(self,input,filter_size,depth,zero_padding,stride,activation):
        '''
        :param input_image: 输入图片
        :param filter_size: 卷积核的尺寸
        :param depth: 卷积核的个数
        :param zero_padding: 是否在周围补零
        :param stride: 卷积步幅
        :param activation: 激活函数的类型
        '''
        self.input = None
        self.size = filter_size
        self.depth = depth
        self.zero_padding = zero_padding
        self.stride = stride
        self.activation = activation
        self.weights = np.random.rand(depth,filter_size,filter_size)
        self.bias = np.random.rand(depth,1)
        self.output_size = [input.output_size[0],self.depth,int((input.output_size[1] - filter_size + 2*zero_padding)/stride + 1),int((input.output_size[2] - filter_size + 2*zero_padding)/stride + 1)]
        self.error = None
        self.z_value = None
        self.a_value = None
        self.delta_weights = None
        self.layer_kind = 'conv'


class pooling_layer:
    def __init__(self,input,pooling_size,pooling_type):
        '''
        :param feature_map: 经过卷积层后的feature map
        :param pooling_size: 池化的大小
        :param pooling_type: 池化的类别：max or average
        '''
        self.input = None
        self.size = pooling_size
        self.output_size = [input.output_size[0],input.output_size[1],int(input.output_size[2]/pooling_size),int(input.output_size[3]/pooling_size)]
        self.up_sampling = None
        self.z_value = None
        self.a_value = None
        self.type = pooling_type
        self.layer_kind = 'pool'


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
        if input.layer_kind == 'conv' or input.layer_kind == 'pool':
            self.size = [layer_size,input.output_size[1]*input.output_size[2]*input.output_size[3]]
        elif input.layer_kind == 'full':
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