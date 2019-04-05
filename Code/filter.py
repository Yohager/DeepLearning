#-*- coding:utf-8 -*-
#author: Yohager

from PIL import Image
import cv2
import numpy as np 


def load_image(filepath):
    image_load = cv2.imread(filepath)
    print(image_load.shape)
    return image_load

def filter_test(data,filter_size):
    #data is a*b*c matrix; filter_size is one m*m matrix 
    height,width,level = data.shape
    m = filter_size.shape[0]
    data_r = data[:,:,0]
    data_g = data[:,:,1]
    data_b = data[:,:,2]
    filter_finish_matrix_r = np.empty([height-m+1,width-m+1])
    filter_finish_matrix_g = np.empty([height-m+1,width-m+1])
    filter_finish_matrix_b = np.empty([height-m+1,width-m+1])
    for i in range(height-m+1):
        for j in range(width-m+1):
            filter_finish_matrix_r[i][j] = np.sum(np.multiply(np.array(data_r[i:i+m,j:j+m]),filter_size))
            filter_finish_matrix_g[i][j] = np.sum(np.multiply(np.array(data_g[i:i+m,j:j+m]),filter_size))
            filter_finish_matrix_b[i][j] = np.sum(np.multiply(np.array(data_b[i:i+m,j:j+m]),filter_size))           
    return np.array([filter_finish_matrix_r,filter_finish_matrix_g,filter_finish_matrix_b])



if __name__ == '__main__':
    filepath = 'adult.jpg'
    image_data = load_image(filepath)
    filter_feature = np.array([[1,1,1],[1,1,1],[1,1,1]])
    print(filter_feature.shape)
    filter_result = filter_test(image_data,filter_feature)
    print(filter_result.shape)
