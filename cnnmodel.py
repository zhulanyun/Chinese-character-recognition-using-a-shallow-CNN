# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:27:16 2018

@author: zhu
"""

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras import models

def cnnModel(input_shape):
    
    data = Input(shape = input_shape)
    
    X = Conv2D(6, (5,5), strides=(1,1), activation='relu', padding='same')(data)
    X = Conv2D(6, (5,5), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=1, epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X= Conv2D(16, (5,5), strides=(1,1), activation='relu', padding='same')(X)
    X= Conv2D(16, (5,5), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=1, epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    X= Conv2D(22, (5,5), strides=(1,1), activation='relu', padding='same')(X)
    X= Conv2D(22, (5,5), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=1, epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides = (2,2))(X)
    
    
    X = Flatten()(X)
    
    X = Dense(300)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    X = Dense(200)(X)
    X = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None, beta_init='zero', gamma_init='one', mode=0)(X)
    X = Activation('relu')(X)
    Y = Dense(100, activation = 'softmax')(X)
    
    model = models.Model(inputs=data, outputs=Y, name='cnnmodel')
    
    return model
    
    
