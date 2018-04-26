# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:48:06 2018

@author: zhu
"""

import numpy as np
import keras
from cnnmoel import cnnModel
from keras.utils import to_categorical

X_train = np.load('train_X28_1.npy')
Y_train = np.load('train_Y28_1.npy')
print('read over')
X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255
Y_train = to_categorical(Y_train.astype('float32'))
model = cnnModel((28,28,1))
model.compile(optimizer=keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 
                 loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, batch_size=16, epochs=80)
print('fit over')
model.save('model.h5')

