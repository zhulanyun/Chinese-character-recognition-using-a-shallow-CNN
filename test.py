# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:33:08 2018

@author: zhu
"""

import numpy as np
from PIL import Image
from scipy import misc
import os
from keras.models import load_model
import csv
import pandas as pd

dirs = 'path1'    #this is the address of your training set
filelists = os.listdir(dirs)
dic = {}
i = 0
for file in filelists:
    dic[str(i)] = file
    i = i+1
    
csvfile = open('list.csv', 'w', newline='')
writer = csv.writer(csvfile)
m = 'filename'
m.encode('utf-8')
n = 'label'
n.encode('utf-8')


    
model = load_model('model_3_7_100.h5')
dirs = 'path2' #this is the address of your test set
filelists = os.listdir(dirs)
files = []
results = []
for file in filelists:
    files.append(file)
    path = dirs + '/' + file
    im = Image.open(path)
    x = np.array(im)
    x = misc.imresize(x,[28,28]).astype('float32')/255
    x = x.reshape((28,28,1))
    x = np.expand_dims(x,axis=0)
    x = model.predict(x)
    x = 1-x
    x = np.argsort(x)
    y = ''
    for j in x[0][:5]:
        i = []
        y = y + dic[str(j)]
    results.append(y)


dataframe = pd.DataFrame({'filename':files, 'label':results})
dataframe.to_csv('list.csv',index=False,sep=',',encoding='utf-8')
    
    
    
    
