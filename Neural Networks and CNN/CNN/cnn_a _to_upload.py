#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:11:36 2019

@author: sameerpande34
"""
#import linecache

import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.utils import to_categorical,Sequence
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,ZeroPadding2D,BatchNormalization,Input

import pandas as pd
import keras.optimizers
import sys
        
trainfile = sys.argv[1]
testfile = sys.argv[2]        
outputfile = sys.argv[3]
#trainfile = 'Data/train.csv'
#testfile = 'Data/test.csv'
#outputfile = 'Data/output.txt'

num_images = 0


x_test  = np.loadtxt(testfile,dtype=np.float32)
x_train = np.loadtxt(trainfile,dtype=np.float32)


num_images = x_train.shape[0]
        
y_train = x_train[:,x_train.shape[1]-1]
y_train = to_categorical(y_train,num_classes=10)
x_train = x_train[:,:(x_train.shape[1]-1)]
x_train = np.array([x.reshape(3,32,32) for x in x_train])
x_train = np.array([np.transpose(x,(1,2,0)) for x in x_train])

