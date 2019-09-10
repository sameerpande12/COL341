#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:11:36 2019

@author: sameerpande34
"""
import keras
import linecache
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,ZeroPadding2D,BatchNormalization
import csv
import numpy as np
import sys
import time

class Data_Generator(Sequence):
    def __init__(self,num_images,path,batch_size):
        self.path = path
        self.num_images = num_images
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(len(self.num_images)/ float(self.batch_size))
    
    def __getitem__(self,idx):
        
        csvreader = csv.reader(filename)
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size
        
        x_train=[]
        
        
        for count in range(start,min(self.num_images-1,end)):
            lineno = count+1
            row = linecache(path,lineno)
            row = row.strip()
            row = row.split()
            row = [float(i) for i in row]
            x_train.append(row)
        """    
        for row in csvreader:
            if(count >= start and count<end):
                row = row[0]
                row = row.strip()
                row = row.split()
                row = [float(i) for i in row]
                x_train.append(row)
            if(count>=end):
                break
        """        
        x_train = np.array(x_train)
        y_train = to_categorical(x_train[:,x_train.shape[1]-1],num_classes=10)
        x_train = x_train[:,:(x_train.shape[1]-1)]
        
        x_train = [ img.reshape(32,32,3) for img in x_train]
        
        return (x_train,y_train)
        
        
        

trainfile = "Data/train.csv"
#x_train = []
num_train_images = 0
start = time.time()
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        num_train_images= num_train_images + 1
end = time.time()
print(end-start)
#x_train = np.array(x_train)
#y_train = x_train[:,x_train.shape[1]-1]
#x_train = x_train[:,:(x_train.shape[1]-1)]



model = Sequential()

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128,kernel_size=3,activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])



my_gen = Data_Generator(num_train_images,train_file,32)
