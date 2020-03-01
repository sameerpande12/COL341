#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 01:30:30 2019

@author: sameerpande34
"""

#import linecache
from keras.models import Sequential
from keras.utils import to_categorical,Sequence
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D,ZeroPadding2D,BatchNormalization,Input,GlobalAveragePooling2D
import numpy as np
from keras.optimizers import SGD
import pandas as pd
import keras.optimizers
from keras import regularizers

# import csv
#import sys
#import time

# class Data_Generator(Sequence):
#     def __init__(self,num_images,start_index,path,batch_size):
#         self.path = path
#         self.num_images = num_images
#         self.batch_size = batch_size
#         self.start_index = start_index
    
#     def __len__(self):
#         return (int)(np.ceil((self.num_images)/ float(self.batch_size)))
    
#     def __getitem__(self,idx):
        
#         #csvreader = csv.reader(filename)
#         start = idx*self.batch_size + self.start_index
#         end = (idx+1)*self.batch_size + self.start_index
        
#         x_train=[]
        
        
#         for count in range(start,min(self.start_index + self.num_images - 1,end)):
#             lineno = count+1
#             row = linecache(self.path,lineno)
#             row = row.strip()
#             row = row.split()
#             row = [float(i) for i in row]
#             x_train.append(row)
#         """    
#         for row in csvreader:
#             if(count >= start and count<end):
#                 row = row[0]
#                 row = row.strip()
#                 row = row.split()
#                 row = [float(i) for i in row]
#                 x_train.append(row)
#             if(count>=end):
#                 break
#         """        
#         x_train = np.array(x_train)
#         y_train = to_categorical(x_train[:,x_train.shape[1]-1],num_classes=10)
#         x_train = x_train[:,:(x_train.shape[1]-1)]
        
#         x_train = [ img.reshape(32,32,3) for img in x_train]
        
#         return (x_train,y_train)
        
        
        

trainfile = "Data/train.csv"
#x_train = []
#start = time.time()
# with open(trainfile,'r') as filename:
#     csvreader = csv.reader(filename)
#     for row in csvreader:
#         num_images= num_images + 1

num_images = 0

x_train = np.loadtxt(trainfile,dtype=np.float32)
# x_test = np.loadtxt(trainfile,dtype=np.float32)

num_images = x_train.shape[0]
        
#num_train_images = (int)(num_images * 0.9)


#num_val_images = num_images - num_train_images
#end = time.time()
#print(end-start)
#x_train = np.array(x_train)
#y_train = x_train[:,x_train.shape[1]-1]
y_train = x_train[:,x_train.shape[1]-1]
y_train = to_categorical(y_train,num_classes=10)
x_train = x_train[:,:(x_train.shape[1]-1)]
x_train = np.array([x.reshape(3,32,32) for x in x_train])
x_train = np.array([np.transpose(x,(1,2,0)) for x in x_train])

model = Sequential()

model.add(ZeroPadding2D(padding=(2,2),input_shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=3,padding='valid',activation='relu'))


model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(ZeroPadding2D(padding=(2,2)))
model.add(Conv2D(192,kernel_size=3,padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(Conv2D(384,kernel_size=3,padding='same',activation='relu'))

model.add(Conv2D(256,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Flatten())

#model.add(BatchNormalization())

model.add(Dense(4096,kernel_regularizer=regularizers.l2(0.0001),activation='relu'))


model.add(Dropout(0.4))
#model.add(BatchNormalization())
model.add(Dense(2048,kernel_regularizer=regularizers.l2(0.0001),activation='relu'))

model.add(BatchNormalization())

model.add(Dense(10,kernel_regularizer=regularizers.l2(0.0001),activation='softmax'))


model.compile(optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])


history = model.fit(x_train,y_train,batch_size=100,epochs=32,verbose=1,validation_split=0.1,use_multiprocessing=True,workers=8)
hist_csv_file = "result_sgd0.01.0.9_nesteroveTrue_100batch_32epochs.csv"
hist_df = pd.DataFrame(history.history)
with open(hist_csv_file, mode='a+') as f:
    hist_df.to_csv(f)

"""
model = Sequential()


model.add(ZeroPadding2D(padding=(2,2),input_shape=()))
model.add(Conv2D(64,kernel_size=3,padding='valid',activation='relu'))


model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(ZeroPadding2D(padding=(2,2)))
model.add(Conv2D(192,kernel_size=3,padding='valid',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(Conv2D(384,kernel_size=3,padding='same',activation='relu'))

model.add(Conv2D(256,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))


model.add(Dropout(0.3))
model.add(Flatten())


model.add(Dense(4096,activation='relu'))


model.add(Dense(2048,activation='relu'))

model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))


model.compile(optimizer=SGD(lr=0.01,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=500,epochs=15,verbose=1,validation_split=0.1,use_multiprocessing=True,workers=8)
"""