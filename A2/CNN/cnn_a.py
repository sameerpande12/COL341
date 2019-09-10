#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:11:36 2019

@author: sameerpande34
"""
#import linecache
from keras.models import Sequential
from keras.utils import to_categorical,Sequence
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,ZeroPadding2D,BatchNormalization,Input
import numpy as np
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
x_train = np.array([x.reshape(32,32,3) for x in x_train])

# x_val = x_train[num_train_images:]
# y_val = y_train[num_train_images:]
# y_train = y_train[:num_train_images]

#img_input = Input(shape=(3,32,32))

model = Sequential()


#model.add(ZeroPadding2D(padding=(1,1),input_shape=(32,32,3)))

model.add(Conv2D(64,kernel_size=3,padding='same',activation='relu',input_shape=(32,32,3)))

model.add(MaxPooling2D(pool_size=(2,2),strides=None))

#model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=None))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(10,activation='softmax'))


model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(x_train,y_train,batch_size=100,epochs=50,verbose=1,validation_split=0.1,use_multiprocessing=True,workers=4)
# batch_size = 32
# my_training_gen = Data_Generator(num_train_images,0,trainfile,batch_size)
# my_validation_gen = Data_Generator(num_val_images,num_train_images,trainfile,batch_size)


# model.fit_generator(generator = my_training_gen,
#                     steps_per_epoch=(num_train_images)//batch_size,
#                     verbose=1,
#                     validation_data=my_validation_gen,
#                     epochs=10,
#                     validation_steps = (num_val_images//batch_size),
#                     use_multiprocessing=True,
#                     workers=8,
#                     max_queue_size=32)