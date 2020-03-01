#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 01:30:30 2019

@author: sameerpande34
"""

#import linecache
import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.utils import to_categorical,Sequence
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D,ZeroPadding2D,BatchNormalization,Input,GlobalAveragePooling2D

from keras.optimizers import SGD

import keras.optimizers
from keras import regularizers
import sys
        


trainfile = sys.argv[1]
testfile = sys.argv[2]        
outputfile = sys.argv[3]
#trainfile = 'Data/train.csv'
#testfile = 'Data/test.csv'
#outputfile = 'Data/output.txt'

num_images = 0

x_train = np.loadtxt(trainfile,dtype=np.float32)


num_images = x_train.shape[0]
        
y_train = x_train[:,x_train.shape[1]-1]
y_train = to_categorical(y_train,num_classes=10)
x_train = x_train[:,:(x_train.shape[1]-1)]
x_train = np.array([x.reshape(3,32,32) for x in x_train])
x_train = np.array([np.transpose(x,(1,2,0)) for x in x_train])

x_test = np.loadtxt(testfile,dtype=np.float32)
x_test = x_test[:,:3072]
x_test = np.array([x.reshape(3,32,32) for x in x_test])
x_test = np.array([np.transpose(x,(1,2,0)) for x in x_test])


model = Sequential()
model.add(Conv2D(64,kernel_size=3,input_shape=(32,32,3),kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(64,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(128,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(256,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(256,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(256,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(512,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(512,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(512,kernel_regularizer=regularizers.l2(0.00025),activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))


model.compile(optimizer=SGD(lr=0.008,decay=1e-6,momentum=0.9,nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=100,epochs=40,verbose=1,validation_split=0.1)


y_test = model.predict(x_test)
y_test = np.argmax(y_test,axis=1)
with open(outputfile,'w+') as f:
    for y in y_test:
        f.write("{}\n".format((int)(y))) 


