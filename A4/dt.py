#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:40:48 2019

@author: sameerpande34
"""

import pandas as pd
import numpy as np
import csv 



class one_hot_encoder:
    def __init__(self,labels):
        self.labels=labels#two dimenstional array
    
    def encode(self,x_input):
        x_output = []
        col_index = 0
        for i in range((self.labels).size):
            if( (self.labels[i]).size == 0):
                x_output.append(x_input[:,col_index])
                
            else:
                for label in self.labels[i]:
                    x_output.append(x_input[:,col_index]==label)
            
            col_index= col_index+1
                
        
        return np.array(x_output)

trainfile = "DT_data/train.csv"
x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        x_train.append(row)

labels = x_train[0]
labels = labels[:-1]
labels = [label.strip() for label in labels]
x_train=x_train[1:]
x_train = np.array(x_train)

y_train = x_train[:,-1]
y_train = np.array([(int)(y) for y in y_train])
continuous= ["Age","Fnlwgt","Education Number","Capital Gain","Capital Loss","Hour per Week"]


label_array = []
for i in range(len(labels)):
    label = labels[i]
    if label in continuous:
        label_array.append(np.array([]))
    else:
        print(label+" not in continuous")
        label_array.append(np.unique(x_train[:,i]))

label_array = np.array(label_array)    
training_encoder = one_hot_encoder(label_array)
x_train_encoded = training_encoder.encode(x_train)
        