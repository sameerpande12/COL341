#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:40:48 2019

@author: sameerpande34
"""

import pandas as pd
import numpy as np
import csv 


continuous= ["Age","Fnlwgt","Education Number","Capital Gain","Capital Loss","Hour per Week"]



class Node:
        def __init__(self,depth):
            self.index
            self.labels
            self.children = []
            self.depth = depth
            
        def setLabels(self,labels,label_counts):
            self.labels = labels
            self.label_counts = label_counts
            
        def setSplitter(self,index,numBins=5):
            self.index = index
            
            if self.labels[index] in continuous:
                numChildren = numBins
            else:
                numChildren = self.label_counts
            for i in range(numChildren):
                self.children.append(Node(self.depth+1))
        
    
    

def getEntropy(y):
    unique,counts = np.unique(y,return_counts=True)
    h = 0
    for i in range(unique.size):
        p = counts[i]/np.sum(counts)
        if p > 0:
            h = h - p*np.log(p)

def getInfoGain(index,labels,x,numBins=5):##x has no header. x's last column is result
    IG  = 0
    IG  = getEntropy(x)
    if labels[index] in continuous:        
        minimum = min(x[:,index])
        maximum = min(x[:,index])
        delta = (maximum - minimum)/numBins
        for i in range(numBins):
            lower_lim = minimum + i * delta
            upper_lim = lower_lim + delta
            if i ==0:
                child = x[x[:,index]<=upper_lim]
            else:
                child = x[x[:,index]>lower_lim]
                child = child[child[:,index]<=upper_lim]
            h_child = getEntropy(child[:,-1])
            p_child = child.shape[0]/x.shape[0]
            IG = IG - p_child * h_child
        
    else:
        attributes  = np.unique(x[:,index])        
        for attribute in attributes:
            child = x[x[:,index]==attribute]
            h_child = getEntropy(child[:,-1])
            p_child = child.shape[0]/x.shape[0]
            IG = IG  - p_child * h_child
        
    return IG        

def getSplitIndex(labels,x):### x has no header. x's last column stands for output label. labels have headers for all the columns excluding output
    maxIG = getInfoGain(0,labels,x)
    maxIndex = 0
    for index in range(len(labels)):
        IG = getInfoGain(index,labels,x)
        if IG > maxIG:
            maxIG = IG
            maxIndex = index
    return maxIndex
    
            
trainfile = "DT_data/train.csv"
x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        x_train.append(row)

x_train= np.array(x_train)
labels = x_train[0,:]
x_train = x_train[1:,:]

for i in range(len(labels)):
    if labels[i] in continuous:
        x_train[:,i] = [ (int)(x.strip()) for x in x_train[:,i]]
