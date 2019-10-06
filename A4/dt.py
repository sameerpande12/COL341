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
            self.index=None
            self.labels
            self.children = []
            self.depth = depth
            self.prediction = None
            self.bins
            
        def setLabels(self,labels,label_counts):
            self.labels = labels
            self.label_counts = label_counts
                    
        
        def setSplitter(self,index,numBins=5):
            self.index = index
            
            if self.labels[index] in continuous:
                numChildren = numBins
                self.bins = numBins
            else:
                numChildren = self.label_counts
            for i in range(numChildren):
                self.children.append(Node(self.depth+1))
        
        def createTrainingDecisionBoundaries(self,x):
            if self.labels is None:
                print("Please let us know the headers first")
                return
            
            category_boundary = []
            for i in range(len(self.labels)):
                if self.labels[i] in continuous:
                    numBins = 5
                    minimum = min(x[:,i])
                    maximum = max(x[:,i])
                    category_boundary.append([numBins,minimum,maximum])
                else:
                    categories = np.unique(x[:,i])
                    catIndex = {}
                    i= 0
                    for category in categories:
                        catIndex[catIndex] = i
                        i = i + 1
                        
                    category_boundary.append((np.unique(x[:,i]),catIndex))
          
            self.category_boundary = category_boundary         
                
                
        
        
        def learnTree(self,x,maxDepth):
        
            if self.depth>=maxDepth == 0:
                y = x[x[:,-1]]
                unique,counts = np.unique(y,return_counts=True)
                self.prediction = unique[ np.argmax(counts)]
                
            else:
                index = self.index##splitter has been set already
                splits = []
                if self.labels[index] in continuous:
                    numBins,minimum,maximum = self.category_boundary
                    childNumber = (x[:,index] - minimum)/numBins
                    splits = []
                    for i in range(numBins):
                        splits.append([])
                    
                    for i in range(x.shape[0]):
                        splits[childNumber[i]].append(x[i])
                
                else:
                    nextNode = x[:,index]
                    catIndex = self.category_boundary[index][1]
                    splits = []
                    for i in range(len(self.category_boundary[index][0])):
                        splits.append([])
                    
                    for i in range(x.shape[0]):
                        splits[catIndex[nextNode[i]]].append(x[i])
                    
                ##since children are already created we only need to tell their splitter
                for i in range(len(self.children)):
                    
                    self.children[i].labels = self.labels
                    self.children[i].label_counts = self.label_counts
                    self.children[i].category_boundary = self.category_boundary
                    self.children[i].depth = self.depth + 1
                    if self.children[i].depth < maxDepth:
                        splitIndex = getSplitIndex(labels,np.array(splits[i]),self.category_boundary)
                        if labels[splitIndex] in continuous:
                            self.children[i].setSplitter(splitIndex,self.category_boundary[splitIndex][0])
                        else:
                            self.children[i].setSplitter(splitIndex)
                    
                    self.children[i].learnTree(np.array(splits[i]),maxDepth)
        
        def predictSingle(self,x):
            if self.prediction is None:
                index = self.index
                childNum = -1
                if labels[index] in continuous:
                    childNum  = (self.category_boundary[index][2] - self.category_boundary[index][1])/self.category_boundary[i][0]
                else:
                    childNum = self.category_boundary[index][1][x[index]]
                
                return self.children[childNum].predictSingle(x)
            else:
                return self.prediction
        
        def predict(self,x_test):
            return self.predictSingle(np.array(x_test))
            
                  
                        


def getEntropy(y):
    unique,counts = np.unique(y,return_counts=True)
    h = 0
    for i in range(unique.size):
        p = counts[i]/np.sum(counts)
        if p > 0:
            h = h - p*np.log(p)

def getInfoGain(index,labels,x,decisionBoundary):##x has no header. x's last column is result
    IG  = 0
    IG  = getEntropy(x)
    if labels[index] in continuous:        
        minimum = decisionBoundary[index][1]
        maximum = decisionBoundary[index][2]
        numBins = decisionBoundary[index][0]
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

def getSplitIndex(labels,x,decisionBoundary):### x has no header. x's last column stands for output label. labels have headers for all the columns excluding output
    maxIG = getInfoGain(0,labels,x,decisionBoundary)
    maxIndex = 0
    for index in range(len(labels)):
        IG = getInfoGain(index,labels,x,decisionBoundary)
        if IG > maxIG:
            maxIG = IG
            maxIndex = index
    return maxIndex
    
            
trainfile = "DT_data/train.csv"
x_train = []
labels = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    count = 0
    for row in csvreader:
        if(count > 0):
            for i in range(len(labels)):
                row[i] = row[i].strip()
                if labels[i] in continuous:
                    row[i] = (float)(row[i])
                elif i == len(row) - 1:
                    row[i] = (int)(row[i])
                x_train.append(row)
                
        else:
            labels = row
            labels = [label.strip() for label in labels]
            count= count + 1


x_train= np.array(x_train,dtype='object')
labels = labels[:-1]