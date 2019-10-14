# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:26:40 2019

@author: Sameer
"""
#Depth,Nodes,train_acc,val_acc,test_acc
import matplotlib.pyplot as plt
import pandas as pd
data = (pd.read_csv('data_customTree_GainRatio_Entropy.csv')).to_numpy()
plt.plot(data[:,0],data[:,2])

plt.plot(data[:,0],data[:,3])

plt.plot(data[:,0],data[:,4])
plt.legend(["Train","Val","Test"])
plt.show()