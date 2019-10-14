# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:26:40 2019

@author: Sameer
"""
#Depth,Nodes,train_acc,val_acc,test_acc
import matplotlib.pyplot as plt
import pandas as pd

files = ["data_customTree_GainRatio_Entropy","data_customTree_GainRatio_Gini","data_customTree_InfoGain_Gini","data_customTree_InfoGain_Entropy"]

for file in files:
    
    data = (pd.read_csv(file+".csv")).to_numpy()
    
    fig = plt.figure()
    plt.plot(data[:,0],data[:,2])
    
    plt.plot(data[:,0],data[:,3])
    
    plt.plot(data[:,0],data[:,4])
    plt.legend(["Train","Val","Test"])
    plt.xlabel("Node Count")
    plt.ylabel("Accuracy")
    parameters = file.split("_")
    plt.title("Not Post Pruned,  {}, {}".format(parameters[2],parameters[3]))
    plt.show()
    fig.savefig(file+'.png')

fig = plt.figure()
data = pd.read_csv("data_prune.csv").to_numpy()
plt.plot(data[:,0],data[:,1])
plt.plot(data[:,0],data[:,2])
plt.plot(data[:,0],data[:,3])
plt.gca().invert_xaxis()
plt.legend(["Train","Val","Test"])
plt.xlabel("Node Count")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Node Count (while pruning)")
plt.show()
fig.savefig("data_prune.png")