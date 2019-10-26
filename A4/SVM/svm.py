# -*- coding: utf-8 -*-

import numpy as np

train_data = np.loadtxt( "SVM_data/train.csv",delimiter=',')



class SVM:
    
    def __init__(self,label1,label2,dimensions=(28,28)):
        self.label1 = label1
        self.label2 = label2
        self.weights = np.zeros(dimensions[0]*dimensions[1])
        self.b = 0
        
    def train(self,x,y,reg=1,T=100000,k=200,projection_step = False):
        """
        reg -> regularization
        T -> number of iterations
        k -> batch size
        label1 -> mapped to -1
        label2 -> mapped to +1
        
        assume svm_a receives y labels which are either label1 or label2
        """
        
        w = self.weights
        b = self.b
        
        y = np.array([-1 if element == self.label1 else 1 for element in y])
        x_prod_y =  np.array([y]).T * x 
        
        
        
        for t in range(1,T+1):
            if(t % 1000 == 0):
                print(t)
            A = np.random.choice(x.shape[0],k,replace=False)
            A_t = [ a for a in A if   y[a] * (np.dot(w,x[a]) + b) < 1 ]
            
            learning_rate = 1/(reg * t)
            w =  (1 - learning_rate * t)  * w + learning_rate/k * np.sum(x_prod_y[A_t],axis = 0)
            b = ( 1 - learning_rate * t)  * b + learning_rate/k * np.sum(y[A_t])
        
        self.weights = w
        self.b = b

set1 = train_data[  np.logical_or(train_data[:,784] == 0,train_data[:,784]==1)]
svm = SVM(0,1)
svm.train(set1[:,:784],set1[:,784])
pred = [ np.dot(s[:784],svm.weights) for s in set1]

