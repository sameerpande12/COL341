# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import sys
trainfile = sys.argv[1]
testfile = sys.argv[2]
train_data = np.loadtxt( trainfile,delimiter=',')
test_data = np.loadtxt(testfile,delimiter=',')
# test_labels = np.loadtxt("SVM_data/test_labels.txt")

# train_data = np.loadtxt( "SVM_data/train.csv",delimiter=',')
# test_data = np.loadtxt("SVM_data/test_public.csv",delimiter=',')
# test_labels = np.loadtxt("SVM_data/test_labels.txt")


class SVM:
    
    def __init__(self,label1,label2,dimensions=(28,28)):
        self.label1 = label1
        self.label2 = label2
        self.weights = np.zeros(dimensions[0]*dimensions[1])
        self.b = 0
        
    def train(self,x,y,reg=1,T=50000,k=200,projection_step = False):
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
        
        for t in range(1,T+1):
            
            A = np.random.choice(x.shape[0],k,replace=False)
            yi_xi = np.zeros(x.shape[1])
            yi = 0
            
            for a in A:
                if 1 - y[a]*(np.dot(w,x[a]) + b) > 0:
                    
                    yi_xi = yi_xi + y[a] * x[a]
                    yi = yi + y[a]
            
            
            w = w * ( 1 - 1/t) + 1/(reg*k*t) * yi_xi
            b = b + 1/(reg* k * t) * yi
            
        self.weights = w
        self.b = b
    
    def predict(self,prediction_set):
        pred = [ np.dot(s,self.weights) + self.b for s in prediction_set]
        pred = [ self.label1 if p < 0 else self.label2 for p in pred]
        return pred

y_labels = np.sort(np.unique(train_data[:,-1]))

svm_classifiers = []
numFeatures = train_data.shape[1]-1
for i in range(0,len(y_labels)):
    for j in range(i+1,len(y_labels)):
        #print( "SVM CLASSIFIER #{}".format(i * len(y_labels)  + j + 1))
        # print("Training svm classifier number for labels : {} {}".format(y_labels[i],y_labels[j]))
        svm = SVM(y_labels[i],y_labels[j])
        train_set = train_data[  np.logical_or(train_data[:,numFeatures] == y_labels[i],train_data[:,numFeatures]==y_labels[j])]
        svm.train(train_set[:,:numFeatures],train_set[:,numFeatures],600,3000,200)
        svm_classifiers.append(svm)
        

def predict_sample(x,svm_classifiers=svm_classifiers,y_labels=y_labels):
    count = {}
    for label in y_labels:
        count[label] = 0
    for svm in svm_classifiers:
        value = svm.predict(np.array([x]))
        #print(" {} vs {} -> {}".format(svm.label1,svm.label2,value))
        count [value[0]] = count[value[0]] + 1
    #print(count)
    return max(count,key = lambda k : count[k])

count = 0
for i in range(len(train_data)):
    
    x = train_data[i,:-1]
    accurate_label = train_data[i,-1]
    
    predicted_label = (int)(predict_sample(x))
    if accurate_label == predicted_label:
        count = count+ 1
    #print("{} : correct =  {}, predicted = {}".format(i+1,accurate_label,predicted_label))

# print("accuracy: {}".format(100 * count/len(train_data)))


# test_data[:,-1] = test_labels

count = 0
outputfilename = sys.argv[3]
outputfile = open(outputfilename,'w+')
for i in range(len(test_data)):
    
    x = test_data[i,:-1]
    accurate_label = test_data[i,-1]
    
    predicted_label = (int)(predict_sample(x))
    outputfile.write("{}\n".format(predicted_label))
    # if accurate_label == predicted_label:
        # count = count+ 1    

# print("accuracy: {}".format(100 * count/len(test_data)))
