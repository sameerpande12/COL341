import csv
import numpy as np
import sys

trainfile = sys.argv[1]
paramfile = sys.argv[2]
weightfile = sys.argv[3]

parameters=[]
with open(paramfile,'r')as f:
    parameters = f.readlines()

for i in range(4):
    parameters[i] = float(paramters[i].strip())

paramters[4] = parameters[1].strip().split()
paramters[4] = [int(p) for p in paramters[4]]

#parameters [learning_type, learning rate/seed, max_epochs, iterations, [array of hidden layers]]


def sigmoid (x):
    return 1/(1+np.exp(-x))

def relu(x):
    return max(x,0)

class neural_network:
    
    def __init__(self,layer_shapes,activation):
        
        #will contain all layer sizes: from input layer to output layer in order
        self.layer_shapes = layer_shapes
        self.num_layers = len(layer_shapes)
        self.weights = np.array([])
        self.activation = activation
        for i in range(len(layer_shapes))-1:
            w = np.zeros([layer_shapes[i]+1, layer_shapes[i+1]])
            self.weights.append(w)
        

    
    def forward(self,x_input,y_input):
        ones = np.ones((x_train.shape[0]),1)
        #x_input = np.append(ones,x_input,axis = 1)
        self.z = np.array([])

        for i in range(len(layer_shapes)):
            if(i==0):
                self.z.append(np.array(x_input))
            else:
                ones = np.ones( (self.z[i-1]).shape[0])
                z_i =  sigmoid(np.dot( np.append(ones,self.z[i-1],axis=1), self.weights[i-1]))
                self.z.append(z_i)
            
    def backpropagate(self,y_input,learning_rate):#z_l is n X num_nodes_l
        self.grad_lz = np.array([])
        f =np.array([])# element wise product product grad_lz[i+1]* z[i+1]*(1-z[i+1])
        for i in range(num_layers):
            self.grad_lz.append(np.array([]))
            f.append(np.array([]))

        z = self.z
        n = y_input.shape[0]
        self.grad_lz[num_layers-1] = (1-y_input)/(1- z[num_layers-1]) - y_input/(z[num_layers-1])

        for i in range(num_layers):
            if (i==0):
                continue
            else:
                j = num_layers-i-1
                f[j] = self.grad_lz[j+1] * self.z * (1 - self.z)
                self.grad_lz[j] = (1.0/n) * np.dot( (self.weights[1:]).T , f[j].T )
        
        for i in range(num_layers - 1):## weights[i] is input to the ith layer
            ones = np.ones(n)
            grad_lw =  np.dot((np.append(ones,self.z[i])).T, f[i])
            np.weights[i] = np.weights[i] - learning_rate * grad_lw
