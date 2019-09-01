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
    if (i==1):
        parameters[i] = float(parameters[i].strip())
    else:
        parameters[i] = int(parameters[i].strip())

parameters[4] = parameters[4].strip().split()
parameters[4] = [int(p) for p in parameters[4]]
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
        self.weights = []
        self.activation = activation
        for i in range(len(layer_shapes)-1):
            w = np.zeros([layer_shapes[i]+1, layer_shapes[i+1]])
            self.weights.append(w)
        self.weights = np.array(self.weights)
        

    
    def forward(self,x_input):
        ones = np.ones((x_input.shape[0],1))
        #x_input = np.append(ones,x_input,axis = 1)
        self.z = []

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


x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        row = [float(i) for i in row]
        x_train.append(row)

x_train = np.array(x_train)
y_train = x_train[:,x_train.shape[1]-1]
x_train = x_train[:,:(x_train.shape[1]-1)]

batch_size = parameters[3]
base_rate = parameters[1]
num_iters = parameters[2]
learning_type = parameters[0]
layer_shapes = [x_train.shape[1]]+parameters[4]+[1]

num_batches = (int)(x_train.shape[0]/batch_size)
if( x_train.shape[0] > batch_size*num_batches):
    num_batches = num_batches + 1

nn = neural_network(layer_shapes,sigmoid)
for i in range(num_iters):
    print("Iteration :{}".format(i+1))
    for j in range(num_batches):
        x = x_train[ j*batch_size : (j+1)*batch_size]
        y = y_train[ j*batch_size : (j+1)*batch_size]
        nn.forward(x)
        learning_rate = base_rate
        if(learning_type == 2):
            learning_rate = learning_rate/np.sqrt(i+1)
        nn.backpropagate(y,learning_rate)

