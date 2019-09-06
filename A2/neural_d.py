import csv
import numpy as np
import sys
from scipy import ndimage as ndi
from skimage import feature

trainfile = "Neural_data/CIFAR10/train.csv"
testfile = "Neural_data/CIFAR10/test_X.csv"
outputfile = "output_weights.txt"
#trainfile = sys.argv[1]
#testfile = sys.argv[2]
#outputfile = sys.argv[3]

#################### sigmoid 
# parameters=[1,0.5,10000,100,[100],0.1]#31
# parameters=[1,0.5,50000,100,[100],0.1]#30.8
# parameters=[2,0.5,10000,100,[100]]#29.1
# parameters=[1,0.5,10000,100,[100,20],0.1]#33.78
# parameters=[1,0.5,10000,100,[100,20],1]#15
# parameters=[1,0.5,10000,100,[100,20],0.01]#31.62
# parameters=[1,0.5,10000,100,[100,50],0.1]#34.02
# parameters=[1,0.5,10000,100,[100,100],0.1]#35
# parameters=[1,0.5,10000,300,[100,100],0.1]#30
# parameters=[1,1,10000,100,[100,100],0.1]#33
# parameters=[1,0.5,10000,100,[100,100],0.1]#35
# parameters=[1,0.5,10000,100,[100,50,10],0.01]#16
########################



#######################softplus
# parameters=[1,0.5,10000,100,[100,100],0.1]#34.44

# parameters=[1,0.5,20000,100,[100,100],0.1]#33.7

#######################relu
# parameters=[1,0.5,10000,100,[100,100],0.1]#10

########################leaky_relu
# parameters=[1,0.5,10000,100,[100,100],0.1]#31


########################tanh
# parameters=[1,0.5,10000,100,[100,100],0.1]#33

# for i in range(4):
#     if (i==1):
#         parameters[i] = float(parameters[i].strip())
#     else:
#         parameters[i] = int(parameters[i].strip())

# parameters[4] = parameters[4].strip().split()
# parameters[4] = [int(p) for p in parameters[4]]
#parameters [learning_type, learning rate/seed, max_epochs, iterations, [array of hidden layers]]



def sigmoid (x):
    return 1/(1+np.exp(-x))

def relu(x):
    x[x<=0]=0
    return x

def softplus(x):
    return np.log(1 + np.exp(x))

def leaky_relu(x):
    x[x<=0] = 0.01 * x[x<=0]
    return x

def tanh(x):
    return np.tanh(x)
    
class preprocessor:
    def fit(self,x_train):
        self.mean = np.mean(x_train,axis=0)
        self.std = np.std(x_train,axis=0)
        self.std[self.std==0] = 1
    
    def normalize(self,x):
        return (x-self.mean)/(self.std)
    def de_normalize(self,x):
        return (x * self.std ) + self.mean
        
            
            
class one_hot_encoder:
    
    def __init__(self,labels_arr):
        self.labels_arr = labels_arr

    def encode(self,x_input):

        x_output =[]

        for label in range(len(self.labels_arr)):
            x_output.append((x_input==label).astype(np.float64))

        x_output = np.transpose(np.array(x_output))
        return x_output
    def decode(self,y_encoded):
        y_out = np.argmax(y_encoded,axis=1)
        y_decoded = [self.labels_arr[j] for j in y_out]
        return y_decoded
        
            
    # def decode(self,y_encoded):
    #     y_out = np.argmax(y_encoded,axis = 1)
    #     y_decoded = []
    #     for i in range(len(self.labels_arr)):
    #         y_decoded = y_decoded + [self.labels_arr[i][j] for j in y_out]

    #     return y_decoded


class neural_network:
    
    def __init__(self,layer_shapes,activation):
        
        #will contain all layer sizes: from input layer to output layer in order
        self.layer_shapes = layer_shapes
        self.num_layers = len(layer_shapes)
        self.weights = []
        self.activation = activation
        for i in range(len(layer_shapes)-1):
            w = (np.random.random([layer_shapes[i]+1, layer_shapes[i+1]]))*np.sqrt(2)/(  layer_shapes[i] * (layer_shapes[i+1]+1))
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
                ones = np.ones( ((self.z[i-1]).shape[0],1) )
                if (i==self.num_layers - 1):
                    z_i =  (np.dot( np.append(ones,self.z[i-1],axis=1), self.weights[i-1]))
                    z_i = np.exp(z_i)
                    col = np.sum(z_i,axis=1)
                    col = np.array([col]).T
                    z_i = z_i/col
                    self.y_pred = z_i
                else:
                    z_i =  self.activation(np.dot( np.append(ones,self.z[i-1],axis=1), self.weights[i-1]))
                self.z.append(z_i)
        
        # self.y_pred = np.exp(self.z[num_layers-1])
        # col = np.sum(self.y_pred,axis = 1)
        # col = (np.array([col])).T
        # self.y_pred = (self.y_pred)/col
        # self.z[num_layers-1] = self.y_pred
        
    def backpropagate(self,y_input,learning_rate,lamda=0.1):#z_l is n X num_nodes_l
        self.grad_lz = []
        num_layers = self.num_layers
        f =[]# element wise product product grad_lz[i+1]* z[i+1]*(1-z[i+1])
        for i in range(num_layers):
            self.grad_lz.append([])
            f.append([])

        
        n = y_input.shape[0]
        if(n==0):
            return
        
        # self.grad_lz[num_layers-1] = (1.0/n)* (self.y_pred - y_input)/((self.y_pred)*(1-self.y_pred))
        for i in range(num_layers):
            j = num_layers-i-1
            if (j==num_layers-1):
                self.grad_lz[j] = - (1.0/n)* (y_input/self.y_pred) 
            elif (j==num_layers-2):
                self.grad_lz[j] = (1.0/n)* np.dot((self.y_pred -y_input),((self.weights[j])[1:]).T)
            else:
                if(self.activation == sigmoid):
                    f[j]=self.grad_lz[j+1] * self.z[j+1] * (1 - self.z[j+1])
                elif(self.activation == softplus):
                    term = np.exp(self.z[j+1])
                    f[j]=self.grad_lz[j+1] * (term - 1) / term
                elif(self.activation == relu):
                    grad = self.z[j+1]
                    grad[grad > 0] = 1
                    grad[grad == 0] = 0.5
                    grad[grad < 0] = 0
                    f[j]=self.grad_lz[j+1] * grad
                elif(self.activation == leaky_relu):
                    grad = self.z[j+1]
                    grad[grad>0]=1
                    grad[grad==0] = 0.5
                    grad[grad<0]=0.01
                    f[j]= self.grad_lz[j+1] * grad
                elif(self.activation == tanh):
                    grad = 1- self.z[j+1]**2
                    f[j] = self.grad_lz[j+1]*grad
                self.grad_lz[j] =  np.dot( f[j] ,((self.weights[j])[1:]).T)
            
        
        
        for i in range(num_layers-1):
            ones = np.ones((n,1))
            if(i==num_layers-2):
                grad_lw = np.dot((np.append(ones,self.z[i],axis=1)).T, (self.y_pred-y_input))*(1.0/n)
            else:
                grad_lw =  np.dot((np.append(ones,self.z[i],axis=1)).T, f[i])
            grad_lw = grad_lw + (lamda/n)*self.weights[i]
            self.weights[i] = self.weights[i] - learning_rate * grad_lw
        # self.grad_lz[num_layers-1]= (1.0/n)*(self.y_pred - y_input)
        # for i in range(num_layers):
        #     j = num_layers-i-1
        #     if (j==num_layers-1):
        #         continue
        #     else:
        #         f[j] = self.grad_lz[j+1] * self.z[j+1] * (1 - self.z[j+1])
        #         self.grad_lz[j] =  np.dot( f[j] ,((self.weights[j])[1:]).T)
        
        # for i in range(num_layers - 1):## weights[i] is input to the ith layer
        #     ones = np.ones((n,1))
        #     grad_lw =  np.dot((np.append(ones,self.z[i],axis=1)).T, f[i])
        #     self.weights[i] = self.weights[i] - learning_rate * grad_lw


x_test = []
with open(testfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        row = [float(i) for i in row]
        x_test.append(row)
x_test = np.array(x_test)
x_test = x_test[:,:(x_test.shape[1]-1)]

x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        row = [float(i) for i in row]
        x_train.append(row)
        
x_train = np.array(x_train)
np.random.shuffle(x_train)
y_train = x_train[:,x_train.shape[1]-1]
x_train = x_train[:,:(x_train.shape[1]-1)]

#temp = x_train.reshape(x_train.shape[0],(int)(np.sqrt(x_train.shape[1])), (int)(np.ceil(x_train.shape[1]/((int)(np.sqrt(x_train.shape[1]))))))
#edge_arr = (np.array([feature.canny(z) for z in temp])).astype(float)
#edge_arr = edge_arr.reshape(x_train.shape[0],x_train.shape[1])
#x_train = np.append(x_train,edge_arr,axis = 1)

#temp = x_test.reshape(x_test.shape[0],(int)(np.sqrt(x_test.shape[1])), (int)(np.ceil(x_test.shape[1]/((int)(np.sqrt(x_test.shape[1]))))))
#edge_arr = (np.array([feature.canny(z) for z in temp])).astype(float)
#edge_arr = edge_arr.reshape(x_test.shape[0],x_test.shape[1])
#x_test = np.append(x_test,edge_arr,axis = 1)

validation_split = (int)(x_train.shape[0]*0.9)
x_train_train = x_train[:(validation_split),:]
x_train_val = x_train[validation_split:,:] 

y_train_train = y_train[:validation_split]
y_train_val = y_train[validation_split:]



normalizer = preprocessor()
normalizer.fit(x_train)
x_train = normalizer.normalize(x_train)
x_test = normalizer.normalize(x_test)
x_train_train = normalizer.normalize(x_train_train)
x_train_val = normalizer.normalize(x_train_val)


num_classes = np.unique(y_train).shape[0]
y_encoder = one_hot_encoder(np.sort(np.unique(y_train)))
y_train = y_encoder.encode(y_train)
y_train_val = y_encoder.encode(y_train_val)
y_train_train = y_encoder.encode(y_train_train)


#parameters=[2,1,40000,50,[100],0.1,sigmoid]
#parameters=[1,0.01,25000,50,[100],0.1,sigmoid]
#parameters=[2,1,20000,50,[100,100],0.1,sigmoid]
#parameters=[2,1,20000,50,[100,100],0.1,softplus]
parameters=[2,5,70000,100,[100],0.1,sigmoid]#31 div by epoch~ 33 div by iter
parameters=[2,5,70000,100,[100],1,sigmoid]#bad div by epoch ~ 30 div by iter


parameters=[2,5,70000,100,[200],1,sigmoid]#bad div by epoch ~ 30 div by iter

#parameters=[2,10,70000,100,[100],0.1,sigmoid]#30 div by epoch, div by iter -> overfit max 31


activation = parameters[6]
regularization_parameter = parameters[5]
batch_size = parameters[3]
base_rate = parameters[1]
num_iters = parameters[2]
learning_type = parameters[0]
layer_shapes = [x_train.shape[1]]+parameters[4]+[num_classes]

num_batches = (int)(x_train_train.shape[0]/batch_size)
if( x_train_train.shape[0] > batch_size*num_batches):
    num_batches = num_batches + 1
    
nn = neural_network(layer_shapes,activation)

train_val_acc = []
for i in range(num_iters):
    j = i%num_batches
    #print(i+1)
    #for j in range(num_batches):
    x = x_train_train[ j*batch_size : (j+1)*batch_size]
    y = y_train_train[ j*batch_size : (j+1)*batch_size]
    
    nn.forward(x)
    learning_rate = base_rate
    epoch_number = (int)(i/num_batches) + 1
    if(learning_type == 2):
        learning_rate = learning_rate/np.sqrt(i+1)
    nn.backpropagate(y,learning_rate,regularization_parameter)
    if (i+1)%1000==0:    
        x = x_train_train
        y = y_train_train
        nn.forward(x)
        
        y_train_actual = np.array(y_encoder.decode(y))
        y_train_pred = np.array(y_encoder.decode(nn.y_pred))
        train_accuracy = np.sum(y_train_actual == y_train_pred)/x.shape[0]
        
        x = x_train_val
        y = y_train_val
        
        nn.forward(x)
        y_val_actual = np.array(y_encoder.decode(y))
        y_val_pred = np.array(y_encoder.decode(nn.y_pred))
        val_accuracy = np.sum(y_val_pred == y_val_actual)/x.shape[0]
        
        train_val_acc.append([i+1,train_accuracy,val_accuracy])
        print("iteration:{} train_acc:{} val_acc:{}".format(i+1,round(train_accuracy,3),round(val_accuracy,3)))
    
    

#print(num_iters)
#p = np.array(train_val_acc)
#import matplotlib.pyplot as plt

#plt.plot(p[:,0],p[:,1],label='train')
#plt.plot(p[:,0],p[:,2],label='validation')
#plt.show()
# print(x_train.shape)
# print(x_test.shape)
nn.forward(x_test)
y_pred = y_encoder.decode(nn.y_pred)
f=open(outputfile,'w+')
for prediction in y_pred:
    f.write("{}\n".format(prediction))
f.close()

# f = open(weightfile,'w+')
# for i in range(nn.weights.shape[0]):
#     for j in range(nn.weights[i].shape[0]):
#         for k in range(nn.weights[i].shape[1]):
#             f.write("{}\n".format(nn.weights[i][j][k]))
# f.close()