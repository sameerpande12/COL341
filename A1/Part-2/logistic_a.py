import csv
import numpy as np
import sys
#from scipy import special
#trainfile=sys.argv[2]
#testfile = sys.argv[3]
#paramfile = sys.argv[4]
#outputfile = sys.argv[5]
#weightfile = sys.argv[6]
trainfile = "train.csv"



def predict(X,W):
    h_matrix = np.exp(np.dot(X,W))
    col = np.sum(h_matrix,axis=1)
    col = col.reshape(col.shape[0],1)
    h_matrix = h_matrix/col
    #h_matrix = special.softmax(np.dot(X,W))
    return h_matrix

def getLoss(X,Y,W):
    y_pred = predict(X,W)
    y_pred = -np.log(y_pred)
    loss = (1.0/(2*X.shape[0]))*np.sum( y_pred * Y )
    return loss

#def gradient_col(X,Y,W,col_no):
#    y_pred_col = np.dot(X,W[:,col_no])
#    return np.dot(X.T,y_pred_col-Y[:,j])

def sgd(X,Y,W,alpha,num_iters):#alpha is learning rate
    x_transpose = np.transpose(X)
    for i in range(num_iters):
        j = i%(W.shape[1])
        y_pred = predict(X,W)
        gradient = -np.dot(x_transpose, (Y[:,j] - y_pred[:,j]))
        direction = -gradient/np.sqrt((np.sum(gradient**2)))
        W[:,j] = W[:,j] - alpha/(2.0*X.shape[0]) *gradient
        print("iteration: {}, Loss: {}".format(i+1,getLoss(X,Y,W)))
    return W
    #y_transpose = np.transpose(Y)    
    #for i in range(num_iters):
       #print("iteration: {}, Loss: {}".format(i,getLoss(X,Y,W)))
    #   j = i%(W.shape[0])
    #   y_pred = predict(X,W)
    #   W[j,:] = W[j,:] + alpha/(2.0*X.shape[0]) * np.dot(y_transpose-y_pred.T, X[:,j])

    
    return W


class one_hot_encoder:

    def __init__(self,x):
        self.labels_arr = []
        for i in range(x.shape[1]):
            self.labels_arr.append(np.unique(x[:,i]))
        
        

    def encode(self,x_input):

        x_output =[]

        for i in range(len(self.labels_arr)):
            for label in self.labels_arr[i]:
             
                x_output.append((x_input[:,i]==label).astype(np.float64))

        x_output = np.transpose(np.array(x_output))
        return x_output
    
    def decode(self,y_encoded):
        y_out = np.argmax(y_encoded,axis = 1)
        y_out = [self.labels_arr[i] for i in y_out]
        



x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        x_train.append(row)

x_train = np.array(x_train)
y_train = np.array([x_train[:,x_train.shape[1]-1]])
y_train = np.transpose(y_train)
x_train = x_train[:,:(x_train.shape[1]-1)]


input_encoder = one_hot_encoder(x_train)
output_encoder = one_hot_encoder(y_train)


x_train = input_encoder.encode(x_train)
y_train = output_encoder.encode(y_train)

ones = np.ones((x_train.shape[0],1))
x_train = np.append(ones,x_train,axis=1)


w = np.random.random([x_train.shape[1],y_train.shape[1]])* np.sqrt(2)/(x_train.shape[1]*y_train.shape[1])

w = sgd(x_train,y_train,w,5,1000)
#print(w)


#for i in range(num_iters):
    #    print("iteration: {}, Loss: {}".format(i,getLoss(X,Y,W)))
    #    j = i%(W.shape[0])
    #    y_pred = predict(X,W)
    #    W[j,:] = W[j,:] + alpha/(2.0*X.shape[0]) * np.dot(y_transpose-y_pred.T, X[:,j])
