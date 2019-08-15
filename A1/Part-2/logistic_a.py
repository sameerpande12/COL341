import csv
import numpy as np
import sys

# trainfile=sys.argv[2]
# testfile = sys.argv[3]
# paramfile = sys.argv[4]
# outputfile = sys.argv[5]
# weightfile = sys.argv[6]
trainfile = "train.csv"



def predict(X,Y,W):
    h_matrix = np.exp(np.dot(X,W))
    h_matrix = h_matrix/(np.sum(h_matrix,axis=1))
    return h_matrix

def getLoss(X,Y,W):
    y_pred = predict(X,Y,W)
    y_pred = np.log(y_pred)
    loss = (1.0/2*X.shape[0])*np.sum( y_pred * Y )


def sgd(X,Y,W,alpha,num_iters):#alpha is learning rate
    x_transpose = np.transpose(X)
    for i in range(num_iters):
        j = i%(W.shape[1])
        y_pred = predict(X,Y,W)
        W[:,j] = W[:,j] + alpha/(2.0*X.shape[0]) * np.dot(x_transpose, (Y[:,j] - y_pred[:,j]))
        print("iteration: {}, Loss: {}".format(i+1,getLoss(X,Y,W)))
    return W


class one_hot_encoder:

    def __init__(self,x):
        self.labels_arr = []
        for i in range(x.shape[1]):
            self.labels_arr.append(np.unique(x[:,i]))
#        print(len(self.labels_arr))
#        print("Done creating encoder")


    def encode(self,x_input):

        x_output =[]

        for i in range(len(self.labels_arr)):
            for label in self.labels_arr[i]:
                #print(i)
                x_output.append((x_input[:,i]==label).astype(np.float64))

        x_output = np.transpose(np.array(x_output))
        return x_output




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


w = np.random.random([x_train.shape[1],y_train.shape[1]])

w = sgd(x_train,y_train,w,0.1,1000)
print(w)
