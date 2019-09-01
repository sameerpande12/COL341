import csv
import numpy as np
import sys
import time
from sklearn.metrics import confusion_matrix,f1_score
#trainfile=sys.argv[1]
#testfile = sys.argv[2]
trainfile = 'train.csv'
testfile = 'test_X.csv'
#paramfile = sys.argv[3]
#outputfile = sys.argv[4]
#weightfile = sys.argv[5]
outputfile = ""
weightfile=""


def printToFile(fname,arr):
    f = open(fname,'w+')
    f.write(",".join(arr))
    f.close()

def predict(X,W):
    h_matrix = np.exp(np.dot(X,W))
    col = np.sum(h_matrix,axis=1)
    col = col.reshape(col.shape[0],1)
    h_matrix = h_matrix/col

    return h_matrix

def getLoss(X,Y,W):
    y_pred = predict(X,W)
    y_pred = -np.log(y_pred)
    loss = (1.0/(X.shape[0]))*np.sum( y_pred * Y )
    return loss


def batch_sgd(X,Y,W,args):#learning_rate is learning rate
    
    
    if(args[0]==1):
        learning_rate = args[1]
        base_rate = learning_rate

    elif(args[0]==2):
        base_rate = args[1][0]
    
        

    else:
        learning_rate = args[1][0]
        base_rate = learning_rate
        alpha= args[1][1]
        beta = args[1][2]

    num_iters = (int)(args[2])
    batch_size = (int)(args[3])
    file = open("Data_L/learning_data_{}.csv".format(base_rate),'w+')
    num_batches = (int)(X.shape[0]/batch_size)
    if(num_batches * batch_size < X.shape[0]):
        num_batches += 1
    
    count  =0
    for i in range(num_iters):
        #print("iteration:{}, num_batches:{}".format(i+1,num_batches))
        print("Base_Rate: {}, iterations: {}".format(base_rate,i))
        
        
        if(args[0]==2):
                learning_rate = base_rate/np.sqrt(i+1)
        for iterator in range(num_batches):
            #print("Batch number: "+str(iterator))
            x = X[(iterator * batch_size):((iterator+1)*batch_size),:]
            y = Y[(iterator * batch_size):((iterator+1)*batch_size),:]
        
            y_pred = predict(x,W)
            gradient = -np.dot(x.T, (y - y_pred))/(x.shape[0])
            if(args[0]==3):
                magnitude = np.sqrt(np.sum(gradient**2))
                direction = -gradient/magnitude
                loss = getLoss(x,y,W)
                
                while True:
                    diff = getLoss(x,y,W+learning_rate*direction) - loss
                    if diff > learning_rate * alpha * magnitude:
                        learning_rate = learning_rate * beta
                    else:
                        break
                    
            count = count + batch_size
            W = W - learning_rate *gradient
        
        file.write("{},{}\n".format(count,getLoss(X,Y,W)))
            
            #loss = getLoss(x,y,W)
            
    
    file.close()
    return W
    


class one_hot_encoder:

    def __init__(self,labels_arr):
        self.labels_arr = labels_arr

    def encode(self,x_input):

        x_output =[]

        for i in range(len(self.labels_arr)):
            for label in self.labels_arr[i]:

                x_output.append((x_input[:,i]==label).astype(np.float64))

        x_output = np.transpose(np.array(x_output))
        return x_output

    def decode(self,y_encoded):
        y_out = np.argmax(y_encoded,axis = 1)
        y_decoded = []
        for i in range(len(self.labels_arr)):
            y_decoded = y_decoded + [self.labels_arr[i][j] for j in y_out]

        return y_decoded


arguments= []


x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        x_train.append(row)

x_test  = []
with open(testfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        x_test.append(row)

x_train = np.array(x_train)
y_train = np.array([x_train[:,x_train.shape[1]-1]])
y_train = np.transpose(y_train)
x_train = x_train[:,:(x_train.shape[1]-1)]
x_test = np.array(x_test)


input_labels_arr = [['usual', 'pretentious', 'great_pret'],
['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
['complete', 'completed', 'incomplete', 'foster'],
['1', '2', '3', 'more'],
['convenient', 'less_conv', 'critical'],
['convenient', 'inconv'],
['nonprob', 'slightly_prob', 'problematic'],
['recommended', 'priority', 'not_recom']]
input_encoder = one_hot_encoder(input_labels_arr)
output_labels_arr = [['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']]
output_encoder = one_hot_encoder(output_labels_arr)


x_train = input_encoder.encode(x_train)
x_test = input_encoder.encode(x_test)

y_train_saved = y_train
y_train = output_encoder.encode(y_train)

ones = np.ones((x_train.shape[0],1))
x_train = np.append(ones,x_train,axis=1)
ones = np.ones((x_test.shape[0],1))
x_test = np.append(ones,x_test,axis=1)

learning_rates = [1,0.1,0.01,0.001,0.0001,0.00001]
confusion_matrices = {}
microf1 = {}
macrof1 = {}
f1 = {}
for learning_rate in learning_rates:
    #arguments[3]=batch_size
    #arguments[2]=1000
    #arguments[0]=1
    #arguments[1]=[0.1]
    batch_size = 3000
    if batch_size >= 1000: 
        arguments = [2,[learning_rate],500,batch_size]
    else:
        arguments = [2,[learning_rate],500,batch_size]
    #arguments[1]= [0.1,0.3,0.5]
    w = np.random.random([x_train.shape[1],y_train.shape[1]])* np.sqrt(2)/(x_train.shape[1]*y_train.shape[1])
    w = batch_sgd(x_train,y_train,w,arguments)
    y_test_output = output_encoder.decode(predict(x_test,w))
    y_train_test = output_encoder.decode(predict(x_train,w))
    y_train_test = np.array([y_train_test]).T
    
    confusion_matrices[learning_rate] = confusion_matrix(y_train_saved,y_train_test,labels=output_labels_arr[0])
    microf1[learning_rate] = f1_score(y_train_saved,y_train_test,average='micro',labels=output_labels_arr[0])
    macrof1[learning_rate] = f1_score(y_train_saved,y_train_test,average='macro',labels=output_labels_arr[0])
    f1[learning_rate] = f1_score(y_train_saved,y_train_test,average=None,labels=output_labels_arr[0])
    
    
    train_accuracy = (np.sum(y_train_test == y_train_saved))/y_train.shape[0]
    
    weightfile = "Data_L/weight_{}.csv".format(learning_rate)
    outputfile = "Data_L/output_{}.csv".format(learning_rate)
    np.savetxt(weightfile,w,delimiter=',')
    printToFile(outputfile,y_test_output)