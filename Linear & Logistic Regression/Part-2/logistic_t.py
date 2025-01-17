import csv
import numpy as np
import sys
#from scipy import special
trainfile='train.csv'
testfile = 'test_X.csv'
#paramfile = 'param_a2.txt'
#outputfile = sys.argv[4]
#weightfile = sys.argv[5]
#trainfile = "train.csv"

def printToFile(fname,arr):
    f = open(fname,'w+')
    f.write(",".join(arr))
    f.close()

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
    loss = (1.0/(X.shape[0]))*np.sum( y_pred * Y )
    return loss

#def gradient_col(X,Y,W,col_no):
#    y_pred_col = np.dot(X,W[:,col_no])
#    return np.dot(X.T,y_pred_col-Y[:,j])

def batch_sgd(X,Y,W,args):#learning_rate is learning rate
    start_time = time.time()
    if(args[0]==1):
        learning_rate = args[1]

    elif(args[0]==2):
        learning_rate = args[1][0]
        seed = args[1][1]

    else:
        learning_rate = args[1][0]
        alpha= args[1][1]
        beta = args[1][2]

    num_iters = (int)(args[2])
    batch_size = (int)(args[3])
    num_batches = (int)(X.shape[0]/batch_size)
    if(num_batches * batch_size < X.shape[0]):
        num_batches += 1
        
    for i in range(num_iters):
        if(i%10==0 and time.time()-start_time > 480):
            break
        for iterator in range(num_batches):
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
                    
                
            W = W - learning_rate *gradient
            
            if(args[0]==2):
                learning_rate = learning_rate/np.sqrt(seed)
        print("iteration:{}, Loss:{}".format(i+1,getLoss(X,Y,W)))
        
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


arguments= [1,1,50000,100]

x_train = []
with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        x_train.append(row)


x_train = np.array(x_train)
y_train = np.array([x_train[:,x_train.shape[1]-1]])
y_train = np.transpose(y_train)
x_train = x_train[:,:(x_train.shape[1]-1)]

x_test = x_train[5400:]
y_test = y_train[5400:]
x_train = x_train[:5400]
y_train= y_train[:5400]


input_labels_arr = [['usual', 'pretentious', 'great_pret'],
['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
['complete', 'completed', 'incomplete', 'foster'],
['1', '2', '3', 'more'],
['convenient', 'less_conv', 'critical'],
['convenient', 'inconv'],
['non-prob', 'slightly_prob', 'problematic'],
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

w = np.random.random([x_train.shape[1],y_train.shape[1]])* np.sqrt(2)/(x_train.shape[1]*y_train.shape[1])

w = batch_sgd(x_train,y_train,w,arguments)


y_test_output = np.array([output_encoder.decode(predict(x_test,w))]).T
y_test = np.array([output_encoder.decode(y_test)]).T
#y_test = np.array([y_test]).T
accuracy = (np.sum(z == y_test_output))/y_test.shape[0]
print(accuracy)
#print (np.array(y_train_saved).shape)
#print (np.array(y_train_test).shape)
#print(train_accuracy)
print(arguments[2])

#np.savetxt(weightfile,w,delimiter=',')

#printToFile(outputfile,y_test_output)
#print(w)


#for i in range(num_iters):
    #    print("iteration: {}, Loss: {}".format(i,getLoss(X,Y,W)))
    #    j = i%(W.shape[0])
    #    y_pred = predict(X,W)
    #    W[j,:] = W[j,:] + learning_rate/(2.0*X.shape[0]) * np.dot(y_transpose-y_pred.T, X[:,j])
