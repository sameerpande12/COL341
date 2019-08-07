import csv
import numpy as np
import sys


def getLoss(x,y,w,l_type,lamda=0):
    (n,m) = x.shape

    if l_type == 'a':
        y_xw = y - np.dot(x,w)
        loss = (0.5/n)*np.sum( y_xw ** 2)
        return loss
    elif l_type == 'b':
        y_xw = y - np.dot(x,w)
        loss = (0.5/n)*np.sum( y_xw ** 2) + (0.5*lamda)* (np.sum(w**2))
        return loss

def cross_validate(x_train,y_train,lambdas,k):
    fold_size = (int)(x_train.shape[0]/k)
    accuracy = 0

    avg_errors = []

    for lamda in lambdas:
        errors = []
        for i in range(k):
            x_train_k = np.concatenate((x_train[:(i*fold_size)],x_train[((i+1)*fold_size):]))
            x_validate_k = x_train[i*fold_size : (i+1)*fold_size]

            y_train_k = np.concatenate((y_train[:(i*fold_size)],y_train[((i+1)*fold_size):]))
            y_validate_k = y_train[i*fold_size : (i+1)*fold_size]

            w = trainB(x_train_k,y_train_k,lamda)

            y_validate_pred = (np.dot(x_validate_k,w))
            errors.append(getLoss(x_validate_k,y_validate_k,w,'b',lamda))

        avg_errors.append( np.mean(np.array(errors)) )

    # print(avg_errors)
    index = avg_errors.index(min(avg_errors))
    # optimum_lamda = lambdas[index]
    return index

def printList(arr,fname):
    f = open(fname,'w+')
    for val in arr:
        f.write(str(val)+"\n")
    f.close()



mode = sys.argv[1]

if mode == 'a':

    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]

elif mode == 'b':
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    regularization_file = sys.argv[4]
    outputfile = sys.argv[5]
    weightfile = sys.argv[6]


elif mode == 'c':

    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outputfile = sys.argv[4]

x_train=[]
x_test=[]
y_train=[]
ytest=[]


with open(trainfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        row = [float(i) for i in row]
        x_train.append(row)

with open(testfile,'r') as filename:
    csvreader = csv.reader(filename)
    for row in csvreader:
        row = [float(i) for i in row]
        x_test.append(row)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = x_train[:,x_train.shape[1]-1]
# ytest = x_test[:,x_test.shape[1]-1]

x_train = x_train[:,:(x_train.shape[1]-1)]
# x_test = x_test[:,:(x_test.shape[1]-1)]

ones = np.ones((x_train.shape[0],1))
x_train = np.append(ones,x_train,axis=1)

ones = np.ones((x_test.shape[0],1))
x_test = np.append(ones,x_test,axis=1)


if mode == 'a':
    def trainA(x_train,y_train):
        xTx = np.dot(x_train.transpose(),x_train)
        w = np.dot(np.dot(np.linalg.pinv(xTx), x_train.transpose()),y_train)
        return w

    w = trainA(x_train,y_train)
    # y_test_pred = (np.round(np.dot(x_test,w))).astype('int64')
    y_test_pred = np.dot(x_test,w)
    printList(y_test_pred,outputfile)
    printList(w,weightfile)



if mode == 'b':
    def trainB(x_train,y_train,lamda):
        n = x_train.shape[0]
        m = x_train.shape[1]
        xTx = np.dot(x_train.transpose(),x_train)
        invTerm = np.linalg.pinv( (1.0/n) * xTx + (1.0/n) *lamda* np.identity(m))
        xTy = np.dot(x_train.transpose(),y_train)
        w = (1.0/n) * np.dot(invTerm,xTy)
        return w

    lambdas = []
    with open(regularization_file,'r') as filename:
        csvreader = csv.reader(filename)
        for row in csvreader:
            if(len(row)>0):
                lambdas.append( float( row[0]))

    # lambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    k = 10
    index = cross_validate(x_train,y_train,lambdas,k)
    optimum_lamda = lambdas[index]
    print(str(optimum_lamda))

    w = trainB(x_train,y_train,optimum_lamda)
    y_test_pred = (np.dot(x_test,w))
    printList(y_test_pred,outputfile)
    printList(w,weightfile)


if mode == 'c':
    def trainC(x_train,y_train):
        xTx = np.dot(x_train.transpose(),x_train)
        w = np.dot(np.dot(np.linalg.pinv(xTx), x_train.transpose()),y_train)
        return w

    w = trainC(x_train,y_train)
    y_test_pred = (np.round(np.dot(x_test,w))).astype('int64')
    printList(y_test_pred,outputfile)


# f = open("y_train.csv",'w+')
# for i in range(len(y_train)):
#     f.write(str((int)(y_train[i]))+"\n")
# f.close()
#
# f = open("y_train_pred.csv",'w+')
# y_train_pred = ((np.dot(x_train,w))).astype('int64')
# for i in range(len(y_train_pred)):
#     f.write(str(y_train_pred[i])+"\n")
# f.close()


# loss = (np.sum((y_train_pred - y_train)**2.0))/(2*(x_train.shape[0]))
# print(loss)
