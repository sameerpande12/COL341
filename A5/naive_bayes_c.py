import numpy as np
import sys
import pandas as pd
#import time
def cleanse_String(line):
    line = line.strip()
    line = line.lower()
    line =  line.replace('.',' ') 
    line =  line.replace(',',' ') 
    line =  line.replace('\\','') ##make sure don't replace "don\'t" by "dont"
    line =  line.replace('\'','') 
    line =  line.replace('"',' ') 
    line =  line.replace('-',' ') 
    line =  line.replace('!',' ') 
    line =  line.replace(':',' ') 
    line =  line.replace('(',' ') 
    line =  line.replace(')',' ') 
    line =  line.replace('*',' ') 
    line =  line.replace('?',' ') 
    line =  line.replace('=',' ') 
    line =  line.replace('/',' ')
    line =  line.replace('\n',' ')
    return line


#trainfile = sys.argv[1]
trainfile = 'traindata.csv'

#testfilename = sys.argv[2]
testfilename = 'testdata.csv'

#outputfile = sys.argv[3]
outputfile = 'output.txt'

x_train = pd.read_csv(trainfile,sep=',',dtype=str).values
x_train = x_train[1:]
for i in range(len(x_train)):
    x_train[i][0] = cleanse_String(x_train[i][0])


positive = x_train[x_train[:,-1]=='positive']
negative = x_train[x_train[:,-1]=='negative']
num_pos = len(positive)
num_neg = len(negative)


prob_pos = len(positive)/(len(positive) + len(negative))
prob_neg = 1- prob_pos

all_words = {}
pos_dict = {}
pos_word_count = 0

for( line,pred ) in positive:
    words = line.split()
    pos_word_count = pos_word_count + len(words)
    #unique_words = np.unique(np.array(words))
    
    for word in words:

        all_words[word] = 1
        if not(word in pos_dict):
            pos_dict[word] = 1
        else:
            pos_dict[word] = pos_dict[word] + 1

neg_dict = {}
neg_word_count = 0
for( line,pred ) in negative:
    words = line.split()
    neg_word_count = neg_word_count + len(words)
    #unique_words = np.unique(np.array(words))
    for word in words:
    
        all_words[word]=1
        if not(word in neg_dict):
            neg_dict[word] = 1
        else:
            neg_dict[word] = neg_dict[word] + 1
            
    
def phi_pos(word): ## give p(word|y=1)
    numerator = 1
    if word in pos_dict:
        numerator = 1 + pos_dict[word]
    return (numerator)/(pos_word_count + 2)    
    #return (numerator)/(num_pos + 2)

def phi_neg(word):
    numerator = 1
    if word in neg_dict:
        numerator = 1 + neg_dict[word]
    return numerator/(neg_word_count + 2)    
    #return numerator/(num_neg +2)     
            

phi_pos_sum = 0
phi_neg_sum = 0

phi_pos_complement_sum = 0
phi_neg_complement_sum = 0
for word in all_words:
    phi_pos_sum = phi_pos_sum + np.log(phi_pos(word))
    phi_pos_complement_sum = phi_pos_complement_sum + np.log ( 1- phi_pos(word))
    
    phi_neg_sum = phi_neg_sum + np.log(phi_neg(word))
    phi_neg_complement_sum = phi_neg_complement_sum + np.log( 1- phi_neg(word))
    
def getLogProb(line,pred):###P(line/y)
    line = cleanse_String(line)
    words = line.split()
    words = np.unique(np.array(words))
    answer = 0.0
    
    if pred==1:
        answer = phi_pos_complement_sum
        for word in words:
            answer = answer + np.log(phi_pos(word)) - np.log(1-phi_pos(word))
    else:
        answer = phi_neg_complement_sum
        for word in words:
            answer = answer + np.log(phi_neg(word)) - np.log(1-phi_neg(word))
     
            
    return answer



all_bigrams = {}
pos_dict_bi = {}
neg_dict_bi = {}

for (line, pred) in positive:
    words = line.split()
    bigrams = []
    for i in range(len(words)-1):
        bigrams.append([words[i]+' '+words[i+1]])
    bigrams = np.array(bigrams)
    bigrams = np.unique(bigrams)
    
    for bigram in bigrams:
        all_bigrams[bigram] = 1
        if not(bigram in pos_dict_bi):
            pos_dict_bi[bigram] = 1
        else:
            pos_dict_bi[bigram] = pos_dict_bi[bigram] + 1        


for (line, pred) in negative:
    words = line.split()
    bigrams = []
    for i in range(len(words)-1):
        bigrams.append([words[i]+' '+words[i+1]])
    bigrams = np.array(bigrams)
    bigrams = np.unique(bigrams)
    
    for bigram in bigrams:
        all_bigrams[bigram] = 1
        if not(bigram in neg_dict_bi):
            neg_dict_bi[bigram] = 1
        else:
            neg_dict_bi[bigram] = neg_dict_bi[bigram] + 1        


def phi_pos_bi(bigram):
    numerator = 1
    if (bigram in pos_dict_bi):
        numerator = 1 + pos_dict_bi[bigram]
    
    return (numerator)/(num_pos + 2)

def phi_neg_bi(bigram):
    numerator = 1
    if (bigram in neg_dict_bi):
        numerator = 1 + neg_dict_bi[bigram]
    
    return (numerator)/(num_neg + 2)


phi_pos_sum_bi = 0
phi_neg_sum_bi = 0

phi_pos_complement_sum_bi = 0
phi_neg_complement_sum_bi = 0
for bigram in all_bigrams:
    phi_pos_sum_bi = phi_pos_sum_bi + np.log(phi_pos_bi(bigram))
    phi_pos_complement_sum_bi = phi_pos_complement_sum_bi + np.log ( 1- phi_pos_bi(bigram))
    
    phi_neg_sum_bi = phi_neg_sum_bi + np.log(phi_neg_bi(bigram))
    phi_neg_complement_sum_bi = phi_neg_complement_sum_bi + np.log( 1- phi_neg_bi(bigram))



def getLogProbBi(line,pred):
    line = cleanse_String(line)
    words = line.split()
    bigrams = []
    for i in range(len(words)-1):
        bigrams.append([words[i]+' '+words[i+1]])
    bigrams = np.array(bigrams)
    bigrams = np.unique(bigrams)
    if pred==1:
        answer = phi_pos_complement_sum_bi
        for word in words:
            answer = answer + np.log(phi_pos_bi(bigram)) - np.log(1-phi_pos_bi(bigram))
    else:
        answer = phi_neg_complement_sum_bi
        for word in words:
            answer = answer + np.log(phi_neg_bi(word)) - np.log(1-phi_neg_bi(word))
    
    return answer

def predict(line):
    pos_measure = getLogProbBi(line,1) + np.log(prob_pos) + getLogProb(line,1)
    neg_measure = getLogProbBi(line,0) + np.log(prob_neg) + getLogProb(line,0)
    
    #pos_measure = np.log(prob_pos) + getLogProb(line,1)
    #neg_measure = np.log(prob_neg) + getLogProb(line,0)
    
    if pos_measure > neg_measure:
        return 1
    else:
        return 0    
"""
count = 0
correct = 0
for (line,pred) in x_train:
    if pred == 'positive':
        pred = 1
    else:
        pred = 0
    
    value = predict(line)
    if(pred == predict(line)):
        correct = correct + 1
    count = count + 1
    
    print(value,correct,count, correct/count)
    
"""
#============================================================================
        #Part of code for test purposes
#============================================================================
        

x_test = pd.read_csv(testfilename,sep=',').values

f = open(outputfile,'w+')
count = 0
for i in range(len(x_test)):
    line = x_test[i][0]
    f.write(str(predict(line)))
    f.write('\n')
    count = count + 1
    #print(count)
f.close()

