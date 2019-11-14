import numpy as np
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
from nltk.stem import PorterStemmer





stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}
 

ps = PorterStemmer()


def cleanse_String(line):
    line = line.lower()
    line =  line.replace('.',' ') 
    line =  line.replace(',',' ') 
    line =  line.replace('\\','') ##make sure don't replace "don\'t" by "don't"
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
    
    word_tokens = line.split()
    words_filtered = [ ps.stem(word) for word in word_tokens if not word in stop_words]
    line = ''
    for word in words_filtered:
        line = line +word + ' '
    
    line.strip()
    
    return line


trainfile = sys.argv[1]
testfilename = sys.argv[2]
outputfile = sys.argv[3]




x_train = pd.read_csv(trainfile,sep=',',dtype=str).values
x_train = x_train[1:]
for i in range(len(x_train)):
    x_train[i][0] = cleanse_String(x_train[i][0])
    

x_test = pd.read_csv(testfilename,sep=',').values


positive = x_train[x_train[:,-1]=='positive']
negative = x_train[x_train[:,-1]=='negative']
num_pos = len(positive)
num_neg = len(negative)




prob_pos = len(positive)/(len(positive) + len(negative))
prob_neg = 1- prob_pos


all_words = {}
pos_dict = {}

for( line,pred ) in positive:
    words = line.split()
    
    unique_words = np.unique(np.array(words))
    for word in unique_words:
        all_words[word] = 1
        if not(word in pos_dict):
            pos_dict[word] = 1
        else:
            pos_dict[word] = pos_dict[word] + 1

neg_dict = {}

for( line,pred ) in negative:
    words = line.split()
    unique_words = np.unique(np.array(words))
    for word in unique_words:
        all_words[word]=1
        if not(word in neg_dict):
            neg_dict[word] = 1
        else:
            neg_dict[word] = neg_dict[word] + 1
            
    
def phi_pos(word): ## give p(word|y=1)
    numerator = 1
    if word in pos_dict:
        numerator = 1 + pos_dict[word]
        
    return (numerator)/(num_pos + 2)

def phi_neg(word):
    numerator = 1
    if word in neg_dict:
        numerator = 1 + neg_dict[word]
        
    return numerator/(num_neg +2)     

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


def predict(line):
    line = cleanse_String(line)
    pos_measure = getLogProb(line,1) + np.log(prob_pos)
    neg_measure = getLogProb(line,0) + np.log(prob_neg)
    
    if pos_measure > neg_measure:
        return 1
    else:
        return 0


            

phi_pos_sum = 0
phi_neg_sum = 0

phi_pos_complement_sum = 0
phi_neg_complement_sum = 0


for word in all_words:
    phi_pos_sum = phi_pos_sum + np.log(phi_pos(word))
    phi_pos_complement_sum = phi_pos_complement_sum + np.log ( 1- phi_pos(word))
    
    phi_neg_sum = phi_neg_sum + np.log(phi_neg(word))
    phi_neg_complement_sum = phi_neg_complement_sum + np.log( 1- phi_neg(word))
    






f = open(outputfile,'w+')
count = 0
for i in range(len(x_test)):
    line = x_test[i][0]
    f.write(str(predict(line)))
    f.write('\n')
    # count = count + 1
    # print(count)
f.close()

