import numpy as np
import csv
import time
column_headers = ['Age', 'Work Class', 'Fnlwgt', 'Education', 'Education Number',
       'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex',
       'Capital Gain', 'Capital Loss', 'Hour per Week',
       'Native Country', 'Rich?']

col_types={
    'Age':'continuous',
    'Work Class': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc','Self-emp-not-inc', 'State-gov', 'Without-pay','Never-worked'],
    'Fnlwgt':'continuous',
    'Education':['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
       'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
       'Masters', 'Preschool', 'Prof-school', 'Some-college'],
    'Education Number':'continuous',
    'Marital Status':['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
    'Occupation':['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
       'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
       'Other-service', 'Priv-house-serv', 'Prof-specialty',
       'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'],
    'Relationship':['Husband', 'Not-in-family', 'Other-relative', 'Own-child',
       'Unmarried', 'Wife'],
    'Race':['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
    'Sex':['Male','Female'],
    'Capital Gain':'continuous',
    'Capital Loss':'continuous',
    'Hour per Week':'continuous',
    'Native Country':['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba',
       'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England',
       'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',
       'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',
       'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
       'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',
       'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',
       'Yugoslavia'],
    'Rich?':[0,1]
}

pruneDataFile = "data_prune.csv"
pruneFile = open(pruneDataFile,'w+')

def readfile(trainfile):
    train_data = []
    headings = []
    with open(trainfile,'r') as filename:
        csvreader = csv.reader(filename)
        isFirst = True
        for row in csvreader:
            if isFirst:
                headings = row
                headings = [heading.strip() for heading in headings]
                isFirst = False
            else:
                for i in range(len(headings)):
                    row[i] = row[i].strip()
                    if col_types[headings[i]] == 'continuous':
                        row[i] = (float)(row[i])
                    elif i == len(row) - 1:
                        row[i] = (int)(row[i])
                train_data.append(row)
                    
    train_data = np.array(train_data,dtype='object')
    
    for heading in headings:
        if not(heading in column_headers):
            print("column headings in-accurate")
    return (headings,train_data)

trainfile = "DT_data/train.csv"
(headings,train_data) = readfile(trainfile)

column_headers = headings



validationfile = 'DT_data/valid.csv'
(headings,val_data) = readfile(validationfile)

testfile = "DT_data/test_public.csv"
(headings,test_data) = readfile(testfile)

root = None##definition

def Entropy(y):
    #print("Entropy")
    if len(y)==0:
        return 0
    unique,counts = np.unique(y,return_counts=True)
    h = 0
    for i in range(unique.size):
        p = counts[i]/np.sum(counts)
        if p > 0:
            h = h - p*np.log2(p)
    return h

def Gini(y):
    
    if len(y) == 0:
        return 0
    unique,counts = np.unique(y,return_counts = True)
    g = 0
    probs = counts/np.sum(counts)
    g = 1 - np.sum(probs ** 2)
    return g
        
        
            

def GainRatio(data,splitIndex,impurityFunction):
    ##MAKE SURE THAT DATA IS NOT EMPTY
    #print("Using GainRatio")
    iv = IntrinsicVal(data,splitIndex)
    
    return InfoGain(data,splitIndex,impurityFunction)/iv
    
def IntrinsicVal(data,splitIndex):
    if(len(data)==0):
        return 0
    col_name = column_headers[splitIndex]
    iv = 0
    if col_types[col_name] == 'continuous':
        median = np.median(data[:,splitIndex])
        l_child = len(data[data[:,splitIndex]<=median])
        r_child = len(data[data[:,splitIndex]>median])
        p_left = l_child /(len(data))
        p_right = r_child/(len(data))
        
        iv =  - p_left * np.log2(p_left) - p_right * np.log2(p_right)
    
    else:
        attributes = col_types[col_name]
        iv = 0
        for attribute in attributes:
            child = len(data[data[:,splitIndex]==attribute])
            p_child = child/len(data)
            if p_child > 0:
                iv = iv - p_child * np.log2(p_child)
        
    return iv
    
def InfoGain(data,splitIndex,impurityFunction):
    #print(impurityFunction)
    IG = impurityFunction(data[:,-1])
    col_name = column_headers[splitIndex]
    if col_types[col_name] == 'continuous':
        
        median = np.median(data[:,splitIndex])##median splitting
        l_child = data[data[:,splitIndex]<=median]
        r_child = data[data[:,splitIndex]>median]
        h_left = impurityFunction(l_child[:,-1])
        h_right = impurityFunction(r_child[:,-1])
        p_left = l_child.shape[0]/(data.shape[0])
        p_right = 1 - p_left
        IG = IG - p_left * h_left - p_right * h_right
        #print("splitIndex:{} l_child_shape:{} r_child_shape:{} h_left:{} h_right{}:".format(splitIndex,l_child.shape[0],r_child.shape[0],h_left,h_right))
    else:
        attributes = col_types[col_name]
        for attribute in attributes:
            child = data[data[:,splitIndex]==attribute]
            h_child = impurityFunction(child[:,-1])
            p_child = child.shape[0]/data.shape[0]
            IG = IG - p_child * h_child
    return IG

def isContinuous(index):
    if index < 0 or index >= len(column_headers):
        return False
    
    return  col_types[column_headers[index]] == 'continuous'

class Node:
    def __init__(self,train_data,num_features,depth,purityFactors):
        #self.leaf = leaf
        self.nodeCount = 1
        self.train_data = train_data
        self.depth = depth
        self.num_features = num_features
        self.splitIndex = 1
        self.continuousSplit = None
        self.label = None
        self.children = []
        self.leaf= False
        self.height = 0
        self.purityFactors = purityFactors
        selectionFunction = purityFactors[0]
        if selectionFunction == 'InfoGain':
            self.selectionFunction  = InfoGain
        elif selectionFunction == 'GainRatio':
            self.selectionFunction = GainRatio
        else:
            self.selectionFunction = Entropy
        
        impurityFunction = purityFactors[1]
        if impurityFunction == 'Entropy':
            self.impurityFunction = Entropy
        elif impurityFunction == 'Gini':
            self.impurityFunction = Gini
        else:
            self.impurityFunction = Entropy
    
    def getSplitIndex(self,data):## returns -1 when the data is empty or no info gain is possible
        if len(data) == 0:
            return -1
        
        maxVal = 0
        maxIndex = -1
        for index in range(self.num_features):
            Val = self.selectionFunction(data,index,self.impurityFunction)
            if Val > maxVal:
                #print("hi")
                maxVal = Val
                maxIndex = index
            #print("index:{}, IG:{}".format(index,IG))
            
        
        return maxIndex
    
    def setSplitIndex(self):## sets splitIndex -1 when data is empty or no info gain possible
        index = self.getSplitIndex(self.train_data)
        if index>-1:
            if isContinuous(index):
                self.continuousSplit = np.median(self.train_data[:,index])
            self.splitIndex = index
        else:
            self.splitIndex = index
    
    def createChildren(self):##make sure tocall this only after split index has been set
        if self.splitIndex == -1:## no need to split
            self.numChildren = 0
            self.children = []
            self.leaf = True
            
        elif isContinuous(self.splitIndex):
            self.numChildren = 2
        else:
            self.numChildren = len(col_types[column_headers[self.splitIndex]])
    
        self.children = []
        continuous = isContinuous(self.splitIndex)
        for i in range(self.numChildren):
            if continuous:
                if i == 0:
                    data = self.train_data[self.train_data[:,self.splitIndex]<=self.continuousSplit]
                else:
                    data = self.train_data[self.train_data[:,self.splitIndex]>self.continuousSplit]
                
            else:
                #print(self.splitIndex,column_headers[self.splitIndex][i])
                data = self.train_data[self.train_data[:,self.splitIndex]==col_types[column_headers[self.splitIndex]][i]]
            
            self.children.append(Node(data,self.num_features,self.depth + 1,self.purityFactors))
    
    def setLeafLabel(self):
        
        if(len(self.train_data)==0):
            self.label = 0
            return
        unique, counts = np.unique(self.train_data[:,-1],return_counts = True)
        
        if(unique.size == 0):
            self.label = 0
            #print("Data empty : Default Label 0")
        else:
            self.label = unique[np.argmax(counts)]
            #print(unique,counts,"Setting label {}".format(self.label))
            
    
    def createTree(self,maxDepth):
        if self.depth >= maxDepth:
            self.leaf = True
            self.setLeafLabel()
            #print("Setting leaf label {}:".format(self.label))
            
        elif self.train_data.shape[0] == 0:
            self.leaf = True
            self.setLeafLabel()
            
        else:
            self.setSplitIndex()
            if self.splitIndex == -1:
                unique, counts = np.unique(train_data[:,-1],return_counts = True)## train_data is not of length zero guaranateed here
                self.leaf = True
                self.setLeafLabel()
                
            self.createChildren()## creates zero children if self.splitIndex== -1
            maxheight = -1
            for child in self.children:
                child.createTree(maxDepth)
                if child.height > maxheight:
                    maxheight = child.height
                
                self.nodeCount = self.nodeCount + child.nodeCount##to update the node count
                
            self.height = maxheight + 1
            
            
    def findChildForSample(self,x):
        if isContinuous(self.splitIndex):
            #print("{} {}\n".format(type(x[self.splitIndex]),type(self.continuousSplit)))
            if(x[self.splitIndex]<=self.continuousSplit):
                return 0
            else:
                return 1
        else:
            attribute_value = x[self.splitIndex]
            index = 0
            for i in range(len(col_types[column_headers[self.splitIndex]])):
                if attribute_value == col_types[column_headers[self.splitIndex]][i]:
                    index= i
            return index
    
    def predict(self,x):
        if self.leaf:
            #print("Reached Leaf {}".format(self.label))
            return self.label
        else:
            child= self.children[self.findChildForSample(x)]
            return child.predict(x)
    def predictMany(self,x_input):
        return np.array([ self.predict(x) for x in x_input])
    
    def createFullTree(self):
        #print(self.depth)
        self.setSplitIndex()
        if self.splitIndex == -1: ## the case when data is empty or no benefit upon splitting
            self.leaf = True
            self.setLeafLabel()
            self.height = 0
        else:
            self.createChildren()
            maxheight = -1
            for child in self.children:
                child.createFullTree()
                if child.height > maxheight:
                    maxheight = child.height
            
            self.height = maxheight + 1
    
    def getAccuracy(self,x,y):
        y_pred = self.predictMany(x)
        unique,counts = np.unique(y_pred == y,return_counts = True)
        if np.array_equal(unique,[True]):
            return 1
        elif np.array_equal(unique,[False]):
            return 0
        elif np.array_equal(unique,[True,False]):
            return counts[0]/(np.sum(counts))
        else:
            return counts[1]/(np.sum(counts))
    
    def wholeAccuracy(self,input_data):
        return self.getAccuracy(input_data[:,:-1],input_data[:,-1])
    
    
    def updateHeight(self):
        if self.leaf:
            self.height = 0
        else:
            maxheight = -1
            for child in self.children:
                child.updateHeight()
                if child.height > maxheight:
                    maxheight = child.height
            
            self.height = maxheight + 1
    
    def updateNodeCount(self):
        if self.leaf:
            self.nodeCount = 1
        else:
            self.nodeCount = 1
            for child in self.children:
                child.updateNodeCount()
                self.nodeCount = self.nodeCount + child.nodeCount
            
    
    def prune(self,prune_data):
        if self.leaf:
            return 
        if len(prune_data) == 0:
            return ## since you don't have pruning data don't prune this part
        
        
        continuous= isContinuous(self.splitIndex)#splitIndex can't be -1 over heres
        
        for i in range(len(self.children)):
            if continuous:
                if i == 0:
                    child_prune_data = prune_data[prune_data[:,self.splitIndex]<=self.continuousSplit]
                else:
                    child_prune_data = prune_data[prune_data[:,self.splitIndex]>self.continuousSplit]
                
            else:
                child_prune_data = prune_data[prune_data[:,self.splitIndex]==col_types[column_headers[self.splitIndex]][i]]
            
            child = self.children[i]
            child.prune(child_prune_data)    
            
        unmerged_acc = self.wholeAccuracy(prune_data)
        unique,counts = np.unique(self.train_data[:,-1],return_counts = True)##won't throw error since if train_data shape were zero it would have been categorized as leaf
        
        label_merged = unique[np.argmax(counts)]
        
        merged_correct = np.sum((prune_data[:,-1] == label_merged).astype(int))
        merged_acc = merged_correct/(len(prune_data)) ## WHAT IF PRUNE_DATA LENGTH IS ZERO
        
        if merged_acc > unmerged_acc:
            self.leaf = True
            self.label = label_merged
            self.children = []
            self.numChildren = 0
            
            root.updateNodeCount()
            
            pruneFile.write("{},{},{},{}\n".format(root.nodeCount,root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)))    
        
        return
            
        
        
test_labels= np.loadtxt("DT_data/test_labels.txt",dtype=object)
test_labels= np.array([ (int)(t) for t in test_labels],dtype=object)
test_data[:,-1] = test_labels
   

root = Node(train_data,train_data.shape[1]-1,0,('InfoGain','Gini'))

root.createFullTree()
print("Done creating full tree")
root.prune(val_data)
root.updateNodeCount()
root.updateHeight()
pruneFile.close()


"""
depths = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
f = open("data_customTree_InfoGain_Gini.csv",'w+')
f.write("Depth,Nodes,train_acc,val_acc,test_acc\n")
begin = time.time()
for depth in depths:
    
    root = Node(train_data,train_data.shape[1]-1,0,('InfoGain','Gini'))
    root.createTree(depth)
    print("MaxDepth:{},Cumulative Time elapsed:{}".format(depth,time.time()-begin))
    train_acc,val_acc,test_acc = root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)
    f.write("{},{},{},{},{}\n".format(root.nodeCount,root.height,train_acc,val_acc,test_acc))
    
    
f.close()





depths = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
f = open("data_customTree_GainRatio_Entropy.csv",'w+')
f.write("Depth,Nodes,train_acc,val_acc,test_acc\n")
begin = time.time()
for depth in depths:
    
    root = Node(train_data,train_data.shape[1]-1,0,('GainRatio','Entropy'))
    root.createTree(depth)
    print("MaxDepth:{},Cumulative Time elapsed:{}".format(depth,time.time()-begin))
    train_acc,val_acc,test_acc = root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)
    f.write("{},{},{},{},{}\n".format(root.nodeCount,root.height,train_acc,val_acc,test_acc))
    
    
f.close()




depths = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
f = open("data_customTree_GainRatio_Gini.csv",'w+')
f.write("Depth,Nodes,train_acc,val_acc,test_acc\n")
begin = time.time()
for depth in depths:
    
    root = Node(train_data,train_data.shape[1]-1,0,('GainRatio','Gini'))
    root.createTree(depth)
    print("MaxDepth:{},Cumulative Time elapsed:{}".format(depth,time.time()-begin))
    train_acc,val_acc,test_acc = root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)
    f.write("{},{},{},{},{}\n".format(root.nodeCount,root.height,train_acc,val_acc,test_acc))
    
    
f.close()
"""


"""
depths = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
f = open("data_customTree_InfoGain_Entropy.csv",'w+')
f.write("Depth,Nodes,train_acc,val_acc,test_acc\n")
begin = time.time()
for depth in depths:
    
    root = Node(train_data,train_data.shape[1]-1,0,('InfoGain','Entropy'))
    root.createTree(depth)
    print("MaxDepth:{},Cumulative Time elapsed:{}".format(depth,time.time()-begin))
    train_acc,val_acc,test_acc = root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)
    f.write("{},{},{},{},{}\n".format(root.nodeCount,root.height,train_acc,val_acc,test_acc))
    
    
f.close()


"""
"""
begin = time.time()
root = Node(train_data,train_data.shape[1]-1,0,('GainRatio','Entropy'))
root.createFullTree()

print("Before pruning ==> train acc: {} , val acc: {}, test acc: {}".format(root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)))
end = time.time()
print("Time taken for full tree growth: {} seconds".format(end - begin))

begin= time.time()
root.prune(val_data)
end = time.time()
print("After pruning  ==> train acc: {} , val acc: {}, test acc: {}".format(root.wholeAccuracy(train_data),root.wholeAccuracy(val_data),root.wholeAccuracy(test_data)))
print("Time taken for pruning: {} seconds".format(end - begin))
"""
