import numpy as np
import csv
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

trainfile = "DT_data/train.csv"
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

column_headers = headings



validationfile = 'DT_data/valid.csv'
val_data= []
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
                val_data.append(row)
val_data = np.array(val_data,dtype='object')




def Entropy(y):
    unique,counts = np.unique(y,return_counts=True)
    h = 0
    for i in range(unique.size):
        p = counts[i]/np.sum(counts)
        if p > 0:
            h = h - p*np.log(p)
    return h

def InfoGain(data,splitIndex):
    IG = Entropy(data[:,-1])
    col_name = column_headers[splitIndex]
    if col_types[col_name] == 'continuous':
        
        median = np.median(data[:,splitIndex])##median splitting
        l_child = data[data[:,splitIndex]<=median]
        r_child = data[data[:,splitIndex]>median]
        h_left = Entropy(l_child[:,-1])
        h_right = Entropy(r_child[:,-1])
        p_left = l_child.shape[0]/(data.shape[0])
        p_right = 1 - p_left
        IG = IG - p_left * h_left - p_right * h_right
        #print("splitIndex:{} l_child_shape:{} r_child_shape:{} h_left:{} h_right{}:".format(splitIndex,l_child.shape[0],r_child.shape[0],h_left,h_right))
    else:
        attributes = col_types[col_name]
        for attribute in attributes:
            child = data[data[:,splitIndex]==attribute]
            h_child = Entropy(child[:,-1])
            p_child = child.shape[0]/data.shape[0]
            IG = IG - p_child * h_child
    return IG

def isContinuous(index):
    return col_types[column_headers[index]] == 'continuous'

class Node:
    def __init__(self,train_data,num_features,depth):
        #self.leaf = leaf
        self.train_data = train_data
        self.depth = depth
        self.num_features = num_features
        self.splitIndex = 1
        self.continuousSplit = None
        self.label = None
        self.leaf= False
    
    
    def getSplitIndex(self,data):
        
        maxIG = InfoGain(data,0)
        maxIndex = 0
        for index in range(self.num_features):
            IG = InfoGain(data,index)
            if IG > maxIG:
                #print("hi")
                maxIG = IG
                maxIndex = index
            print("index:{}, IG:{}".format(index,IG))
            
        #print("depth:{} maxIndex:{}".format(self.depth,maxIndex))
        return maxIndex
    
    def setSplitIndex(self):
        index = self.getSplitIndex(self.train_data)
        if isContinuous(index):
            self.continuousSplit = np.median(self.train_data[:,index])
        self.splitIndex = index
    
    def createChildren(self):##make sure tocall this only after split index has been set
        if isContinuous(self.splitIndex):
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
            
            self.children.append(Node(data,self.num_features,self.depth + 1))
    
    def setLeafLabel(self):
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
            self.label = 0
        else:
            self.setSplitIndex()
            self.createChildren()
            for child in self.children:
                child.createTree(maxDepth)
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
                if attribute_value == col_types[column_headers[self.splitIndex][i]]:
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
        return self.predict(x_input)

root = Node(train_data,train_data.shape[1]-1,0)
root.createTree(1)

x = val_data[0][:-1]