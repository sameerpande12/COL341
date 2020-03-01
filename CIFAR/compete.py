import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.utils import to_categorical,Sequence
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D,ZeroPadding2D,BatchNormalization,Input,GlobalAveragePooling2D
from keras.callbacks import CSVLogger
from keras.optimizers import SGD
import sys
import keras.optimizers
from keras import regularizers

trainfile = sys.argv[1]
testfile = sys.argv[2]
outputfile = sys.argv[3]

# Any results you write to the current directory are saved as output.
x_train= (pd.read_csv(trainfile,header=None,delim_whitespace=True)).to_numpy()
y_train = x_train[:,-1]
y_train = to_categorical(y_train,num_classes = 100)
y_train_sup = x_train[:,-2]
x_train  = x_train[:,:-2]

x_train = np.array([x.reshape(3,32,32) for x in x_train])
x_train = np.array([np.transpose(x,(1,2,0)) for x in x_train])

x_test = (pd.read_csv(testfile,header=None,delim_whitespace=True)).to_numpy()
x_test = x_test[:,:-2]
x_test = np.array([x.reshape(3,32,32) for x in x_test])
x_test = np.array([np.transpose(x,(1,2,0)) for x in x_test])


model = Sequential()
model.add(Conv2D(64,kernel_size=3,input_shape=(32,32,3),kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(64,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(128,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(256,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(256,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(256,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(512,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(512,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512,kernel_size=3,kernel_regularizer=regularizers.l2(0.00025),padding='same',activation='relu'))
model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(512,kernel_regularizer=regularizers.l2(0.00025),activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(100,activation='softmax'))

model.compile(optimizer=SGD(lr=0.008,decay=1e-6,momentum=0.9,nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=100,epochs=300,verbose=1,validation_split=0.1)

#hist_df = pd.DataFrame(history.history)
#hist_csv_file = 'history.csv'
#with open(hist_csv_file, mode='w') as f:
#    hist_df.to_csv(f)

#model.save("mymodel.h5")
#model.load_weights("/kaggle/input/mymodel-vgg/mymodel.h5")
### the 3072 indexed column -> superclass , 3073 -> class


y_test = model.predict(x_test)
y_test = np.argmax(y_test,axis=1)
f = open(outputfile,'w+')
for y in y_test:
    f.write("{}\n".format(y))
f.close()