import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
def normInput(x):return (x-meanPx)/stdPx
pth='../input/train.csv'
trainX=pd.read_csv(pth)
pth='../input/test.csv'
test=pd.read_csv(pth)
test=test.as_matrix()
trainY=trainX['label']
trainY=to_categorical(trainY,10)
#trainX.drop(columns=['label'],inplace=True)
trainX.drop(['label'],axis=1,inplace=True)

trainX=trainX.as_matrix()
meanPx=trainX.mean().astype(np.float32)
stdPx=trainX.std().astype(np.float32)
trainX=trainX.reshape(-1,28,28,1)
test=test.reshape(-1,28,28,1)

trainX,valX,trainY,valY=train_test_split(trainX,trainY,test_size=.1)

model = Sequential()
#model.add(Dense(10, input_dim=784,activation='softmax'))

model.add(Lambda(normInput,input_shape=(28,28,1)))
model.add(Conv2D(32,3,activation='relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=1))

model.add(Conv2D(64,3,activation='relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=1))

model.add(Conv2D(64,3,activation='relu'))
#model.add(MaxPooling2D())
model.add(BatchNormalization(axis=1))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=8,batch_size=64,validation_data=(valX,valY))

idg=ImageDataGenerator(rotation_range=20,shear_range=.2,zoom_range=.2)
model.fit_generator(idg.flow(trainX,trainY),steps_per_epoch=500,validation_data=(valX,valY))

#for i in range(10):
preds=model.predict(test)
predsV=model.predict(valX)
trainXp=np.concatenate((trainX,test,valX))
trainYp=np.concatenate((trainY,preds,predsV))
model.fit(trainXp,trainYp,epochs=1,batch_size=64,validation_data=(valX,valY))

preds=model.predict(test)
preds=np.argmax(preds,axis=1).tolist()
idx=[i for i in range(1,len(preds)+1)]
preds=pd.DataFrame(data={'ImageId':idx,'Label':preds})
#preds['ImageId']=preds.index+1
preds.to_csv('sub.csv',index=False)
