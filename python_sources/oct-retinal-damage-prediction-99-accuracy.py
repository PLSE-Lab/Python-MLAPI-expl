#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle


# In[ ]:


image_size=(128,128,3)
path1 ="/kaggle/input/kermany2018/OCT2017 /train"
path2 ="/kaggle/input/kermany2018/OCT2017 /test"
path3 ="/kaggle/input/kermany2018/OCT2017 /val"
epochs = 10


# In[ ]:


myList = os.listdir(path1)
print("Total Number of Classes Detected :",len(myList))
print(myList)


# In[ ]:


noOfclasses= len(myList)


# In[ ]:


print(myList)


# In[ ]:


print("Importing Classes...")


# In[ ]:


x_train=[]
y_train=[]
CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]
count=0
for x in myList:
  myPicList = os.listdir(path1+"/"+str(x))
  for y in myPicList:
      curImg = cv2.imread(path1+"/"+str(x)+"/"+y)
      curImg = cv2.resize(curImg,(image_size[0],image_size[1]))
      x_train.append(curImg)
      y_train.append(CATEGORIES.index(x))
  print(x,end=" ")


# In[ ]:


x_test=[]
y_test=[]
CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]
for x in myList:
  myPicList = os.listdir(path2+"/"+str(x))
  for y in myPicList:
    curImg = cv2.imread(path2+"/"+str(x)+"/"+y)
    curImg = cv2.resize(curImg,(image_size[0],image_size[1]))
    x_test.append(curImg)
    y_test.append(CATEGORIES.index(x))
  print(x,end=" ")


# In[ ]:


x_val=[]
y_val=[]
CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]
for x in myList:
  myPicList = os.listdir(path3+"/"+str(x))
  for y in myPicList:
    curImg = cv2.imread(path3+"/"+str(x)+"/"+y)
    curImg = cv2.resize(curImg,(image_size[0],image_size[1]))
    x_val.append(curImg)
    y_val.append(CATEGORIES.index(x))
  print(x,end=" ")


# In[ ]:


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


# In[ ]:


numofSamples=[]
for x in range(0,noOfclasses):
  numofSamples.append(len(np.where(y_train==x)[0]))


# In[ ]:


print(numofSamples)


# In[ ]:


plt.figure(figsize=(10,5))
plt.bar(range(0,noOfclasses),numofSamples)
plt.title('No of Images for each Class')
plt.xlabel("Class ID")
plt.ylabel("No of Images")
plt.show()


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns

X_trainShape = x_train.shape[1]*x_train.shape[2]*x_train.shape[3] #49k

X_trainFlat = x_train.reshape(x_train.shape[0], X_trainShape)
Y_train = y_train


#ros = RandomUnderSampler()
ros = RandomUnderSampler(sampling_strategy='auto',random_state=1)
X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)


# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])[0010]
Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 4)
# Make Data 2D again
for i in range(len(X_trainRos)):
    height, width, channels = image_size[0],image_size[1],3
    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
# Plot Label Distribution
dfRos = pd.DataFrame()
dfRos["labels"]=Y_trainRos
labRos = dfRos['labels']
sns.countplot(labRos)


# In[ ]:


x_train = X_trainRosReshaped
print(x_train[0].shape)


# In[ ]:


print(x_test[0].shape)


# In[ ]:


#print(x_validation[0].shape)
x_validation = x_val
print(x_validation[0].shape)


# In[ ]:


print(x_train.shape)


# In[ ]:


X_train=x_train
print(X_train.shape)
X_test=x_test
X_validation = x_validation


# In[ ]:


y_train = to_categorical(y_train,noOfclasses)
y_test = to_categorical(y_test,noOfclasses)
y_validation = to_categorical(y_val,noOfclasses)


# In[ ]:


print(X_train.shape)
print(Y_trainRosHot.shape)
print(X_validation.shape)
print( y_validation.shape)


# In[ ]:


class CustomCallback(tf.keras.callbacks.Callback):
  def __init__(self,fraction):
    super(CustomCallback,self).__init__()
    self.fraction = fraction
  def on_epoch_begin(self,epoch,logs=None):
    lr= tf.keras.backend.get_value(self.model.optimizer.lr)
    print("Learning Rate: "+ str(lr))
  def on_epoch_end(self,epoch,logs=None):
    lr= tf.keras.backend.get_value(self.model.optimizer.lr)
    lr *= self.fraction
    tf.keras.backend.set_value(self.model.optimizer.lr,lr)


# In[ ]:


def train(model, name,epochs=epochs):
  history = model.fit(X_train,Y_trainRosHot,epochs=epochs,validation_data =(X_validation,y_validation) ,batch_size=64,
                      shuffle=True,
                      max_queue_size=20,
                      use_multiprocessing=True,
                      workers=1,
                    callbacks=[CustomCallback(fraction=0.9)])
  np.save(str(name)+'_model_trained_'+str(epochs)+'epochs.npy',history)
  score = model.evaluate(X_test,y_test,verbose=0)
  print('Test Score = ',score[0])
  print('Test Accuracy = ',score[1])
  model.save(str(name)+'_model_trained_'+str(epochs)+'epochs.model')
  plt.figure(1)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.legend(['training','validation'])
  plt.title('Loss')
  plt.xlabel('epoch')
  plt.figure(2)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.legend(['training','validation'])
  plt.title('Accuracy')
  plt.xlabel('epoch')
  plt.show()


# In[ ]:



from keras.applications import ResNet50
model = Sequential()
model.add(ResNet50(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))
model.add(Flatten())
model.add(Dense(noOfclasses,activation="softmax"))
model.summary()
model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])
train(model,'ResNet50',10)


# In[ ]:



from keras.applications import ResNet50
model = Sequential()
model.add(ResNet50(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))
model.add(Flatten())
model.add(Dense(noOfclasses,activation="softmax"))
model.summary()
model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])
train(model,'ResNet50',25)


# In[ ]:



from keras.applications import ResNet50
model = Sequential()
model.add(ResNet50(include_top=False, input_shape=(image_size[0],image_size[1],image_size[2])))
model.add(Flatten())
model.add(Dense(noOfclasses,activation="softmax"))
model.summary()
model.compile(Adam(lr=0.0001),loss="categorical_crossentropy",metrics=['accuracy'])
train(model,'ResNet50',50)

