#!/usr/bin/env python
# coding: utf-8

# **Lenet-5 Architecture**

# <img src="https://cdn-images-1.medium.com/max/2400/1*1TI1aGBZ4dybR6__DI9dzA.png" width="2000px">
# 
# 

# Hey Guys! Welcome back.
# 
# In this kernel we are going to classify hand-written digits based on **MNIST** data 

# 1> 
# Importing necessary libraries
# 
# 2>
# Data Preprocessing and visualisation
# 
# 3>
# Building the model
# 
# 4>
# Prediction,evauation and Performance Analysis.
# 
# 5> 
# Competition Submission

# 1.  **First we are importing the necessary libraries.**
# 
# In this Kernel we have used primarily Keras with TensorFlow backend for our Convulutional Model.

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Activation,Dropout,Dense,Conv2D,AveragePooling2D,Flatten,ZeroPadding2D,MaxPooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import seaborn as sns
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical 
import math


# **2. Next step is loading the data and do pre-processing and visualising it.** 
# 
# Now we visualise the data given to us by plotting the frequency of digits inthe given dataset.

# In[8]:


test=pd.read_csv('../input/test.csv',delimiter=',')
train=pd.read_csv('../input/train.csv',delimiter=',')
print(train.head())
label=train['label']
print(label.shape)
del train['label']
print(label.head())
sns.countplot(label)
train=train.values
train=train.reshape(train.shape[0],28,28,1)
test=test.values
test=test.reshape(test.shape[0],28,28,1)


# Then we show some sample images present in the dataset

# In[9]:


plt.figure(figsize=(25,10))
for i in range(0,10):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train[i][:,:,0],cmap='gray')
    #these are the sample images  


# In[10]:


label=to_categorical(label,10)


# We divide the data for testing and training in **90-10** since it is a competition submission.However it is generally splitted in the ratio 
# of **70-30**.

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(train,label, test_size = 0.1)


# **3. Building the model**
# 
# This is a standard LeNet-5 Implementation which was proposed for digit recognition by **Yaan Lecunn**.
# [Here is the link for the official research paper](http://yann.lecun.com/exdb/lenet/).

# In[15]:


model=Sequential()
def build():
    model.add(Conv2D(6,(5,5),strides=1,padding='valid', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(16,(5,5),strides=1,padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='AdaDelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(X_train,Y_train,epochs=30,batch_size=128,verbose=1)
    
build()


# **4. Prediction/Evaluation phase**
# 
# 

# In[16]:


modelsub=model.predict(X_train,batch_size=None, verbose=1)
c=0
f=0
for i in range(modelsub.shape[0]):
    if np.argmax(modelsub[i])==np.argmax(Y_train[i]):
        c+=1
    else:
        f+=1
accuracy=c/(c+f)
print('Train Accuracy:',accuracy*100)


# In[17]:


a=model.evaluate(X_test, Y_test,verbose=1)
print('Test Accuracy',a[1])


# In[18]:


mode=model.predict(test,batch_size=None, verbose=1)


# Printing the predicted frequencies of train data.

# In[19]:


ls=[0,0,0,0,0,0,0,0,0,0]
for i in range(modelsub.shape[0]):
    ls[np.argmax(modelsub[i])]+=1
print(ls)


# In[20]:


count1=0
count2=0
for i in range(modelsub.shape[0]):
    if(np.argmax(Y_train[i])==np.argmax(modelsub[i])):
        count1+=1
    else:
        count2+=1
print(count1,' ',count2)


# Printing some samples which were classified wrong.

# In[21]:


count=0
plt.figure(figsize=(25,10))
while count<10:
    for i in range(modelsub.shape[0]):
        if(np.argmax(Y_train[i])!=np.argmax(modelsub[i])):
            plt.subplot(1,10,count+1)
            plt.xticks([])
            plt.yticks([])
            s='Predicted:'
            s=s+' '+str(np.argmax(modelsub[i]))
            plt.xlabel(s)
            plt.imshow(train[i][:,:,0],cmap='gray')
            count+=1
            if count is 10:
                break
            


# **5. Competition Submission**

# In[22]:


ccc=[]
for i in range(mode.shape[0]):
    ccc.append(np.argmax(mode[i]))
d=np.arange(0,test.shape[0])+1
d.shape
df=pd.DataFrame({'ImageId':d,'Label':ccc
},index=d)
df.to_csv("cnn_mnist_datagen.csv",index=False)


# <img src="https://www.thebalancecareers.com/thmb/Df3jp07jm7AK30eNji0G6Fkl93s=/2122x1415/filters:fill(auto,1)/185002046-56b0974c3df78cf772cfe3c5.jpg" width="2000px">
