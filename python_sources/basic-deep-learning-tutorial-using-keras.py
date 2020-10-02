#!/usr/bin/env python
# coding: utf-8

# Hi, this is a basic Keras tutorial. In this notebook you will learn to build a basic neural network.
# We will go through these steps in our notebook:
# 1. Import Libraries
# 2. Load dataset
# 3. Build Model
# 4. Train Model
# 5. Find Accuracy on Validation Set
# 6. Predict Values

# In[1]:


#importing Libraries :)

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Activation,Layer,Lambda

from sklearn.cross_validation import train_test_split


# In[2]:


#load dataset

dataset=pd.read_csv("../input/data.csv")


# In[3]:


dataset.head()


# In[4]:


#No idea why Unnamed:32 column exists. And we also don't need id column.
dataset=dataset.drop(["id","Unnamed: 32"],axis=1)
dataset.head()


# In[5]:


dataset.shape


# In this tutorial I'm not focusing on feature engineering. I'm just focusing on implementation of a simple neural network. So I'll use the dataset as it is.

# In[6]:


#just checking if there is any null value..
pd.isnull(dataset).sum()


# I don't think  so. We are good to go :D
# Before proceeding, first map diagnosis to integer value.

# In[7]:


#mapping function to map different string objects to integer
def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data


# In[8]:


dataset=mapping(dataset,feature="diagnosis")


# In[9]:


dataset.sample(5)


# Malignant is mapped to 0, Benign is mapped to 1

# In[10]:


#divide dataset into x(input) and y(output)
X=dataset.drop(["diagnosis"],axis=1)
y=dataset["diagnosis"]


# In[11]:


#divide dataset into training set, cross validation set, and test set
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)


# Now that we have our dataset, we will move to our 3rd step that is building model.

# In[12]:


#hein?? what is all this??
def getModel(arr):
    model=Sequential()
    for i in range(len(arr)):
        if i!=0 and i!=len(arr)-1:
            if i==1:
                model.add(Dense(arr[i],input_dim=arr[0],kernel_initializer='normal', activation='relu'))
            else:
                model.add(Dense(arr[i],activation='relu'))
    model.add(Dense(arr[-1],kernel_initializer='normal',activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])
    return model


# Above I've defined a function that will return us a neural network model.<br>
# We will pass an array of integer to define the no. of hidden units in each layer. First layer will have same no. of units as input dimension. Each subsequent layer will have the units set in the array  passed.<br>
# model=Sequential() will give us a model. <br>
# model.add() is used to add a layer to the  model. <br>
# We will set activation function for each layer. Since we need binary classification, we'll use sigmoid activation in the output layer.<br>
# At the end we will compile the build model, with loss function, optimizer and the metrics we want when we will evaluate the model.

# Now we will define different models so that we can test each of them on validation set and check the accuracy.
# 1. Firstly, we'll use a small model which contains 3 layers with hidden units 30, 50 and 1.
# 2. Then we'll use a wider network which will also have 3 layers but more hidden units in the hidden layer which is 30,100 and 1
# 3. Then we'll use a deeper network which will have 5 layers.
# 
# <br>
# We are just checking different structures to get different results.

# In[13]:


firstModel=getModel([30,50,1])


# Now we will create a callback function which will plot loss on each epoch end. we will override on_epch_end() method to plot the graph.

# In[14]:


import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()


# In[15]:


firstModel.fit(np.array(trainX),np.array(trainY),epochs=40,callbacks=[plot_losses])


# In[16]:


scores=firstModel.evaluate(np.array(valX),np.array(valY))


# 'scores' will contain two values, one is loss, which is default and the other is accuracy which we passed as an arguement as 'metrics' when we compiled the model

# In[17]:


print("Loss:",scores[0])
print("Accuracy",scores[1]*100)


# In[18]:


secondModel=getModel([30,100,1])
secondModel.fit(np.array(trainX),np.array(trainY),epochs=40,callbacks=[plot_losses])


# In[19]:


scores2=secondModel.evaluate(np.array(valX),np.array(valY))


# In[20]:


print(scores2)


# In[21]:


thirdModel=getModel([30,50,70,40,1])


# In[22]:


thirdModel.fit(np.array(trainX),np.array(trainY),epochs=100,callbacks=[plot_losses])


# In[23]:


scores3=thirdModel.evaluate(np.array(valX),np.array(valY))


# In[24]:


print(scores3)


# Now we will move toward our final step that is prediction of values.

# In[25]:


predY=firstModel.predict(np.array(testX))
predY=np.round(predY).astype(int).reshape(1,-1)[0]


# In[26]:


from sklearn.metrics import confusion_matrix
m=confusion_matrix(predY,testY)
tn, fn, fp, tp=confusion_matrix(predY,testY).ravel()
m=pd.crosstab(predY,testY)
print("Confusion matrix")
print(m)


# In[27]:


sens=tp/(tp+fn)
spec=tn/(tn+fp)
print("Senstivity:",sens)
print("Specificity:",spec)


# So, this is all for basic neural network using Keras. If you have any queries please post it in the comment section. If you think I've made any mistake, please comment it below. If you like this tutorial, please give it an upvote. :D
# <br><br>
# Take care. :)

# In[ ]:




