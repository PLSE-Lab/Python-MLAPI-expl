#!/usr/bin/env python
# coding: utf-8

# # Image Classification with MNIST Dataset

# In[17]:


import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np


# # Load Data

# In[18]:


dfTrain = pd.read_csv('../input/train.csv')   #Training Dataset
df_Features=dfTrain.iloc[:,1:785]
df_Label=dfTrain.iloc[:,0:1]
x_train=df_Features.values
y_train=df_Label.values
x_train=x_train.reshape(x_train.shape[0], 28,28)

print("Training Data",x_train.shape,y_train.shape)
dfTest = pd.read_csv('../input/test.csv')   #Training Dataset
df_Features=dfTest.iloc[:,0:784]
df_Label=dfTest.iloc[:,0:1]*-1
x_test=df_Features.values
y_test=df_Label.values
x_test=x_test.reshape(x_test.shape[0], 28,28)
print("Test Data",x_test.shape,y_test.shape)



# # Preview Sample

# In[19]:


image_index = 7777 
print("Actual Label=",y_train[image_index]) 
plt.axis('off')
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()


# # Split Distribution

# In[20]:


plt.figure(figsize=(12,4))
plt.subplot('121')
Y_histogram=keras.utils.to_categorical(y_train, 10)   
Y_histogram=Y_histogram.sum(axis=0)
plt.bar(range(10),Y_histogram)
plt.xticks(range(10),rotation=90)
plt.xlabel(Y_histogram)
plt.title("Training Data")


plt.subplot('122')
Y_histogram=keras.utils.to_categorical(y_test, 10)   
Y_histogram=Y_histogram.sum(axis=0)
plt.bar(range(10),Y_histogram)
plt.xticks(range(10),rotation=90)
plt.xlabel(Y_histogram)
plt.title("Test Data")
plt.show()


# # Input Transformations

# In[21]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# # Model

# In[22]:


model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# # Train

# In[23]:


model.fit(x=x_train,y=y_train, epochs=10)


# # Evaluate 

# In[24]:


model.evaluate(x_train, y_train)


# # Predict

# In[25]:


image_index = 4444
plt.axis('off')
plt.imshow(x_train[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_train[image_index].reshape(1, 28, 28, 1))

print("Predicted Value=",pred.argmax())


# # Create Confusion Matrix

# In[26]:


def getConfusionMatrix(Y_True,Y_Pred):
    classCount=Y_True.shape[1]
    #in case Y is in term of probablities
    Y_Pred=keras.utils.to_categorical(Y_Pred.argmax(axis = 1), classCount) 
    cnfMtrx=np.zeros((classCount,classCount))
    for i in range(classCount):
        cnfMtrx[i,:]=Y_Pred[np.where(Y_True[:,i]==1)].sum(axis=0)
    return cnfMtrx


# In[27]:


Y_pred = model.predict(x_train.reshape(x_train.shape[0], 28, 28, 1))
Y_true = keras.utils.to_categorical(y_train,10)
confusion_mtx = getConfusionMatrix(Y_true, Y_pred)


# # Plot Confusion Matrix

# In[28]:


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Confusion matrix')
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=90)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in (range(cm.shape[1])):
            if cm[i, j] > cm.max() / 2:
                txtclr="white"
            else:
                txtclr="black"
            plt.text(j, i, cm[i, j],horizontalalignment="center",color=txtclr)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
       

    plt.colorbar()
    plt.show()


# In[29]:


plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# In[34]:


Y_pred = model.predict(x_test.reshape(x_test.shape[0], 28, 28, 1))
Y_New=Y_pred.argmax(axis=1)


# In[35]:


dfs = pd.read_csv('../input/sample_submission.csv')   #Training Dataset
dfs['Label']=np.round(Y_New,0).astype(int)
dfs.head()
#dfs.to_csv('../input/manmohan291_minst_submission.csv',index=False)

