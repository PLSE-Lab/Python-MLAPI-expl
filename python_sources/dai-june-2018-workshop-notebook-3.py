#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1> Danbury AI June 2018: Workshop Part 3</h1>
# <h2>MNIST</h2>
# </center>
# We begin our journey into using neural networks for modeling image data by starting with the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/). This academic dataset consists of well pre-processed images of digits 0 through 9. Due to the preprocessing, this dataset is simple enough to model with simple neural networks, but still maintains sufficient complexity to be a non-novel dataset for evaluating the effectivenss of neural networks versus other methods. In our next notebook we will be using a more complicated natural image dataset which will require more sophisticated neural architectures to model appropriately. 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from ipywidgets import interact
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input/digit-recognizer"))


# Let's read in our data and display the first few rows. 

# In[2]:


mnistTrainingData = pd.read_csv("../input/digit-recognizer/train.csv")
mnistTrainingData.head()


# Since the labels and features are in the same matrix, we will need to split the matrix into a feature matrix *x* and a label vector *y*.

# In[3]:


X = mnistTrainingData.values[:,1:]
y = mnistTrainingData.values[:,0]


# We can use the interactive ipython components to visualize the digits. The label of the digit is displayed above the images. 

# In[15]:


def disp(imSelIdx=0):
    plt.title(y[imSelIdx])
    plt.imshow(X[imSelIdx].reshape(28,28), cmap="gray")

interact(disp,imSelIdx=(0,X.shape[0]))


# We will now convert the integer labels contained in *y* to [one-hot vectors](https://www.youtube.com/watch?v=2Uyr93f3C2M). 

# In[5]:


def oneHotEncoder(integerVal,maxClasses):
    out = np.zeros(maxClasses)
    out[integerVal] = 1
    return out

y_onehot = []

for i in y:
    y_onehot.append(oneHotEncoder(i,y.max()+1))

y_onehot = np.stack(y_onehot)

print(y_onehot)
print("Shape of y vector: {0}".format(y.shape))
print("Shape of y one-hot matrix: {0}".format(y_onehot.shape))


# Now we will split our data 80/20 into our *training* and *validation* sets. 

# In[12]:


X_train, X_validation, y_train , y_validation = train_test_split(X,y_onehot, test_size=0.2)


# At this point we are ready to train our first simple *feed-forward neural network*. 

# In[16]:


inputs = Input(shape=(X.shape[1],))

x = Dense(100, activation='sigmoid')(inputs)
x = Dense(y_onehot.shape[1], activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


# In[18]:


hist = model.fit(X_train,y_train,epochs=50, batch_size=100, validation_data=(X_validation,y_validation)) 


# Here we will define a function which allows us to plot the training history of our neural network. 

# In[26]:


def learningCurves(hist):
    histAcc_train = hist.history['acc']
    histLoss_train = hist.history['loss']
    histAcc_validation = hist.history['val_acc']
    histLoss_validation = hist.history['val_loss']
    maxValAcc = np.max(histAcc_validation)
    minValLoss = np.min(histLoss_validation)

    plt.figure(figsize=(12,12))
    epochs = len(histAcc_train)

    plt.plot(range(epochs),histLoss_train, label="Training Loss", color="#acc6ef")
    plt.plot(range(epochs),histLoss_validation, label="Validation Loss", color="#a7e295")

    plt.scatter(np.argmin(histLoss_validation),minValLoss,zorder=10,color="green")

    plt.xlabel('Epochs',fontsize=14)
    plt.title("Learning Curves",fontsize=20)

    plt.legend()
    plt.show()

    print("Max validation accuracy: {0}".format(maxValAcc))
    print("Minimum validation loss: {0}".format(minValLoss))


# We will now use the function defined above to visualize the learning curves/training curves of our neural network. 

# In[27]:


learningCurves(hist)


# We will now add a dropout layer between the hidden layer and output layer. 

# In[33]:


inputs = Input(shape=(X.shape[1],))

x = Dense(100, activation='sigmoid')(inputs)
x = Dropout(0.5)(x)
x = Dense(y_onehot.shape[1], activation='softmax')(x)

model3 = Model(inputs=inputs, outputs=x)
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model3.summary()


# In[34]:


hist3 = model3.fit(X_train,y_train,epochs=50, batch_size=100, validation_data=(X_validation,y_validation)) 


# In[35]:


learningCurves(hist3)


# **Workshop Problems**
# * In part 2 we saw how to use sklearn models like linear regression, random forests, and boosted trees. How do these models compare to the effectiveness of neural networks on this problem? Apply these models to MNIST here. 
# * Modify the networks above by adding layers, adjusting layer widths, and changing how much dropout is used. How do these changes impact the learning curves? 
