#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import glob
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
# filter warnings
warnings.filterwarnings('ignore')

import os
#print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.


# **Normally the data is at image form. We should convert image data to array so we will use cv2 library.  **

# In[ ]:


("../input/fruits-360_dataset/fruits-360/Training/Banana/*.JPG")

fruits=[]
files=glob.glob("../input/fruits-360_dataset/fruits-360/Training/Banana/*")
files2=glob.glob("../input/fruits-360_dataset/fruits-360/Training/Avocado/*")

for i in files:
    im=cv2.imread(i,0)
    fruits.append(im)
for i in files2:
    im2=cv2.imread(i,0)
    fruits.append(im2)
    
fruits2=np.asarray(fruits)
fruits2=fruits2/255


# After conversion, the pixels are in array from.

# In[ ]:


x=fruits2
zeros=np.zeros(427)
ones=np.ones(490)
y=np.concatenate((zeros,ones),axis=0).reshape(x.shape[0],1)
    


# **We have our datas as(x) and outputs(y),  y includes zeros(avocado) and ones(banana)                                             
# amount of bananas are 490, amount of avocados are 427**

# In[ ]:


print(x.shape)
print(y.shape)


# 917 is total amount of fruits (avocado and banana). 100x100 is our pixels.

# In[ ]:


plt.subplot(1,2,1)
plt.imshow(x[260])
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x[500])
plt.axis("off")
plt.show()
plt.subplot(1,2,1)
plt.imshow(x[1])
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x[900])
plt.axis("off")
plt.show()
plt.subplot(1,2,1)
plt.imshow(x[120])
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x[820])
plt.axis("off")
plt.show()


# 3 samples of both fruits

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


x_train_flatten=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test_flatten=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_train.shape[2])


# In[ ]:


print(x_train_flatten.shape)
print(x_test_flatten.shape)


# **We reduced our 3D matrix to 2D matrix in order to use our input labels, they should be 2D. Now they are in form of 100x100=10000 pixels.**

# In[ ]:


def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head


# We defined our sigmoid function to calculate output values.

# In[ ]:


x_train_flatten.shape


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import  Dense

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=8,kernel_initializer="uniform",activation="relu",
                         input_dim=x_train_flatten.shape[1]))
    classifier.add(Dense(units=4,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,epochs=10)
accuracies=cross_val_score(estimator=classifier,X=x_train_flatten,y=y_train,cv=3)
mean=accuracies.mean()
variance=accuracies.std()

print("accuracy mean: ",str(mean))
print("accuracy variance: ",str(variance))


# We have a ccuracy 0.9331 and variance of 0.0037. They both are really good numbers.

# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
classifier.fit(x_train_flatten,y_train)
y_pred=classifier.predict(x_train_flatten)
conf_mat=confusion_matrix(y_train,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.show()


# We predicted zeros(avocados) 100% but we should improve our system for bananas.

# **In conclusion, at the beginning we have only images. We wanted to make an image classification to seperate avocados and bananas.**
# 1. We converted all images to array with their pixels.
# 1. We marked bananas with number 1 and avocados with number 0.
# 1. We used keras library to classify both fruits.
# 1. We found accuracy and variance values.

# In[ ]:




