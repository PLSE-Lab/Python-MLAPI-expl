#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visulization 
from sklearn import svm  # Support vector machine
from sklearn.model_selection import train_test_split  # split the train data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#loading data files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape


# In[ ]:


#exploring data files
train_image = train.iloc[0:5000,1:]
train_label = train.iloc[0:5000,:1]
train.head()


# In[ ]:


#Since the image is currently one-dimension, we load it into a numpy array and reshape it so that it is two-dimensional (28x28 pixels)
i=0
train_img = train_image.iloc[i].as_matrix()
train_img = train_img.reshape((28,28))
plt.imshow(train_img, cmap = 'gray')
train_label.iloc[i]
train_img1.shape


# In[ ]:


#Split the training data into train & validation set
train_images, val_images , train_labels, val_labels = train_test_split(train_image, train_label,train_size = 0.8, random_state =0)


# In[ ]:


#trying SVM 
clf = svm.SVC()


# In[ ]:


#fitting the modle
clf.fit(train_images,train_labels.values.ravel())


# In[ ]:


#checking the accuracy for validation set
clf.score(val_images, val_labels)


# **Pre-Porcessing **
# 
# **Feature Standardization/regularization  ( It is used to centre the data around zero mean and unit variance)
# #1st: converting all the value grater than zero to one and rest all remain zero.
# #2nd: converting the value with standarization or divide the value by 255 as pixle value is varying from (0-255) to achive the regularization**[](http://)

# In[ ]:


train_image/=255
test_SVM=test/255
train_images/=255
val_images/=255


# In[ ]:


clf.fit(train_images,train_labels.values.ravel())


# In[ ]:


clf.score(val_images,val_labels)


# In[ ]:


results_SVM=clf.predict(test_SVM[0:])


# In[ ]:


#convert the result into desierd file(.csv)
df = pd.DataFrame(results_SVM)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('../results_SVM.csv', header=True)
df.head()


# In[ ]:


#Exploring orignal data files and converting them itno array
x_train = np.array(train.iloc[:,1:])
y_train = np.array(train.iloc[:,:1])
x_test = np.array(test)


# In[ ]:


n_features_train = x_train.shape[1]        #[1]represent col
n_samples_train = x_train.shape[0]        #[0]represent row
n_features_test = x_test.shape[1]
n_samples_test = x_test.shape[0]
print(n_features_train, n_samples_train, n_features_test, n_samples_test)
print(x_train.shape, y_train.shape, x_test.shape)


# In[ ]:


# show the image
def show_img(X):
    plt.figure(figsize=(8,7))
    n_samples = X.shape[0]
    X = X.reshape(n_samples, 28, 28)
    for i in range(20):
        plt.subplot(5, 4, i+1)
        plt.imshow(X[i])
    plt.show()


# In[ ]:


show_img(x_train)


# In[ ]:


show_img(x_test)


# In[ ]:




