#!/usr/bin/env python
# coding: utf-8

# # Content
# * Introduction
# * Import library
# * Load data
# * Sample data representation
# * Data split train and test
# * Converting the 3D matrix to 2D matrix
# * Logistic regression create model and test accuracy score print
# * Conclusion

# # Introduction
# * This is my second logistic regression work
# * This time I worked with data from the images data.
# * This data is composed of sign language images.

# # Import library
# * We import libraries that we will use.
# * matplotlib.pyplot
# * train_test_split in sklearn.model_selection 
# * LogisticRegression in sklearn.linear_model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load data
# * In this data there are 2062 sign language digits images.
# * At the beginning of tutorial we will use only sign 0 and 1 for simplicity. 
# * In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
# * Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).
# * Lets prepare our X and Y arrays. X is image array (zero and one signs) and Y is label array (0 and 1).

# In[ ]:


# load data
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64
#print(x_l)
#print(Y_l)


# # Sample data representation
# * We are plotting our sample data.

# In[ ]:


# sample data representation
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()


# * In order to create image array, we concatenate zero sign and one sign arrays
# * Then we create label array 0 for zero sign images and 1 for one sign images.

# In[ ]:


X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)


# # Data split train and test
# * With the train_test_split method we import from sklearn library, we divide the data into two as train and test.
# * We will train our model with the train part of the data we have divided and we will test our model with the test part.
# * We split the data to 80% train and 20% test.

# In[ ]:


# data division train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)
print('Y_train: ',Y_train.shape)
print('Y_test: ',Y_test.shape)


# # Converting the 3D matrix to 2D matrix
# * We have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use as input for our first.
# * Our label array (Y) is already flatten(2D) so we leave it like that.

# In[ ]:


# Converting the 3D matrix to 2D matrix
print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten
x_test = X_test_flatten
y_train = Y_train
y_test = Y_test

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# # Logistic regression create model and  test accuracy score print
# * We create our logistic regression model.
# * We train our model with x_train and y_train data.
# * We test our model with x_test and y_test data and print the accuracy value.

# In[ ]:


# logistic regression create model and train test accuracy print
logreg = LogisticRegression(random_state = 42,max_iter= 150, penalty='l2')
logreg.fit(x_train, y_train.ravel())
print("test accuracy: {} ".format(logreg.score(x_test, y_test)))


# # Conclusion
# * Our test accuracy value is 97%. We can conclude from this value that we have created a correct model.
# * I'm new to programming. I'm even more new in data science, machine learning, deep learning and artificial intelligence. But I am working. And I'il be an artificial intelligence developer. Your comments are very important to me.
# * Thank you for reading my notebook. Waiting for your criticism.
