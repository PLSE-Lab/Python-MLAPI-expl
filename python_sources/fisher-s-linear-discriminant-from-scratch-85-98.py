#!/usr/bin/env python
# coding: utf-8

# **What are we trying to do?**
# 
# We will be building a Fisher's Linear Discriminant from scratch, The accuracy score is not bad for a linear classifier.
# This classifier works by trying to find the best decision boundry given that it would maximize separation between classes means while minimizing the within-class variance. 
# 
# The image on the left shows a bad decision boundry while the one on the right shows a good one. Discriminant function performs dimensionality reduction, we are trying to find the line which if the data was projected on, would give us the maximum separation between classes and smallest within class variance.
# 
# <img src="https://drive.google.com/uc?id=14RIvA5W5rE0rfJFapE37pM_xTH9VH5de">
# 
# We will be using the One-versus-the-rest approach for class decisions.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Load the data**:
# 
# Just separating the labels from the features, count the classes and features.
# Storing the indexes of every class for later use.
# 

# In[ ]:


# Load Training Images:

im_train = np.loadtxt('../input/train.csv', delimiter=',', dtype=int, skiprows=1)
train_labels = im_train[:, 0]
im_train = im_train[:, 1:]
nclasses = len(np.unique(train_labels))
nfeatures = np.size(im_train, axis=1)
class_indexes = []
for i in range(nclasses):
    class_indexes.append(np.argwhere(train_labels == i))


# In[ ]:


im_train.shape


# SW, the within-class covariance matrix equation is as below (shouldn't be hard to understand if you break it down)
# xn are the data points, m1 is the mean of class 1 and m2 is the mean of class 2 (all other classes combined) 
# <img src="https://drive.google.com/uc?id=1QrpVfz8g_i0aWIHSXbU9ssdyusdCxPUv">

# The weights vector, w, which is orthogonal to the decision boundy and w0 (the bias) are equal to:
# 
# <img src="https://drive.google.com/uc?id=1hJlV-v4nU0drdIA6VSmWga_qlBVkNmBd">
# 

# In[ ]:


# Initializing needed variables
class_means, other_class_means = np.empty((nclasses, nfeatures)), np.empty((nclasses, nfeatures))
other_class = []
SW_one, SW_two, SW = np.zeros((nclasses, nfeatures, nfeatures)), np.zeros((nclasses, nfeatures, nfeatures)), np.zeros((nclasses, nfeatures, nfeatures))
W = np.zeros((nclasses, nfeatures, 1))
W0 = np.zeros((nclasses))


# Using some vectorization and matrix properties we can avoid all the looping on the dataset to create SW_one and SW_two and end up with a model that does not actually take a long time to train.

# In[ ]:


# Calculating SW, W & W0 #
for i in range(nclasses):
    class_means[i] = np.mean(im_train[class_indexes[i]], axis=0)
    other_class.append(np.delete(im_train, class_indexes[i], axis=0)) # one-versus-the-rest approach
    other_class_means[i] = np.mean(other_class[i], axis=0)
    between_class1 = np.subtract(im_train[class_indexes[i]].reshape(-1, nfeatures), 
                                 class_means[i])
    SW_one[i] = between_class1.T.dot(between_class1)
    between_class2 = np.subtract(other_class[i], other_class_means[i])
    SW_two[i] = between_class2.T.dot(between_class2)
    SW[i] = SW_one[i] + SW_two[i]
    W[i] = np.dot(np.linalg.pinv(SW[i]), 
                  np.subtract(other_class_means[i], 
                              class_means[i]).reshape(-1, 1))
    W0[i] = -0.5 * np.dot(W[i].T, (class_means[i] + other_class_means[i]))


# In[ ]:


print(SW.shape)
print(W.shape)
print(W0.shape)


# **Load the test data**:
# 

# In[ ]:


im_test = np.loadtxt('../input/test.csv', delimiter=',', dtype=int, skiprows=1)
im_test.shape


# **Classification**:
# <img src="https://drive.google.com/uc?id=1kMuH5FEGLuGR41-OGxmvEmsff1gCqj0B">
# 
# We would calculate Y for every image we want to classify, every Y is a 1D array having length = number of possible classes.
# The prediction is simply the argmin of Y (index of smallest Y)

# In[ ]:


Y = np.zeros((len(im_test), nclasses))
predict = np.zeros((len(im_test)), dtype=int)
for j in range(len(im_test)):
    for i in range(nclasses):
        Y[j, i] = np.dot(W[i].T,  im_test[j]) + W0[i]
    predict[j] = np.argmin(Y[j])


# In[ ]:


predict[:10]


# Sounds good, let's try to plot the first 10 testing images too.

# In[ ]:


for i in range(10):
    plt.subplot(1, 10, i+1) # plot index can not be 0
    plt.imshow(im_test[i].reshape(28, 28))
    plt.axis('off')
plt.show()


# So apparently it made 9 correct decision out of 10. This is very good for a linear discriminant.

# In[ ]:


submission = pd.DataFrame({"ImageId": np.arange(1, len(im_test)+1), "Label": predict})
submission.to_csv('submission.csv', index=False)

