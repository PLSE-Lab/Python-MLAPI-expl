#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[ ]:


x_train.shape
x_test.shape


# In[ ]:


for i in range(len(x_train[:10])):
    print(y_train[i])
    plt.imshow(x_train[i].reshape(28, 28))
    plt.show()


# In[ ]:


X_train = x_train[:1000,:]
X_train.shape


# In[ ]:


X_train = X_train.reshape(len(X_train),28*28)
X_train.shape


# In[ ]:


Y_train = y_train[:1000]
Y_train.shape


# In[ ]:


x_test = x_test.reshape(len(x_test),28*28)
x_test.shape


# In[ ]:


for i in range(len(x_test[:10])):
    print(y_test[i])
    plt.imshow(x_test[i].reshape(28, 28))
    plt.show()


# In[ ]:


clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 1)  
# KNN with K = 10 and p = 1, this is equivalent to using manhattan_distance (l1), 
#                   and euclidean_distance (l2) for p = 2. 
#                    For arbitrary p, minkowski_distance (l_p) is used.
clf.fit(X_train, Y_train)


# In[ ]:


y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred)*100


# In[ ]:


# for i in range(len(X_train[:1000])):
#     if Y_train[i] in [np.uint8(1),np.uint8(7)]:
#         print(Y_train[i])
#         plt.imshow(X_train[i].reshape(28, 28))
#         plt.show()


# In[ ]:


sample = x_test[-5:-1]
for i in range(len(sample)):
    plt.imshow(sample[i].reshape(28, 28))
    plt.show()


# In[ ]:


k = clf.predict(sample)
print(k)


# In[ ]:


sample = x_test[-5:-1]
for i in range(len(sample)):
    print(k[i])
    plt.imshow(sample[i].reshape(28, 28))
    plt.show()

