#!/usr/bin/env python
# coding: utf-8

# # By K-NN Algo predict gender based on "Occupation, Purchase"and genereting output.

# As heading suggested we use the K-NN algo to predict accuracy and genereting output using Sale.csv data set. 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import countplot

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 1. ** split data with 0.23% of TestSize.**

# In[ ]:


df = pd.read_csv("../input/Sale.csv")


# **df.shape** give the info of data set, how many rows and colunms in the data set.
# 

# In[ ]:


df.shape


# By **df.info()** function we chack our data set all rows and column if any row or column containing null(NaN) values then this function will show it. It will show by the column and also show the datatype of column, means a particuler column have float values or intger values. 

# In[ ]:


df.info()


# In[ ]:


x = df[["Occupation", "Purchase"]].values
y = df["Gender"].values


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.5, random_state = 12)


# 2. Training model with this data and finding the accuracy for test data.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=12)
knn  = knn.fit(x_train, y_train)


# In[ ]:


y_pred = knn.predict(x_train)
accuracy_score(y_train, y_pred)


# In[ ]:


y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)


# 3. Training model with different K values (1, 30) and plot train_accuracy and test_accuracy plot.

# In[ ]:


train_acc = []
test_acc = []
for k in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn  = knn.fit(x_train, y_train)
    
    y_train_pred = knn.predict(x_train)
    y_test_pred = knn.predict(x_test)
    
    train_acc.append(accuracy_score(y_train_pred, y_train))
    test_acc.append(accuracy_score(y_test_pred, y_test))
    print("Step -", k)


# In[ ]:


plt.figure(figsize=(15, 6))
plt.plot(np.arange(1,31),test_acc, label = "Test Acc")
plt.plot(np.arange(1,31),train_acc, label = "Train Acc")
plt.scatter(np.arange(1,31),test_acc)
plt.scatter(np.arange(1,31),train_acc)
plt.legend()
plt.show()


# 4. Finding best value for K, where we will get best test accuracy. 

# In[ ]:


test_max = max(test_acc)
k = test_acc.index(test_max) + 1
k


# **On k = 29 we get beat test accuracy of our model. **

# 5. Ploting Count Plot based on Gender.

# In[ ]:


countplot(y)

