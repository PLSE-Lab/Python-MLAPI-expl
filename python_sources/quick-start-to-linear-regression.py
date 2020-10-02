#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import style
style.use('ggplot')


# Import dataset, we will perform linear regression of imported dataset.

# In[ ]:


df = pd.read_csv("../input/ex1data1.txt", header=None)
df.columns = ["X", "Y"]
print(df.head())


# Lets visualize these data points.
# We can use matplotlib for visualisation purpose.

# In[ ]:


plt.scatter(df.X, df.Y, color='black')


# To make dependent and independent variables in same range, we should scale data. This process is also known as **feature scaling**.

# In[ ]:


X = np.array(df.drop(['Y'],1))
Y = np.array(df.Y)
Scaled_X = preprocessing.scale(X)
Scaled_Y = preprocessing.scale(Y)


# We can fit linear line into given data points. Its equation will be like **Y = m*X+c**, where m and c are parameters which can be learnt by algorithm. Linear regression uses generally mean square error (MSE) andreduce that loss with the help of** gradient descent** algorithm. **MSE** is highly effected by **outliers**, so make sure that your dataset have no outliers.

# Lets split our data into test and train with ratio 80:20 in training and validation set.

# In[ ]:


X_train , X_test ,  Y_train , Y_test = train_test_split(Scaled_X,Scaled_Y,test_size=0.1)


# Perform linear regression.

# In[ ]:


clf = LinearRegression()
clf.fit(X_train,Y_train)


# In[ ]:


plt.scatter(X_test,Y_test,color='black')
plt.scatter(X_test,clf.predict(X_test),color='red')
plt.plot(X_train,clf.predict(X_train),color="green")


# Black dots are actual data points and red dots are predicted data points. We can also visualize best fit regression line(green).

# In[ ]:


accuracy = clf.score(X_test,Y_test)
print(accuracy)


# In[ ]:





# In[ ]:




