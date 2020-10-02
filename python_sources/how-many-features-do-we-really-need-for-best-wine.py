#!/usr/bin/env python
# coding: utf-8

# In this mini tutorial I show the process of picking the best number of variables for predictions using a method called Principal Components Analysis (PCA) ;)

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


# Importing our dataset:

# In[ ]:


from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv('../input/winequality-red.csv')

data.head()


# In[ ]:


# X = data[[data.columns]]
X = data.drop('quality',axis=1)
y = data.quality
X.head()


# When standardizing data, the following formula is applied to every data point:
# 
# Z = (Sample - Mean)/(Stan.Dev)
# 
# In other words, we are calculating z-scores, centering the samples by the mean and th standard deviation.

# In[ ]:


X = preprocessing.StandardScaler().fit(X).transform(X)


# In[ ]:


model = PCA()
results = model.fit(X)
plt.plot(results.explained_variance_)
plt.show()


# As we can see, the more variables we add the more of the information we represent.
# 
# i would say that 5 varaibles is a good meausure.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

gnb = GaussianNB()
fit = gnb.fit(X,y)
pred = fit.predict(X)
print (confusion_matrix(pred,y))
print("accuracy: ")
print(confusion_matrix(pred,y).trace()*100/confusion_matrix(pred,y).sum())


# Now,  let's see how much the accuracy get's affected with different number of variables:

# In[ ]:


predicted_correct = []
for i in range(1,10):
    model = PCA(n_components = i)
    results = model.fit(X)
    Z = results.transform(X)
    fit = gnb.fit(Z,y)
    pred = fit.predict(Z)
    predicted_correct.append(confusion_matrix(pred,y).trace())
plt.plot(predicted_correct)
plt.show()


# The plot shows that with only 3  variables. 
# also adding more variables beyond 5 doesn't add much predictive power as the first 5.

# In[ ]:





# In[ ]:





# In[ ]:




