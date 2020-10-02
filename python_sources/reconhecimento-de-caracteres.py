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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import sklearn.datasets as dt


# 

# In[ ]:


dic = dt.load_digits()
dic.keys()


# In[ ]:


dic.data.shape


# In[ ]:


dic.images.shape


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow


# In[ ]:


x=dic.data
y=dic.target


# In[ ]:


sklearn.datasets = iris
sklearn.datasets[, -5] = scale(iris[, -5])


# In[ ]:


train_index = sample(1:nrow(sklearn.datasets)
treino = data.frame()
treino = iris_normal[train_index,]
 


# In[ ]:





x=dic.data
y=dic.target
x_train=input_variables_values_training_sklearn.datasets 
y_train=target_variables_values_training_sklearn.datasets 


linear = linear_model.LinearRegression()
#Treina o modelo usando os dados de treino e confere o score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
dic.(keys)

