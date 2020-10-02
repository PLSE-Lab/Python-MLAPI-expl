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


# In[ ]:


dataset=pd.read_csv("../input/iris-dataset/iris.csv")


# In[ ]:


dataset.head()


# We will use the accuracy as the measure of accuracy on this dataset as this is a classification task.
# 

# **Cleaning the data**

# In[ ]:


#cheking for the null values and replacing them with the na values
dataset.isnull().sum()


# In[ ]:


#luckly we do not have any null values
#now we can describe the data
dataset.describe()


# **Visualization**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#making scatter plot by class
classes=dataset['species'].unique()
print("Different calsses---names")
print(classes)


# In[ ]:


sns.pairplot(dataset,hue='species')


# by the data visualization we can see that how the data is saperable clearly all the three species are seperabel easily

# Some clear intuation can be drawn from voilen plots as we can see the two class vergicolor and verginica are overlapping so we can gain some usefull insigit from the voilen plots

# In[ ]:


plt.figure(figsize=(10, 10))

for column_index, column in enumerate(dataset.columns):
    if column == 'species':
        continue
    plt.subplot(2, 2, column_index + 1)
    sns.violinplot(x='species', y=column, data=dataset)


# Violin plots are very helpfull in showing where the density of the data is going 

# We can see that the virginica tend to have the majority of its species petal width and similarly length in the highest range
# and similarly we can do for all to get more insight of the data

# **Train and Testing the data**

# *Splitting the data*

# In[ ]:


from sklearn.model_selection import train_test_split
labels = dataset["species"].values
new = dataset.drop(columns=["species"])
X_train,X_test,Y_train,Y_test = train_test_split(new,labels,test_size=0.25,random_state=42)


# In[ ]:


X_train.head()


# In[ ]:


Y_train[:5]


# We are using two models for now 
# 
# 1.Logitic Regression
# 
# 2.Svm

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test)
lr.score(X_test,Y_test)


# In[ ]:


#definately it is overfitting we can visualize the data to see the overfitting


# In[ ]:


from sklearn import svm
from sklearn.metrics import accuracy_score


# In[ ]:


svc = svm.SVC()
svc.fit(X_train,Y_train)
y_pred=svc.predict(X_test)
accuracy_score(Y_test,y_pred)


# i guess they are pretty much easily saperable in this case we can visualize the decision boundry we can not take the full use of it as the model is already preforming good

# In[ ]:




