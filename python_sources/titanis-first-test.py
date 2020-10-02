#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_gender = pd.read_csv('../input/gender_submission.csv', sep=',')
titanic_gender.shape


# In[ ]:


data = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


titanic_train.isnull().any()


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


titanic_train['SibSp'].isnull().value_counts()


# In[ ]:


data.head(7)


# In[ ]:


data.columns


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='PassengerId', y='Survived',alpha = 0.5,color = 'red')
plt.xlabel('PassengerId')              # label = name of label
plt.ylabel('Survived')
plt.title('PassengerId Survived Graph')            # title = title of plot


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='PassengerId',by = 'Survived')


# In[ ]:


data.head()


# In[ ]:


data_new = data.head()    # I only take 5 rows into new data
data_new


# In[ ]:


data1 = data.head()
data2= data.tail()
data3= data_new
conc_data_row = pd.concat([data1,data2,data3],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# In[ ]:


data.dtypes


# In[ ]:


data1 = data.loc[:,["PassengerId","Survived"]]
data1.plot()
# it is confusing


# In[ ]:


data1.plot(subplots = True)
plt.show()


# In[ ]:


data1.plot(kind = "scatter",x="Survived",y = "PassengerId")
plt.show()


# In[ ]:


data1.plot(kind = "hist",y = "Survived",bins = 3,range= (0,1),normed = True)
plt.show()


# In[ ]:


data.describe()


# In[ ]:


data1.describe()


# In[ ]:


data2.describe()


# In[ ]:


data3.describe()


# In[ ]:


# End of test data :)
# First test of this data, next one is train data study...

