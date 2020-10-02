#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# * [Reading data](#read)
# * [Checking first 10 data for look features](#head)
# * [Checking correlation between features](#cor)
# * [Checking data types](#info)
# * [Converting feature to int](#con)
# * [Normalization](#norm)
# * [Total "class" values count](#count)
# * [Splitting data to train and test](#split)
# * [Implementing KNN](#knn)

# In[27]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="#read"> </a>
# ## READ DATA

# In[28]:


data= pd.read_csv('../input/column_2C_weka.csv')


# <a id="#head"></a>
# ## CHECKING FIRST 10 DATA FOR LOOK FEATURES

# In[29]:


data.head(10)


# <a id="#cor"> </a>
# ## CHECKING CORRELATION BETWEEN FEATURES

# In[30]:


f, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# <a id="#info"> </a>
# ## CHECKING DATA TYPES

# In[23]:


data.info()


# <a id="#con"> </a>
# ## CONVERTING FEATURE TO INT
# As we can see "class" is "object". We can't use "object" for classifaciton problems. The Logistic Algortihm is has to be 2 situation. For this dataset our situatins is "Abnormal" or "Normal" and there outputs has to be 0 or 1, so lets convert to "int"

# In[31]:


data["class"] = [1 if(each == "Abnormal") else 0 for each in data["class"]]

y= data["class"].values
x_data= data.drop(["class"],axis=1)


# <a id="norm"> </a>
# ## NORMALIZATION

# In[32]:


x= (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))


# <a id="count"> </a>
# ## TOTAL CLASS VALUES COUNT    

# In[13]:


sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()


# <a id="#split"> </a>
# ## SPLITTING DATA TO TRAIN AND TEST

# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=1)


# <a id="knn"> </a>
# ## IMPLEMENTING KNN
# K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).

# In[37]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x,y)
prediction = knn.predict(x)
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))


# As we can see accuracy %70. Its not a good result for this algortihm we can change parameters for reach optimal result.

# In[41]:


score_list = []
for each in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,20),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# If we set "n_neighbors" to 19 we will reach best accuarcy.
