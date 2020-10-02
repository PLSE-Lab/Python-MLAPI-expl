#!/usr/bin/env python
# coding: utf-8

# In[77]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot
from matplotlib.pyplot import plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[78]:


Iris=pd.read_csv("../input/Iris.csv")


# In[79]:


Iris.head()


# In[80]:


Iris.shape


# In[81]:


print(Iris[Iris.Species=='Iris-setosa'].describe())
print(Iris[Iris.Species=='Iris-virginica'].describe())
print(Iris[Iris.Species=='Iris-versicolor'].describe())


# In[82]:


Iris.isnull().any()


# In[83]:


Iris.groupby(['Species']).count()


# In[84]:


df=Iris
g = sns.FacetGrid(Iris, col="Species")
g.map(sns.kdeplot,"SepalLengthCm")


# In[85]:


df=Iris
g = sns.FacetGrid(Iris, col="Species")
g.map(sns.kdeplot,"SepalWidthCm")


# In[86]:


df=Iris
g = sns.FacetGrid(Iris, col="Species")
g.map(sns.kdeplot,"PetalLengthCm")


# In[87]:


df=Iris
g = sns.FacetGrid(Iris, col="Species")
g.map(sns.kdeplot,"PetalWidthCm")


# #### Insights from above plots:-
# * 

# In[88]:


Iris.at[0,'Species']


# In[89]:


for i in range(len(Iris)):
    if(Iris.at[i,'Species'] =='Iris-setosa'):
        Iris.loc[[i],['Species']]=1
    elif(Iris.at[i,'Species'] == 'Iris-versicolor') :
        Iris.loc[[i],['Species']]=2
    else:
        Iris.loc[[i],['Species']]=3    


# In[90]:


Iris


# In[91]:


y = Iris.Species
Iris_predictors=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']


# In[92]:


X=Iris[Iris_predictors]


# In[93]:


from sklearn.tree import DecisionTreeRegressor
Iris_model = DecisionTreeRegressor()
Iris_model.fit(X, y)


# In[97]:


print("making predictions for first five records")
print(X.head())
print("predictions are:-")
print(Iris_model.predict(X.tail()))


# In[100]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
Iris_model.fit(train_X, train_y)
val_predictions = Iris_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[110]:


A=pd.DataFrame({"predicted":val_predictions,"actual":val_y})


# In[112]:


A[A.predicted != A.actual]


# In[ ]:




