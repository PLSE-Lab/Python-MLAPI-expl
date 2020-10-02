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

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[ ]:


df=pd.read_csv("../input/housing-in-london/housing_in_london.csv")
df.head()


# In[ ]:


df.describe()


# Considering the null values present in the data-set********
# 

# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# 

# In[ ]:


df.drop(['no_of_houses','date'],axis=1,inplace=True)
df.drop(['recycling_pct','life_satisfaction','median_salary','mean_salary'],axis=1,inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df['no_of_crimes']=df['no_of_crimes'].fillna(df['no_of_crimes'].mean())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# Ensure, we dont have any null values present in dataset,since this will not fit in our algorithm

# In[ ]:


df['houses_sold']=df['houses_sold'].fillna(df['houses_sold'].mean())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df.info()


# We can see, there are datatypes=objects in our dataset, which we need to convert into float/int values.

# In[ ]:


df.shape


# Now, since our column "code",exist of values, like E9000.., we'll replace the value "E" with "nothing", so that our object datatype will be converted into a float value.

# In[ ]:


df['code']=df.code.str.replace('E','').astype(float)
df.info()

Considering the Area column,its still in object type, so here we assign each of the city with a unique number,eg "london"="1","enfield"="2", and so on.
# In[ ]:


df['area'] = pd.factorize(df.area)[0]
df['area'] = df['area'].astype("float")
df.info()


# In[ ]:


df.describe()


# In[ ]:


df.head()


# Taking our features into X, while taking our target features into y, for prediction

# In[ ]:


X=df[['area','code','houses_sold','no_of_crimes','borough_flag']]
y=df[['average_price']]


# Spliting the Dataset into train and test.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=1,)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state=0,min_samples_split=3)
model.fit(X_train,y_train)


# In[ ]:


prediction=(model.predict(X_test).astype(int))
print("predictions:",prediction)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(prediction,y_test)


# In[ ]:




