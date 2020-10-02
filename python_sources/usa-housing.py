#!/usr/bin/env python
# coding: utf-8

# This notebook is dedicated towards predicting sale price of plots.
# In this tutorial our focus is purely on predicting sales price

# First thing first, lets import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# loading csv file data to our DataFrame df

# In[ ]:


data = pd.read_csv('../input/usa-housing-dataset/housing_train.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# Checking for null values 

# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(data.isna())


# This dataset contains plenty of null values. Lets remove/fix it

# In[ ]:


nullCols = []

for i in data.columns:
    if(data[i].isna().sum()>=146):
        data.drop(i,inplace=True,axis=1)
    else:
        data[i] = data[i].fillna(method='ffill')


# In this code block if any column contains null value greater than 146(10% of our dataset) we will remove those columns and for rest we will fix it forward fix method

# In[ ]:


sns.heatmap(data.isna())


# In[ ]:


len(data.columns)


# In[ ]:


data.duplicated().sum()


# In[ ]:


data.info()


# Checking for correlation

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')


# You can plot such more plot, using countplot, FacetGrid for data visualization

# In[ ]:


sns.countplot(data=data,x='SaleCondition')


# Let's convert categorical data into numeric

# Creates list of columns whose data type is object

# In[ ]:


dt = data.dtypes==object
col = data.columns[dt].tolist()


# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# Applied Label Encoder

# In[ ]:


le = LabelEncoder()
x = data[col].apply(le.fit_transform)


# One Hot Encoding

# In[ ]:


oneList=[]
one = OneHotEncoder()
oneData = one.fit_transform(x).toarray()


# In[ ]:


oneData.shape


# Droping column Id as it is of no use

# In[ ]:


data.drop('Id',inplace=True,axis=1)
dt = data.dtypes!=object
col = data.columns[dt].tolist()


# 

# In[ ]:


new_x = np.append(data[col],x,axis=1)
new_x.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Spliting our data into train and test

# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(new_x,data['SalePrice'],test_size=0.2,random_state=101)


# Applied Standard Scaler for normalising/scaling our dataset

# In[ ]:


sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


lr = LinearRegression()
lr.fit(train_x,train_y)
predict = lr.predict(test_x)
print('MAE ',mean_absolute_error(test_y,predict))
print('MSE ',mean_squared_error(test_y,predict))
print('RMSE ',np.sqrt(mean_squared_error(test_y,predict)))


# In[ ]:




