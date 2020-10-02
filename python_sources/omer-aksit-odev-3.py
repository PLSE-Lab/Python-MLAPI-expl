#!/usr/bin/env python
# coding: utf-8

# **DIAGNOSE AND CLEAN DATA: AN INTRO**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


data.shape     #checking how many rows and columns there are


# In[ ]:


data.columns         #We will check the columns first


# In[ ]:


data.head()      #checking the first 5 rows to get an insight


# In[ ]:


data.tail()    #checking the last 5 rows to get an insight.


# In[ ]:


data.dtypes       #checking data types of columns


# In[ ]:


#frequency of chest pain types
data["cp"].value_counts(dropna=False)


# In[ ]:


data["age"].describe()


# In[ ]:


#checking Q1, Q2, Q3 values manually
data1=data["age"].sort_values()   #data1 is sorted but index numbers did not change.
data2=[i for i in data1] #data2 has the same values as data1, but index numbers are reassigned, they goes from 0 to forward.
print("Q1:",data2[75])
print("median:",data2[151])
print("Q3:",data2[227])


# In[ ]:


data[["age"]].boxplot()   #There is no outlier as we can see below.
plt.show()


# In[ ]:


data.boxplot(column="age")
plt.show()


# In[ ]:


data.head()


# In[ ]:


#tidy data (melting)
data3=data.head()
print(data3)

melted=pd.melt(data3,id_vars="age",value_vars=["trestbps","oldpeak"])
melted


# In[ ]:


#pivoting melted data
melted.pivot(index="age",columns="variable",values="value")


# In[ ]:


#concatenating data
data4=data.loc[0:4,["trestbps","chol"]]
data5=data.loc[0:4,["fbs","restecg"]]
concat1=pd.concat([data4,data5],axis=1,sort=False)
print(concat1)
print("")
concat2=pd.concat([data4,data5],axis=0,sort=False,ignore_index=True)
print(concat2)


# In[ ]:


#data types
print(data.dtypes)
data["sex"]=data["sex"].astype("int32")      #changing data type of "sex" column
data["trestbps"]=data["trestbps"].astype("category")   #changing data type of "trestbps" column
print(data.dtypes)      #new data types


# In[ ]:


concat2["trestbps"].value_counts(dropna=False)    #determining NaN values in trestbps column of concat2 dataframe


# In[ ]:


concat2["trestbps"].dropna(inplace=True)     #dropping NaN values
concat2["trestbps"].value_counts(dropna=False)   #check if we could drop NaN values


# In[ ]:


assert concat2["trestbps"].notnull().all()    #check if we could drop NaN values


# In[ ]:


concat2["trestbps"].fillna("empty",inplace=True)    #filling NaN values
concat2["trestbps"].value_counts(dropna=False)      #check the trestbps column

