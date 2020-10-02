#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For data visualization
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('/kaggle/input/fish-market/Fish.csv')


# In[ ]:


#Displaying our dataframe
df


# In[ ]:


#Displaying shape of our dataframe
df.shape


# In[ ]:


#Displaying datatypes of features used in our dataframe
df.dtypes


# In[ ]:


#Checking for null values in our dataset
df.isnull()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.isnull(),yticklabels=False)


# *So there are no null values in our dataset.*

# In[ ]:


#Creating our new dataframe of float64 type varaibles
df2=df.iloc[:,1:7]


# In[ ]:


df2


# In[ ]:


#Checking outliers in our feature matrix dataframe
figure, axes=plt.subplots(nrows=3,ncols=2,figsize=(10,10))
sns.boxplot(df2['Weight'],color='green',ax=axes[0,0])
sns.boxplot(df2['Length1'],color='red',ax=axes[0,1])
sns.boxplot(df2['Length2'],color='blue',ax=axes[1,0])
sns.boxplot(df2['Length3'],color='green',ax=axes[1,1])
sns.boxplot(df2['Height'],color='red',ax=axes[2,0])
sns.boxplot(df2['Width'],color='blue',ax=axes[2,1])
plt.tight_layout()


# In[ ]:


#Removing Outliers through IQR method
name=df2.columns
q1=df2[name].quantile(0.25)
q3=df2[name].quantile(0.75)
IQR=q3-q1
l_w=q1-1.5*IQR
u_w=q3+1.5*IQR


# In[ ]:


df2=df2[df2[name]<u_w]
df2=df2[df2[name]>l_w]
df2


# In[ ]:


#Creating a new dataframe of our cleaned data
Species=df['Species']
df2['Species']=Species
df2


# In[ ]:


df2.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df2.isnull(),yticklabels=False)


# In[ ]:


#Removing outliers
df3=df2.dropna()


# In[ ]:


df3.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df3.isnull(),yticklabels=False)


# **Now our data is clean**

# In[ ]:


df3


# In[ ]:


df3.shape


# In[ ]:


sp_types=df3['Species'].value_counts()


# In[ ]:


sp_types


# In[ ]:


sns.barplot(x=sp_types.index,y=sp_types)
plt.xlabel('Species')
plt.ylabel('Count of Species')
plt.grid(color='grey',axis='y')


# # Creating feature matrix and dependent variable vector

# In[ ]:


#Creating feature matrix and dependent variable vector
X=df3.iloc[:,1:6]
Y=df3.iloc[:,0:1]


# In[ ]:


#Displaying our feature matrix
X


# In[ ]:


#Displaying our dependent variable vector
Y


# # Spliting our model

# In[ ]:


#Spliting our data in training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# # APPLYING OUR LINEAR REGRESSION MODEL

# In[ ]:


#Applying Linear Regression model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
predicted_wt=reg.predict(x_test)


# In[ ]:


Y_pred = pd.DataFrame(predicted_wt, columns=['Estimated Weight'])
Y_pred.head(10)


# In[ ]:


Y_test = pd.DataFrame(y_test)
Y_test = Y_test.reset_index(drop=True)
Y_test.head(10)


# In[ ]:


# Printing the actual weight and predicted weight in a dataframe
final_data = pd.concat([Y_test, Y_pred], axis=1)
final_data


# In[ ]:


#Calculating R_square error
from sklearn import metrics
r2_square=metrics.r2_score(Y_test,Y_pred)
print('R_SQUARE ERROR: ',r2_square)


# Our predicted model has a R_Square error of 0.91 which is a good value.
# 

# In[ ]:




