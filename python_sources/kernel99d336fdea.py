#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # seaborn for visualizationn
import matplotlib.pyplot as plt  # for visualization
from sklearn.linear_model import LinearRegression  #Import Linear regression model
from sklearn.model_selection import train_test_split  #To split the dataset into Train and test randomly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/insurance.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


# Defining dependent and independent variable

x= df.iloc[:,0:7].values




# In[ ]:


y= df.iloc[:, 7:8].values


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace = True)


# In[ ]:


df.duplicated().sum()


# In[ ]:


# to convert categorical to numerical by label encoder

from sklearn.preprocessing import LabelEncoder
Label_x= LabelEncoder


# In[ ]:


x[:,5] = Label_x.fit_transform(x[:,5])


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[ ]:


Label_x= LabelEncoder()


# In[ ]:


x[:,4]= Label_x.fit_transform(x[:,4])


# In[ ]:


x[:,1]= Label_x.fit_transform(x[:,1])


# In[ ]:


onehotencoder_x= OneHotEncoder(categorical_features=[1])


# In[ ]:


x=onehotencoder_x.fit_transform(x).toarray()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


ax = sns.boxplot(df['age'])
ax.set_title('Dispersion of Age')
plt.show(ax)


# In[ ]:


#To split test and train


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=.2, random_state = 0)


# In[ ]:


xtrain


# In[ ]:


ytrain


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.scatterplot(x='age',y='expenses',hue = 'bmi',size = 'bmi', data=df)
ax = ax.set_title("Age vs Expenses by BMI")
plt.xlabel("Age")
plt.ylabel("Expenses")
plt.show(ax)


# In[ ]:


#Scatter plot clearly states that, Age with sex are not influencing the expenses.
plt.figure(figsize=(10,7))
ax = sns.scatterplot(x='age',y='expenses', hue='sex',style = 'sex',data=df)
ax.set_title("Age vs Expenses by Sex")
plt.show(ax)


# In[ ]:


#Both Age and smoker are highly influncing the expenses. Smoker yes
plt.figure(figsize=(10,7))
ax = sns.scatterplot(x='age',y='expenses', hue=df['smoker'],style = df['smoker'],size = df['smoker'], data=df)
ax.set_title("Age vs Expenses by Smoker")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Expenses")
plt.show(ax)

df.corr()
# In[ ]:


df.corr()


# In[ ]:


#Swarm plot shows how smoker feature is influencing the expeneses compare with smoker and non-smoker
ax = sns.swarmplot(x='smoker',y='expenses',data=df)
ax.set_title("Smoker vs Expenses")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Expenses")
plt.show(ax)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.iloc[:,4] = labelencoder.fit_transform(df.iloc[:,4])


# In[ ]:


df.head()


# In[ ]:


x = df[['age','bmi','smoker']]
y = df['expenses']


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(xtrain, ytrain) 


# In[ ]:


print("Intercept value:", model.intercept_)
print("Coefficient values:", model.coef_)


# In[ ]:




