#!/usr/bin/env python
# coding: utf-8

# ![190921_07_40_09_5DS27847.0.jpg](attachment:190921_07_40_09_5DS27847.0.jpg)
# 
# **This notebook has two parts,**
# 
# **Part1 - Data Analysis and Visualization**
# Here the data is analyzed with respect to the features to obtain a meaningful visualization which can be used to tella story and define the relationship that exists.
# 
# **Part2 - Regression Model**
# In this part the features are measured for their p value using Ordinary Least Square method and manually removing the features having a p value higher than the significance value which in this case is 0.05. After which the best features are used to train regression models and their accuracy is compared.
# 
# So, lets begin!

# # Part 1 - Data Analysis and Visualization

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#load the dataset
dataset = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


#first 5 rows
dataset.head()


# In[ ]:


#check if there is any null value
dataset.isnull().sum()


# In[ ]:


#datatypes
dataset.dtypes


# In[ ]:


#dropping the null values
dataset = dataset.drop(['name', 'host_name', 'last_review', 'reviews_per_month'], axis = 1)


# In[ ]:


#checking the dataset
dataset.isnull().sum()


# In[ ]:


#description of the dataset
dataset.describe().T


# *Lets now analyze the data*

# In[ ]:


#different types of rooms
dataset.room_type.unique()


# In[ ]:


#different neighbourhoods
dataset.neighbourhood.unique()


# In[ ]:


#number of different neighbourhoods
d = dataset.neighbourhood.unique()
len(d)


# In[ ]:


#different neighbourhood groups
dataset.neighbourhood_group.unique()


# In[ ]:


#count plot for neighbourhood grouos
sns.countplot(dataset.neighbourhood_group, data = dataset)


# In[ ]:


#countplot for types of room
sns.countplot(dataset.room_type, data = dataset)


# In[ ]:


#room type based on the neighbourhood
sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = dataset)


# In[ ]:


#distplot to see price
#distplot for availability
#distplot for minimum nights
df1 = dataset[dataset.price < 500]
df2 = dataset[dataset.availability_365 < 370]
df3 = dataset[dataset.minimum_nights < 200]
f, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(df1.price, ax = axes[0])
sns.distplot(df2.availability_365, ax = axes[1])
sns.distplot(df3.minimum_nights, ax=axes[2])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[ ]:


#each neighbourhood group price variation
f, axes = plt.subplots(3,2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(df1[(df1.neighbourhood_group == 'Brooklyn')]['price'],color='k', axlabel = 'Brooklyn Price', ax = axes[0,0])
sns.distplot(df1[(df1.neighbourhood_group == 'Manhattan')]['price'],color='k', axlabel = 'Manhattan Price', ax = axes[0,1])
sns.distplot(df1[(df1.neighbourhood_group == 'Queens')]['price'],color='k', axlabel = 'Queens Price', ax = axes[1,0])
sns.distplot(df1[(df1.neighbourhood_group == 'Staten Island')]['price'],color='k', axlabel = 'Staten Island Price', ax = axes[1,1])
sns.distplot(df1[(df1.neighbourhood_group == 'Bronx')]['price'],color='k', axlabel = 'Bronx Price', ax = axes[2,0])


# In[ ]:


#price in each neighbourhood group
sns.set(style="ticks", palette="pastel")
sns.boxplot(x = dataset.neighbourhood_group, y = dataset.price,  data = dataset)
sns.despine(offset=10, trim=True)


# In[ ]:


#price distribution wrt minimum nights
sns.jointplot(y = dataset.price, x = dataset.minimum_nights)


# In[ ]:


#price and availability
sns.jointplot(y = dataset.price, x = dataset.availability_365)


# In[ ]:



#availability wrt price and neighbourhood groups
sns.scatterplot(x = dataset.availability_365, y = dataset.price, hue = dataset.neighbourhood_group, data = dataset)


# In[ ]:


#price with respect to neighbourhood group
sns.scatterplot(x = dataset.price, y = dataset.neighbourhood_group, data = dataset)


# In[ ]:


#violin plot for neighbourhood group with price and room type
sns.catplot(x="neighbourhood_group", y="price", hue="room_type",
            kind="violin", data=df1)


# In[ ]:


#top 5 neighbouhoods
plt.figure(figsize = (6,6))
df4 = dataset.neighbourhood.value_counts().head(5)
sns.barplot(x = df4.index, y = df4.values)


# In[ ]:


#top 5 host id
plt.figure(figsize = (6,6))
df5 = dataset.host_id.value_counts().head(5)
sns.barplot(x = df5.index, y = df5.values)


# In[ ]:


#latitude and longitude
plt.figure(figsize = (10,10))
sns.scatterplot(dataset.longitude, dataset.latitude, hue = dataset.neighbourhood_group)
plt.ioff()


# In[ ]:


#based on room type
plt.figure(figsize = (10,10))
sns.scatterplot(dataset.longitude, dataset.latitude, hue = dataset.room_type)
plt.ioff()


# # Part 2 - Regression Models
# 
# Different regression models are used to determine the model with the best accuracy. The models are
# 1. Multivariate Linear Regression
# 1. SVM Regression
# 1. Random Forest Regression

# In[ ]:


#check the dataset
dataset


# In[ ]:


#check datatypes
dataset.dtypes


# In[ ]:


#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.neighbourhood_group = le.fit_transform(dataset.neighbourhood_group)
dataset.neighbourhood = le.fit_transform(dataset.neighbourhood)
dataset.room_type = le.fit_transform(dataset.room_type)


# In[ ]:


#check correlation with respect to price
dataset.corr()['price'].sort_values


# In[ ]:


#visual display of correlation
plt.figure( figsize = (10,10))
sns.heatmap(dataset.corr(), annot = True)


# In[ ]:


#split the dataset into X and y
X = dataset.drop(['price'], axis = 1).values
y = dataset.price.values


# In[ ]:


#p value calculation
#backward elimination model used to determine the best features
import statsmodels.api as sm
X = np.append(arr = np.ones((48895,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
reg_OLS = sm.OLS(y,X_opt).fit()
reg_OLS.summary()


# In[ ]:


#removing 8 - minimum_nights
X_opt = X[:, [0,1,2,3,4,5,6,7,9,10,11]]
reg_OLS = sm.OLS(y,X_opt).fit()
reg_OLS.summary()


# In[ ]:


#removing 1 - host_id
X_opt = X[:, [0,2,3,4,5,6,7,9,10,11]]
reg_OLS = sm.OLS(y,X_opt).fit()
reg_OLS.summary()
#best features


# In[ ]:


#updating the values of X
X = X_opt[:,1:]


# In[ ]:


#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **Multivariate Linear Regression**

# In[ ]:


#Multivariate Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predict the value
y_pred = regressor.predict(X_test)


# In[ ]:


#check metrics
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)


# In[ ]:


#mse value
mean_squared_error(y_test, y_pred)


# **Support Vector Machine Regression**

# In[ ]:


#SVM
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
#predict the value
y_pred = regressor.predict(X_test)


# In[ ]:


#r2 score
r2_score(y_test, y_pred)


# In[ ]:


#mse value
mean_squared_error(y_test, y_pred)


# **Random Forest Regression**

# In[ ]:


#random forest 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50)
regressor.fit(X_train, y_train)
#predict the value
y_pred = regressor.predict(X_test)


# In[ ]:


r2_score(y_test, y_pred)


# In[ ]:


#mse value
mean_squared_error(y_test, y_pred)


# In[ ]:


#regression plot of random forest
sns.regplot(y = y_test, x = y_pred, color = 'blue')


# The Random Forest Regression gives the model with the most accuracy for the given dataset.

# *I hope this notebook was helpful!*

# In[ ]:




