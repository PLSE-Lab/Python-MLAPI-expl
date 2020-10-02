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
print(os.listdir("../input/waves-measuring-buoys-data-mooloolaba"))

# Any results you write to the current directory are saved as output.


# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np


# # Upload the dataset

# In[ ]:


waves = pd.read_csv('../input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv')


# In[ ]:


# View the data
waves.head(10)


# # Change the Objects as Meaning Full

# In[ ]:


waves = waves.rename(columns = {'Hs' : 'significant_wave_height' , 'Hmax' : 'maximum_wave_height', 'Tz' : 'zero_wave_period',
                       'Tp' : 'peak_wave_period' , 'SST' : 'sea_surface_temperature' , 'Peak Direction' : 'peak_direction'})


# In[ ]:


waves.head(10)


# In[ ]:


# find the variables and Objects on the Data
waves.shape


# In[ ]:


# Find the type of data in the dataset
waves.info()


# # Check if any null values in the dataset

# In[ ]:


waves.isnull().sum()


# There is no NA/NAN in the data set

# # Data Cleaning

# In[ ]:


# Delete First Column
waves1 =waves.drop(columns={'Date/Time'})
waves1.head(10)


# # Exploratory Data Analysis (EDA)

# In[ ]:


waves.describe().transpose()


# # Skewness and Kurtosis

# In[ ]:


# Import Libraries
from scipy.stats import skew ,kurtosis


# In[ ]:


# Find the skewness and Kurtosis on Wave Height Column
print("Skewness of the Waves Height : " ,skew(waves['significant_wave_height']))
print("Kurtosis of the Waves Height : " ,kurtosis(waves['significant_wave_height']))


# In[ ]:


# Find the skewness and Kurtosis on maximum_wave_height Column
print("Skewness of the maximum_wave_height : " ,skew(waves['maximum_wave_height']))
print("Kurtosis of the maximum_wave_height : " ,kurtosis(waves['maximum_wave_height']))


# In[ ]:


# Find the skewness and Kurtosis on zero_upcrossing_wave_period Column
print("Skewness of the zero_upcrossing_wave_period : " ,skew(waves['zero_wave_period']))
print("Kurtosis of the zero_upcrossing_wave_period : " ,kurtosis(waves['zero_wave_period']))


# In[ ]:


# Find the skewness and Kurtosis on peak_energy_wave_period Column
print("Skewness of the peak_energy_wave_period : " ,skew(waves['peak_wave_period']))
print("Kurtosis of the peak_energy_wave_period : " ,kurtosis(waves['peak_wave_period']))


# In[ ]:


# Find the skewness and Kurtosis on Peak Direction Column
print("Skewness of the Peak Direction : " ,skew(waves['peak_direction']))
print("Kurtosis of the Peak Direction : " ,kurtosis(waves['peak_direction']))


# In[ ]:


# Find the skewness and Kurtosis on sea_surface_temperature Column
print("Skewness of the sea_surface_temperature : " ,skew(waves['sea_surface_temperature']))
print("Kurtosis of the sea_surface_temperature : " ,kurtosis(waves['sea_surface_temperature']))


# We have observed there is -Ve skewness for all objects and +Ve kurtosis for all objects

# # Graphical Visualizations

# # Univariate Analysis

# In[ ]:


# Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns


# # 1. Wave Height

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(waves['significant_wave_height'])
plt.show()


# In[ ]:


waves['significant_wave_height'].hist()


# In[ ]:


sns.boxplot(waves['significant_wave_height'])


# In[ ]:


plt.figure(figsize=(10,6)) 
plt.title("significant_wave_height") 
sns.kdeplot(data=waves['significant_wave_height'], label="significant_wave_height", shade=True)


# # 2. Maximum Wave Height

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(waves['maximum_wave_height'])
plt.show()


# In[ ]:


waves['maximum_wave_height'].hist()


# In[ ]:


sns.boxplot(waves['maximum_wave_height'])


# In[ ]:


plt.figure(figsize=(10,6)) 
plt.title("maximum_wave_height") 
sns.kdeplot(data=waves['maximum_wave_height'], label="maximum_wave_height", shade=True)


# # 3. Zero Wave Period

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(waves['zero_wave_period'])
plt.show()


# In[ ]:


waves['zero_wave_period'].hist()


# In[ ]:


sns.boxplot(waves['zero_wave_period'])


# In[ ]:


plt.figure(figsize=(10,6)) 
plt.title("zero_wave_period") 
sns.kdeplot(data=waves['zero_wave_period'], label="zero_wave_period", shade=True)


# # 4. Peak Wave Period

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(waves['peak_wave_period'])
plt.show()


# In[ ]:


waves['peak_wave_period'].hist()


# In[ ]:


sns.boxplot(waves['peak_wave_period'])


# In[ ]:


plt.figure(figsize=(10,6)) 
plt.title("peak_wave_period") 
sns.kdeplot(data=waves['peak_wave_period'], label="peak_wave_period", shade=True)


# # 5. Wave Direction

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(waves['peak_direction'])
plt.show()


# In[ ]:


waves['peak_direction'].hist()


# In[ ]:


sns.boxplot(waves['peak_direction'])


# In[ ]:


plt.figure(figsize=(10,6)) 
plt.title("Peak Direction") 
sns.kdeplot(data=waves['peak_direction'], label="Peak Direction", shade=True)


# # 6. Sea Surface Temperature

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(waves['sea_surface_temperature'])
plt.show()


# In[ ]:


waves['sea_surface_temperature'].hist()


# In[ ]:


sns.boxplot(waves['sea_surface_temperature'])


# In[ ]:


plt.figure(figsize=(10,6)) 
plt.title("sea_surface_temperature") 
sns.kdeplot(data=waves['sea_surface_temperature'], label="sea_surface_temperature", shade=True)


# # Normalize the Data

# In[ ]:


# import libraries
from sklearn.preprocessing import scale
scale(waves1)


# There is negative values in the data then we can use Exponentials of the data

# In[ ]:


np.exp(scale(waves1))


# # Regression Model

# # 1. Pair Plot

# In[ ]:


sns.pairplot(waves)


# # 2. Correlation

# In[ ]:


plt.figure(figsize=(14,10))
sns.heatmap(waves.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()


# # Correlation Matrix

# In[ ]:


waves.corr()


# In[ ]:


# Prepare Regression Model using all Objects
# import libraries 
import statsmodels.formula.api as smf


# In[ ]:


# Preparing model                  
Regression = smf.ols('significant_wave_height~maximum_wave_height+zero_wave_period+peak_wave_period+sea_surface_temperature+peak_direction',data=waves).fit() # regression model


# In[ ]:


# Getting coefficients of variables               
Regression.params


# In[ ]:


# Summary
Regression.summary()


# In[ ]:


# Import Libraries
from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm


# In[ ]:


x = waves1.drop(['significant_wave_height'], axis = 1)
y = waves1.significant_wave_height.values


# In[ ]:


# Fit the Model
regr = linear_model.LinearRegression()
regr.fit(x, y)


# In[ ]:


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


# Build Model
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 


# In[ ]:


print_model = model.summary()
print(print_model)


# In[ ]:




