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


# Data Acquisition, Data Wrangling, Data Formatting, Data Normalization, Binning  and Dummy Varibles
# The Data Set Used For this Exercise is the car price estimation dataset


# In[ ]:


#importing Liabraries in python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#DATA ACQUISITION


# In[ ]:


#importing files from computer for now
df_car = pd.read_csv("../input/car_pricing.csv", header = None)


# In[ ]:


df_car.head()


# In[ ]:


#Giving Header Names

header = ["symbolling","normalized-losses","make","aspiration","fuel-type","num-of-doors","body-style","drive-wheels",
         "engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders",
         "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg",
         "price"]


print(header)


# In[ ]:


df_car.columns = header


# In[ ]:


df_car.head()


# In[ ]:


#Data Preperation


# In[ ]:


#to know about the basics of data

df_car.describe()


# In[ ]:


# to know about the data types

df_car.dtypes


# In[ ]:


# Converting All "?" values to "NaN"

df_car.replace("?", np.nan, inplace = True)


# In[ ]:


df_car.head()


# In[ ]:


#To Check whether there are Missing Values

df_car.isnull().sum()


# In[ ]:


# How to Visualize that where are the missing values are seen in the data set
#For that it is neede to import a library called "missingno"

get_ipython().system('pip install missingno')
import missingno as msno


# In[ ]:


# to visualize the missing data
msno.matrix(df_car)


# In[ ]:


# The other method is using Heatmap
sns.heatmap(df_car.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# In[ ]:


# To get knowledge about the a single Attribute from the Entire DataFrame
# Method is Countplot
sns.set_style("whitegrid")
sns.countplot(x = "normalized-losses", data = df_car, palette = "RdBu_r")


# In[ ]:


df_car.isnull().sum()


# In[ ]:


#the missing Data are
# normalized-losses  - 41
# num-of-doors       - 2
# bore               - 4
# stroke             - 4
# horsepower         - 2
# peak-rpm           - 2
# price              - 4

# these are the attributes which are missing.


# In[ ]:


# Dealing with missing values is done by different methods

# 1. Replacing the Missing Values by the average of the Attributes


# In[ ]:


#taking average of the normalized-losses and replace the missing values with it

avg_norm_losses = df_car["normalized-losses"].astype("float").mean(axis = 0)
avg_norm_losses


# In[ ]:


# Replacing the missing value with the mean value

df_car.replace(np.nan, avg_norm_losses, inplace = True)


# In[ ]:


df_car.head()


# In[ ]:


#similarly change the values of all missing data by this method for the continuous values


# In[ ]:


avg_bore = df_car["bore"].astype("float").mean()


# In[ ]:


df_car["bore"].replace(np.nan, avg_bore, inplace = True)


# In[ ]:


#stroke
avg_stroke = df_car["stroke"].astype("float").mean()
df_car["stroke"].replace(np.nan, avg_stroke, inplace = True)


# In[ ]:


#horsepower
avg_horsepower = df_car["horsepower"].astype("float").mean()
df_car["horsepower"].replace(np.nan, avg_horsepower, inplace = True)


# In[ ]:


#horsepower
avg_peak_rpm = df_car["peak-rpm"].astype("float").mean(axis = 0)
df_car["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace = True)


# In[ ]:


#horsepower
avg_price = df_car["price"].astype("float").mean()
df_car["price"].replace(np.nan, avg_price, inplace = True)


# In[ ]:


msno.matrix(df_car)


# In[ ]:


df_car.isnull().sum()


# In[ ]:




