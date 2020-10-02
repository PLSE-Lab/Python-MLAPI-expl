#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #a virtualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/BlackFriday.csv")


# In[ ]:


#top 5 rows from the dataset
data.head()


# In[ ]:


#after loading our data let's see some information about our data
data.info()


# 1. Above table tells us that we have 537577 rows of data. 
# 2. We have 12 columns in our data
# 3. We have 2 float variables, 5 integers, and 7 string
# 4. Size of the data is 49.2 MB or more

# In[ ]:


#check if dataset has any NaN value
data.isnull().sum()


# In[ ]:


#we can fill our NaN values.
data.fillna(data['Product_Category_1'].dropna().median(), inplace = True)
data.fillna(data['Product_Category_2'].dropna().median(), inplace = True)


# In[ ]:


data.isnull().sum()
#now our dataframe has 0 null values. 


# In[ ]:


#we can drop User_ID, Product_ID from the dataset. 
data = data.drop(["User_ID", "Product_ID"], axis = 1)


#  Now we check our correlation rate. Let's say the correlation between two features is 1. That means these features are direct proportion.
# 
# For example: 
# If the quantity of a room in a house increase, then the price of the house also increase.

# In[ ]:


f,ax = plt.subplots(figsize = (18, 18))
sns.heatmap(data.corr(), annot = True, linewidths = 1, fmt = ".1f", ax = ax)
plt.show()


# * There is a 0.3 correlation between Purchase and Product_Category_3 in black friday
# * Product_Category_1 and Product_Category_2 there is a negative correlation between Purchase. Which mean that in black friday, Product1 and Product 2's sales decrease.
# 
# Product_Category_1 gets discount in Black Friday.

# In[ ]:


#we can see our data's count, max, min values as well as lower, 50 and upper percentiles with the help of describe function
data.describe()


# In[ ]:


def bargraph(xvalue, yvalue, data):
    sns.barplot(x = xvalue, y = yvalue, data = data)


# In[ ]:


bargraph("Gender", "Purchase", data)
#males have bought more than female in black friday


# In[ ]:


bargraph("Marital_Status", "Purchase", data)
#married and single people have bought same quantity of items.


# In[ ]:


bargraph("City_Category", "Purchase", data)


# City C have purchased more item in Black Friday than other cities.

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.set(style="darkgrid")
sns.countplot(data['Age'],hue=data['Gender'])
plt.show()


# Above graph shows that female purchase less than male in black friday for all age interval. 

# In[ ]:


#remove + sign from Stay_In_Current_City colums
data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.str.replace('+', '')


# In[ ]:


#remove + sign from Age
data.Age = data.Age.str.replace('+', '')


# In[ ]:


#we can use & symbol for logical and operation
data[(data['Purchase'] > 20000) & (data['Gender'] == 'F')]


# In[ ]:


#or we can use logical_and from numpy library
data[np.logical_and(data['Purchase'] > 20000, data['Gender'] == 'M')]


# In[ ]:


data.loc[:50, ['Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Gender']]


# In[ ]:


gender_mapping = {'M' : 0, 'F' : 1}
data['Gender'] = data['Gender'].map(gender_mapping)
data.head(10)


# In[ ]:


#frequency of Product 1
print(data['Product_Category_1'].value_counts(dropna = False)) #if there are NaN values that also be counted


# In[ ]:


#frequency of Product 2
print(data['Product_Category_2'].value_counts(dropna = False))


# In[ ]:


#frequency of Product 3
print(data['Product_Category_3'].value_counts(dropna = False))


# An outlier is an observation that is numerically distant from the rest of the data. When reviewing a bloxplot, an outlier is defined as a data point that is located outside the fences of the boxplot. 

# In[ ]:


data.boxplot(column = 'Purchase', by='Product_Category_1')


# Those black dots are represent outliers.

# In[ ]:


data.boxplot(column = 'Purchase', by = 'Product_Category_2')


# In[ ]:


data.boxplot(column = 'Purchase', by = 'Product_Category_3')


# In[ ]:


#categorize city 
city_mapping = {'A': 0, 'B': 1, 'C': 2}
data['City_Category'] = data['City_Category'].map(city_mapping)
data.head(10)


# In[ ]:


#list comprehension
data['Age'] = [0 if i == "0-17" else 1 if i == "18-25" else 2 if i == "26-35" else 3 if i == "36-45" else 4 if i == "46-50" else 5 if i == "51-55" else 6 for i in data['Age']]
data.head(10)


# In[ ]:


#we can remove purchase from dataset because we have purchase_level
data = data.drop('Purchase', axis = 1)


# In[ ]:


data.head()


# In[ ]:


#we can tidy our dataframe with melt function.
melted = pd.melt(frame = data, id_vars = 'Purchase_level', value_vars = ["Product_Category_1", "Product_Category_2", "Product_Category_3"])
melted


# In[ ]:


#concatenate data
data_head = data.head() 
data_tail = data.tail()
data_concatenate = pd.concat([data_head, data_tail], axis = 1, ignore_index = False)
data_concatenate


# We got NaN values when we concatenate tables because of the indexing.
# 
# https://stackoverflow.com/questions/40339886/pandas-concat-generates-nan-values

# In[ ]:




