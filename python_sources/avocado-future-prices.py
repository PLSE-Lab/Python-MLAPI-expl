#!/usr/bin/env python
# coding: utf-8

# **Outline**
# * Problem Description
# * Import The Libraries
# * Understanding The Dataset
#     * Data Cleaning
# * EDA
# * Feature Engineering
# * Modelling

# # 1. Problem Description
# 
# Our general goal is to understand the avacado sales in the US; find useful insights from the dataset that can be used by the farmers/sellers and consumers. Then be quite specific and predict what will be the buying price/Retailers selling price of a single avocado given all the features and only the **month **for date.
# 
# I am quite interested with this study, not just because of my love for avocados but because avacado is a farm product. Farm products's prices are highly dependent on supply and demand forces. The supply fluctuates with the seasons; having very little or even no supply during low season and alot of produce during peak seasons. If this scenario is not well managed we see food going to waste during peak seasons. 
# 
# Using insights from data, proper management strategies can be developed to ensure that there is proper pricing, well distribution of produce from high peak season regions to low season regions, and food preservation of produce that the market will not consume despite the right price.
# 
# 
# We do acknowledge that food supply chain is extremely complex, and the delivery of a single type of food such as avocado to a consumer involves many actors. However, the fact is, the end consumer buying price and and the produce determines how the inbetween will be.
# 

# # 2. Import The Libraries
# Lets start off with the basic libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# # 3. Understanding The Dataset

# In[ ]:


# Upload and view the dataset
data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")
data.head(8)


# In[ ]:


data.shape


# **A brief description of the 14 columns:** 
# * Date - observation date, 
# * Average price - avg price per pc, 
# * Total Volume - Total pcs of avocados sold, 
# * 4046,4225, 4770 - Bulk produce codes/labels of total pcs sold, 
# * Total Bags - Total bags sold, 
# * Small Bags, Large Bags, XLarge Bags- 3 different size of bags sold, 
# * Type -  convention or Organic type of avocados of the observation
# * Year- Year of observation. from 2015 - 2018
# * Region - The city
# 

# In[ ]:


# All the regions (cities)
data.region.unique()


# In[ ]:


# The type of avocados
data.type.unique()


# In[ ]:


# What are the Data type and do we have some null
data.info()


# No missing data, most dtypes are float. .and 3 objects

# In[ ]:


# do we have any duplicates, that we may need to drop
data.duplicated().sum()


# Nope! We are good, no duplicates

# In[ ]:


data.describe()


# Highest price is 3.25 and lowest is 0.4 with no 0 price.

# # 4. Exploratary Data Analysis

# In[ ]:


# compare the prices of the two types and identify any outliers

#define the plot
f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x="type", y="AveragePrice",data=data,);
plt.title("Average Price Per Piece",fontsize = 25,color='black')
plt.xlabel('Type of Avocado',fontsize = 15,color='black')
plt.ylabel('Avg Price',fontsize = 15,color='black')


# The organic are generaly more expensive which is quite expected now that many people are going the organic way, hence there is a higher demand and I guess it must be more expensive to grow organic avocados. The organic seems to have outliers on both ends, which we can look into later. The conventional type has outliers on the higher side.

# In[ ]:


# compare the price over the four years and identify any outliers
f,ax = plt.subplots(figsize = (10,7))
sns.boxplot(x="year", y="AveragePrice",data=data,);
plt.title("Average Price Per Piece",fontsize = 25,color='black')
plt.xlabel('year',fontsize = 15,color='black')
plt.ylabel('Avg Price',fontsize = 15,color='black')


# Seems like the mean price was generally higher in 2017

# In[ ]:


# plot AveragePrice distribution
sns.distplot(data['AveragePrice']);


# In[ ]:


# what is the relationship between the price and the rest of the features
# Is there a correlation
# Perform correlation

corr = data.corr()
corr.sort_values(["AveragePrice"], ascending = False, inplace = True)
print(corr.AveragePrice)


# We start learning more about price, interesting to see that there is 'less' correlation with the total volume,however look further into maybe each year.

# # 5. Feature Engineering

# In[ ]:



data.shape


# In[ ]:


# convert Date from object to a datetime
import datetime as dt
data.Date= pd.to_datetime(data.Date) 
data.Date.dt.month
data["Month"]= data.Date.dt.month


# In[ ]:


data.info()


# In[ ]:


#drop unecessary features
data.drop(['Date'],axis =1,inplace=True)
data.drop(['Unnamed: 0'],axis =1,inplace=True)
data.drop(['year'],axis =1,inplace=True)


# In[ ]:


#get dummies for categorical data
data = pd.get_dummies(data)


# In[ ]:


# checking the shape at this point
data.shape


# # 6. Train Model
# **Given all the features can we predict price?**

# In[ ]:


#Define X and y value.
x=data.drop('AveragePrice',axis=1)
y=data ['AveragePrice']

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[ ]:


#Use numpy to convert to array
x_train = np.array(x_train)
y_train = np.array(y_train)


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train);


# In[ ]:


#Use numpy to convert to test train to an array
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[ ]:


# Use the forest's predict method on the test data
predictions = rf.predict(x_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# our MAE is 0.11 degrees

# In[ ]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Our model has a 92.19% accuracy in predicting the price of an avocado given its features

# In[ ]:




