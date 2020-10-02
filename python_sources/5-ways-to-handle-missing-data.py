#!/usr/bin/env python
# coding: utf-8

# <h1>Introduction</h1>

# In Data Science problems there are many instances where the data provided will have missing values. Reason for missing values can be anything like data curropt while processing, connection timeout/break during transfer of data or someone can willfully provide incomplete information(like in survey forms).
# 
# If missing data is not handled properly then it can cause incorrect results of our Data Science model resulting in inaccurate predictions.
# 
# In this notebook I have used House Prices training data set. You can access this data from kaggle from this [LINK](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
# 

# <h2>Handling Missing Data Approach</h2>

# According to me, there are two basic approach to handle missing data:
# 1. When missing data too large
# 2. When missing data is small
# 
# When missing data is too large, more then 80% or more than 90% (As per requirement) then it is advisable to drop entire columna nd not consider that particular column for further analysis. This is because the missing data is too large that predecting values can be tough and too risky. If values are predected wrong then it can also cause errors in our analysis.
# 
# When missing data is small we can do two things. First we can remove only those **rows** which have missing values, secondly we can try to **impute or replace those missing values** be certain analysis of already available values in our data set.
# 
# Let's import and look at our data.

# In[ ]:


# importing two basic libraries
import pandas as pd
import os


# In[ ]:


# Print list of available files
print(os.listdir("../input"))


# In this exercise we are going to use train.csv and try to handle missing values in this data set.

# In[ ]:


# Import train.csv file in df_train dataframe
df_train = pd.read_csv("../input/train.csv")

# Print all the columns available from file:
df_train.columns


# <h1>Checking Null values in our data set</h1>
# 
# Data is missing randomly so it becomes really important to visualize missing data in a tabular form. Here we will be presenting missing data in terms of "Total" missing values and "Percent" of missing values.

# In[ ]:


# How to calculate total missing values
total = df_train.isnull().sum().sort_values(ascending=False)

# Calculate percentage of missing values
percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending=False)

# Presenting in tabular form by concatinating both values and create a seperate data set
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

# Now check top results of missing_data
missing_data.head(20)


# <h1>Ways to handle missing data in a data set</h1>
# 
# There can be multiple ways to handle missing data and this notebook I have mentioned below 5 best methods. Any method can be choosen depending upon the application. you can also think of any other method also depending upon the project requirement you have.
# 
# <b>when missing data is small:</b>
# 1. Deleting Rows
# 2. Replacing with Mean/Median/Mode values
# 3. Predicting the missing value
# 4. Using algorithm which support missing values
# 
# <b>When missing data is too large:</b>
# 5. Dropping the entire column
# 
# Our primary data set will remain **df_train** and dataframe having missing data will be **missing_data**. For simplicity I will be creating a seperate dataframe.

# <h3>1. Deleting Rows</h3>
# This method should be used only when missing values are small. Applications where data is present in lakhs and crores, this method will only create positive impact in our further analysis.
# 
# For this step, let's create a different dataframe using **df_train** say **df_train_delrows**

# In[ ]:


# Copying data frame
df_train_delrows = df_train


# Since we are taking action based on percentage of missing data, we are going to use **missing_date** to filter df_train_delrows data.

# In[ ]:


# All rows with less than 5% and greater than 0% of missing data
missing_data[(missing_data['Percent']<5) & (missing_data['Percent']>0)]


# In[ ]:


# We can select rows by using below statement and then use it seperatly for each feature rows:
df_train_delrows[df_train_delrows['Electrical'].isnull()]


# In[ ]:


# Deleting rows based on our above statement:
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtExposure'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtFinType2'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtFinType1'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtCond'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtQual'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['MasVnrArea'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['MasVnrType'].isnull()]).index,0)
df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['Electrical'].isnull()]).index,0)


# In[ ]:


# Now verify if there is any missing data left for any of these columns
df_train_delrows['BsmtQual'].isnull().sum()
df_train_delrows['MasVnrArea'].isnull().sum()


# <b>Pros:</b>
# 1. Complete removal of missing rows helps develop highly accurate model
# 2. Deleting less number of rows doesn't impact our data set and can be used effectively
# 
# <b>Cons:</b>
# 1. Mojour loss will be there in case this method is used where data missing is too large(say 50%)

# <h3>2. Relacing with Mean/Median/Mode values</h3>
# The method is also used when missing data is small and we do not want to drop entire column. This is a statistical approach to replace misssing data and sometimes this approximation causes variance in our data set.
# 
# This method can only be applied where missing data is numeric like HousePrice or Airline Fares.

# In[ ]:


# Let's see missing_data dataset
missing_data.head(20)


# Say for our analysis we consider **'MasVnrArea' feature** and we will try to replace it's missing values.
# 
# Predefined methods for **Mean is df_train.mean()**, for **median is df_train.median** and for **mode is df_train.mode()**

# In[ ]:


# Loading a seperate dataset for mean
df_train_men = df_train


# In[ ]:


# Calculate Mean
df_train_men['MasVnrArea'].mean()


# In[ ]:


df_train_men[df_train_men['MasVnrArea'].isnull()]


# In[ ]:


# now we can fill NaN records with Mean values
df_train_men['MasVnrArea'].fillna(df_train_men['MasVnrArea'].mean(), inplace=True)


# In[ ]:


# Verify if NaN records still exists or not
df_train_men[df_train_men['MasVnrArea'].isnull()]


# Similarly, we can replace missing values with Mean/Midean/Mode.

# <b>Pros: </b>
# 
# 1. No rows are deleted using this method
# 2. God approach when missing data is small
# 
# <b>Pros: </b>
# 
# 1. Imputing values creates variance in data set and may cause incorrect analysis.

# <h3>3. Predicting the mising values</h3>
# 
# Using features which don't have missing values we can predict the null values. This can be done by choosing the best possible values by analyzing other available features.

# In[ ]:


# Load df_train in seperate dataframe
df_train_null = df_train


# In[ ]:


df_train_null[df_train_null['MasVnrType'].isnull()]


# Now We can predict values of 'MasVnrType' by analysing other features like SalePrice and Basement area. This depends entirely on what feature we choose based on requirement.
# 
# One Buyer gives priority to Basement are and other buyer giver priority to location. Values will change accordingly.

# <h3>4. Using algorithm which support missing values</h3>
# 
# KNN is a machine learning algorithm which works on the principle of distance measure. We can use this algorithm when there are null data present s the daa set. When the algorithm is applied, KNN considers the missing values by taking the majority of the K nearest values
# 
# Another algorithm which can be used is Random Forest. It adapts to the data structure taking into consideration of the high variance or the bias, producing better results on large datasets.

# <h3>5. Dropping the entire Columns</h3>
# 
# We choose to drop entire column only when missing values are too large(say more than 15%). It becomes almost impossible to replace these values by any other values because it will only create variation and incorrect analysis.
# 
# In our notebook we will use missing_data dataframe and see what are the large values which are missing.

# In[ ]:


# Percentage of missing values:
missing_data.head(20)


# Now consider we have decided to drop all those features for which values are missing more than 15%

# In[ ]:


# Dropping entire feature
df_train_null = df_train_null.drop((missing_data[missing_data['Percent']>15]).index,axis=1)


# In[ ]:


# Verify if columns exist or not
df_train_null.isnull().sum().sort_values(ascending=False).head(20)

