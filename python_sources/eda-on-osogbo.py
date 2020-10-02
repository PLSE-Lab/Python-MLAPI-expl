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


# for Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore Unnessary warnings
import warnings
warnings.filterwarnings('ignore')
# set the style to use for plotting
plt.style.use('ggplot')


# In[ ]:


# import diffrent algorithms. This will be use to establish a baseline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor


# In[ ]:


df = pd.read_csv('/kaggle/input/aiosogbo-certification-competition/Train.csv')


# In[ ]:


# check the first 5 rows
df.head()


# In[ ]:


# checking the features (columns) names of the data
df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# we can observe that from the result of the info(), there are missing values in 'Item_Weight' and 'Outlet_Size'
# to get a count of what missing we will use isnull()

# In[ ]:


df.isnull().sum()


# SO there are 1463 missing values in Item_Weight and 2410 missing values in Outlet_size

# Explore data Visually to understand the nature of data in terms of distribution of the individual features, finding missing values, relationship with other variables and many others.
# *UNIVARIATE* Exploratory Data Analysis (EDA) involves exploring features individually. continuous variables should be done using histogram and or scatter plots. Box plots too can help to visualize and understand outliers in the data. Outliers are data points that are either too low or too high as compared to the other values in the dataset. It is also adviced to use bar plots to explore the categorical featurs present in the data
# 

# before we start analysis we could split our data into integer type and floats and categorical so we can better analyze each one thoroughly 

# In[ ]:


numeric_features = df.select_dtypes(include=['int64', 'float64'])
categorical_features = df.select_dtypes(include='object')
print('Numeric Columns are {}'.format(numeric_features.columns))
print('-----------'*10)
print('Categorical Columns are {}'.format(categorical_features.columns))


# In[ ]:


# Our target is continous, so we can get the distribution using a histogram

plt.figure
plt.hist(df['Item_Outlet_Sales'], bins=50)
plt.xlabel('Item_Outlet_Sales')
plt.ylabel('count')
plt.title('Histogram of Item_Outlet_Sales')
plt.show()


# the plot shows the feature is right skewd and would need some data transformation to treat its skewness
# 
# **Keep in mind that item_Outlet_Sales** is the target. so take notes of all you do

# In[ ]:


# independent Variables (Numeric variables)
# histogram helps us to visualize the distribution of the variable
Item_Weight = df['Item_Weight']
fig, ax = plt.subplots()
ax.hist(Item_Weight.dropna(), color='blue', bins=50, alpha=0.9)
plt.xlabel('Item_Weight')
plt.ylabel('count')
plt.title('Histogram of Item_Weight')


# There seems to be no clear-cut pattern in Item_Weight
# We can see that it is normally distributed across the dataset

# In[ ]:


Item_Visibility = df['Item_Visibility']
fig, ax = plt.subplots()
ax.hist(Item_Visibility.dropna(), color='green', bins=80, alpha=0.9)
plt.xlabel('Item_Visibility')
plt.ylabel('count')
plt.title('Histogram of Item_Visibility')


# Item_Visibility is right skewed and should be transformed to fix its skewness

# In[ ]:


Item_MRP = df['Item_MRP']
fig, ax = plt.subplots()
ax.hist(Item_MRP.dropna(), color='red', bins=90, alpha=0.9)
plt.xlabel('Item_MRP')
plt.ylabel('count')
plt.title('Histogram of Item_MRP')


# We can clearly see 4 different distributions for Item_Mrp. it is an intresting insight

# **Gaining** insights from categorical variable which can only have a finite set of values
# # lets plot the Item_Fat_Content

# In[ ]:


df['Item_Fat_Content'].value_counts().plot(kind='bar')

plt.xlabel('Item_Fat_Content')
plt.ylabel('count')
plt.title('Histogram of Item_Fat_Content')


# in the figure LF, Low Fat and low fat are same category and Regular and reg are also the same we can combine this and plot again

# In[ ]:


df.Item_Fat_Content[df['Item_Fat_Content'] == 'LF'] = 'Low Fat'
df.Item_Fat_Content[df['Item_Fat_Content'] == 'low fat'] = 'Low Fat'

# for regular
df.Item_Fat_Content[df['Item_Fat_Content'] == 'reg'] = 'Regular'

# plot again

df['Item_Fat_Content'].value_counts().plot(kind='bar')

plt.xlabel('Item_Fat_Content')
plt.ylabel('count')
plt.title('Bar Chart of Item_Fat_Content')


# In[ ]:


# plot for Item_Type

df['Item_Type'].value_counts().plot(kind='bar')

plt.xlabel('Item_Fat_Content')
plt.ylabel('count')
plt.title('Bar Chart of Item_Fat_Content')


# from the chart above, Fruits and Vegetables, as the highest count
# 

# In[ ]:


# plot for Outlet_Identifier

df['Outlet_Identifier'].value_counts().plot(kind='bar')


# In[ ]:


# plot for Outlet_Size

df['Outlet_Size'].value_counts().plot(kind='bar')


# In[ ]:


# plot for Establishment_Year
df.Outlet_Establishment_Year.value_counts().plot(kind = 'bar')


# In[ ]:


# plot for Outlet_Type
df['Outlet_Type'].value_counts().plot(kind='bar')


# # Bivariate Analysis

# ### in Bivariate Analysis we will explore the independent variables with respect to the target variable.
# ### this hepls to discover hidden patterns between independent and target variable.
# ### we can then use those findings to deal with missing data imputation and feature engineering
# 
# ### scatter plots is advisable for the continuous or numeric features while violin plots for categorical
# 
# 

# In[ ]:


# Item_Weight vs Item_Outlet_Sales
plt.scatter(df['Item_Weight'], df['Item_Outlet_Sales'], c='violet', alpha=0.3, marker='.')
plt.xlabel('Item_Weight'), plt.ylabel('Item_Outlet_Sales'), plt.title('Item_Weight vs Item_Outlet_Sales')

######################## Item_Outlet_Sales is spread well across the entirerang


# In[ ]:


# Item_Visibility vs Item_Outlet_Sales
plt.scatter(df['Item_Visibility'], df['Item_Outlet_Sales'], c='violet', alpha=0.3, marker='.')
plt.xlabel('Item_Visibility'), plt.ylabel('Item_Outlet_Sales'), plt.title('Item_Visibility vs Item_Outlet_Sales')


# There is a string of point at 0.0 for Item_visibility which is not possible more into this soon.

# In[ ]:


# Item_MRP vs Item_Outlet_Sales
plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'], c='violet', alpha=0.3, marker='.')

plt.xlabel('Item_MRP'), plt.ylabel('Item_Outlet_Sales'), plt.title('Item_MRP vs Item_Outlet_Sales')


# # Visualize Categorical Variables
# 
# check the distribution of the target across all categorical. violin and boxplot would fit in perfectly, i used violin as it shows full distributionthe width of a violin at a particular level indicates the concentration or density of data target the height tells us about the range of the largest variable values

# In[ ]:


sns.violinplot(df['Item_Type'], df['Item_Outlet_Sales'])
plt.xticks(rotation=90)


# In[ ]:


sns.violinplot(df['Item_Fat_Content'], df['Item_Outlet_Sales'])
plt.xticks(rotation=90)


# In[ ]:


sns.violinplot(df['Outlet_Size'], df['Item_Outlet_Sales'])
plt.xticks(rotation=40)


# dealing with Missing Values

# In[ ]:


df.isnull().sum()


# In[ ]:


# fill int or float data type with median and fill categorical data type with mode.
# Note this is optional you can use what you see best
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())


# In[ ]:


# remember Item_Visibility as lots of 0 values which is not possible
plt.hist(df['Item_Visibility'], bins=70, color='grey')
plt.show()


# In[ ]:


# let replace the zeroes and plot again to see the changes
zero_index = df['Item_Visibility'] == 0

df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.median(df.Item_Visibility))
plt.hist(df['Item_Visibility'], bins=70, color='grey')
plt.show()


# End of Simple EDA hope you have learnt something if yes please do upvote this kernal. Thank you ALL
