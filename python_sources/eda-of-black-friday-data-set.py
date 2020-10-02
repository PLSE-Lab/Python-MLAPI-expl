#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
sns.set(style='whitegrid')
# Any results you write to the current directory are saved as output.


# ### Black Friday
# Black Friday is an informal name for the Friday following Thanksgiving Day in the United States, which is celebrated on the fourth Thursday of November. The day after Thanksgiving has been regarded as the beginning of America's Christmas shopping season.Many stores offer highly promoted sales on Black Friday and open very early, such as at midnight, or may even start their sales at some time on Thanksgiving. Black Friday is not an official holiday, but California and some other states observe "The Day After Thanksgiving" as a holiday for state government employees. 

# **Lets load the data**

# In[ ]:


dataFrame = pd.read_csv('../input/BlackFriday.csv')


# In[ ]:


#Let see the first five rows of the dataset
dataFrame.head()


# In[ ]:


#Lets see some statistics about the data
dataFrame.describe()


# In[ ]:


#Lets see present total Nans in each column
dataFrame.isnull().sum()


# In[ ]:


dataFrame.shape


# From the above two cells we can see that we have 537577 no. of rows or entries and product_category_2 and 3 has NaNs present. We will deal with NaNs latter, Lets visualize and try to get some good insights about the data.

# **Visualizing Occupation**

# In[ ]:


sns.countplot(x='Occupation', data=dataFrame)


# Here we can see that we have 20 different occupations. **x-axies is occupations and y-axis is no. of people belong to that occupations(thus count)**. From the plot we can see that occupation '8' has leat no. of users associated with it and occupation 4, 0, 7 are Top 3.

# In[ ]:


#Lets see what are the city category we have here
dataFrame.City_Category.value_counts()


# so, we have 3 categories of city namely A, B, C and B, C, A is the Descending order of the people live in that city categoty. 

# ### Occupation in City_category
# Lets see the distribution of each occupation in each city

# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
hue_order = ['A', 'B', 'C']
sns.countplot(x='Occupation', hue='City_Category', hue_order=hue_order, data=dataFrame, palette='BuGn')


# In[ ]:


#Let see what are the age groups we have here
dataFrame.Age.value_counts()


# so, we have 7 different age groups here.

# In[ ]:


#Lets visualize age-group distribution 
age_order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
sns.countplot(x='Age', data=dataFrame, order=age_order)


# **Lets see the distribution of Gender in age_group**

# In[ ]:


sns.countplot(x='Age', hue='Gender', data=dataFrame, order=age_order, palette=sns.cubehelix_palette(8))


# **We can see that in every age group male users are dominating**

# ### Age-group vs Product_Category_x
# Lets see how each age-group has purchased each of the product category  

# In[ ]:


#creating a separate dataframe with two columns namely age and product_category_1 from original dataframe
data_age_prod1 = pd.concat([dataFrame['Age'], dataFrame['Product_Category_1']], axis=1)
#mapping each age group to integer value
data_age_prod1['Age'] = data_age_prod1['Age'].map({'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6})


# In[ ]:


data_age_prod1.head()


# In[ ]:


#age vs product_category_1
plt.subplots(figsize=(10,7))
sns.boxplot(x='Age', y='Product_Category_1', data=data_age_prod1)


# In[ ]:


#Filling the NaNs, here NaNs means users haven't purchased that product 
dataFrame.fillna(value=0, inplace=True)


# In[ ]:


dataFrame['Product_Category_2'].unique()


# In[ ]:


dataFrame['Product_Category_2'] = dataFrame['Product_Category_2'].astype(int)
dataFrame['Product_Category_3'] = dataFrame['Product_Category_3'].astype(int)


# In[ ]:


plt.subplots(figsize=(20,10))
sns.boxplot(x='Age', y='Product_Category_2', hue='Gender', order=age_order, data=dataFrame)


# The above plot describes distribution of purchase of product_category_2 by each age group and separated by gender.

# In[ ]:


plt.subplots(figsize=(20,10))
sns.boxplot(x='Age', y='Product_Category_1', hue='Gender', order=age_order, data=dataFrame)


# The above plot describes distribution of purchase of product_category_1 by each age group and separated by gender.

# In[ ]:


plt.subplots(figsize=(20,10))
sns.boxplot(x='Age', y='Product_Category_3', hue='Gender', order=age_order, data=dataFrame)


# The above plot describes distribution of purchase of product_category_3 by each age group and separated by gender.

# ### City type vs Purchase
# Lets see how each city purchased on blackfriday

# In[ ]:


city_order = ['A', 'B', 'C']
sns.barplot(x='City_Category', y='Purchase', order=city_order, data=dataFrame)


# The difference between by users from each city is less. City category B has heights no. of users followed by C, followed by A. Despite having most of the occupation users from B spends less than C.

# **Lets see how gender played a role in purchasing**

# In[ ]:


sns.countplot(dataFrame['Gender'])


# Male users are highly dominating in this case. 

# In[ ]:


sns.barplot(x='Gender', y='Purchase', data=dataFrame)


# In[ ]:


sns.boxplot(x='Gender', y='Purchase', data=dataFrame)


# This plot shows gender wise distribution of purchase. 

# **Lets see how marital status played in purchase**

# In[ ]:


sns.barplot(x='Marital_Status', y='Purchase', data=dataFrame)


# In[ ]:


sns.boxplot(x='Marital_Status', y='Purchase', data=dataFrame)


# **Users' tendency of purchase isn't effected by their marital status**

# **Lets find out total purchase of each occupation**

# In[ ]:


#Creating a new dataframe by concating 'Occupation' and 'Purchase' from original dataframe
df_occu_purchase = pd.concat([dataFrame['Occupation'], dataFrame['Purchase']], axis=1)


# In[ ]:


df_occu_purchase.head()


# In[ ]:


#Here we are creating another dataframe from df_occu_purchase using groupby occupation and then taking the sum
df2 = pd.DataFrame(df_occu_purchase.groupby('Occupation').sum().reset_index())


# In[ ]:


df2.head()


# In[ ]:


sns.set(style = 'white')
red = sns.color_palette('Reds')[-2]
sns.jointplot(x='Occupation', y='Purchase', data=df2, kind='kde', space=0, height=7, cmap='Reds')


# This plot shows the distribution of total purchase over occupation

# In[ ]:


#we can drop User_ID and Product_ID as these are not needed further
dataFrame.drop(columns=['User_ID', 'Product_ID'], inplace=True)


# In[ ]:


#Now, we intend to see correlation matrix, for this mapping object type value to integers 
dataFrame['Age'] = dataFrame['Age'].map({'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6})
dataFrame['City_Category'] = dataFrame['City_Category'].map({'A':0, 'B':1, 'C':2})
dataFrame['Gender'] = dataFrame["Gender"].map({'F':0, 'M':1})


# In[ ]:


dataFrame.head()


# In[ ]:


corr_mat = dataFrame.corr()
f, ax = plt.subplots(figsize=(9,5))
sns.heatmap(corr_mat, annot=True, ax=ax)


# **As shops offer highly discounted sale and people consider this as their prime time for winter shopping, people spends heavily.** 
