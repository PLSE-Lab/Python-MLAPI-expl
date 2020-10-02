#!/usr/bin/env python
# coding: utf-8

# In this notebook I have performed Exploratory Data Analysis on the Black Friday dataset and tried to identify relationship between a Purchase Price and various other features

# I hope you find this kernel helpful and some **<font color=red>UPVOTES</font>** would be very much appreciated

# **Importig required libraries**

# In[ ]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# **Loading the dataset**

# In[ ]:


df = pd.read_csv('../input/BlackFriday.csv')
df.head(4)


# In[ ]:


df.info()


# **Dimensions of the dataset**

# In[ ]:


print('Number of rows in dataset: ', df.shape[0])
print('Number of columns in dataset: ', df.shape[1])


# **Describing the dataset**

# Since the **'User_ID'** column is of no use in describing the dataset, I will remove it during describing

# In[ ]:


df.drop('User_ID',axis = 1).describe()


# **Total Number of Categorical Attributes**

# In[ ]:


print('No. of categorical attributes: ', df.select_dtypes(exclude = ['int64','float64']).columns.size)


# **Total Number of Numerical Attributes**

# In[ ]:


print('No. of numerical attributes: ', df.select_dtypes(exclude = ['object']).columns.size)


# **Checking for Null Values in the dataset**

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')
plt.title('Null Values present in the dataset',fontsize=14)
plt.show()


# Most of the values in column **Product_Category_2** and **Product_Category_3** are missing

# Filling in the Null(NaN) values

# In[ ]:


df['Product_Category_2'].fillna(0, inplace = True)
df['Product_Category_3'].fillna(0, inplace = True)


# **Histogram of features in the dataset **

# In[ ]:


sns.set_style('whitegrid')
df.drop('User_ID',axis=1).hist(figsize = (13,10), color = 'darkgreen')
plt.tight_layout()
plt.show()


# **Correlation matrix of features in the dataset**

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=1)
plt.show()


# **Distribution of amount of purchase**

# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.distplot(df['Purchase'],kde=False,bins = 30,color='green')
plt.show()


# ### **Plotting relationships between 'Purchase' and various other attributes in the dataset******

# **1. Purchase vs. Gender**

# In[ ]:


sns.boxplot(x='Gender',y='Purchase', data = df, width=0.4)
plt.show()


# In[ ]:


df.groupby('Gender').agg({'Purchase':['max','min','mean','median']}).round(3)


# It is seen that on an average Females spent more money than Men shopping on a Black Friday.

# **2. Purchase vs. Age**

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x = 'Age',y='Purchase', data = df,palette='hls',
            order=['0-17','18-25','26-35','36-45','46-50','51-55','55+'],width=0.5)
plt.show()


# In[ ]:


df.groupby('Age').agg({'Purchase':['min','max','mean']}).round(3)


# From the given data, people from all age groups except people in age range '51-55' spent almost the same average amount on shopping during a black friday. People in range '51-55' had a slightly higher average purchase amount than other age groups.

# **3. Purchase vs. Occupation**

# In[ ]:


plt.figure(figsize=(14,6))
sns.boxplot(x='Occupation',y='Purchase', data = df, width=0.6)
plt.show()


# **4.Purchase vs. City_Category**

# In[ ]:


plt.figure(figsize=(14,6))
sns.boxplot(x='City_Category',y='Purchase', data = df,
            width=0.4,palette='hls',order=['A','B','C'])
plt.show()


# **5.Purchase vs. Marital_Status**

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Marital_Status',y='Purchase', data = df,
            width=0.4,palette='GnBu')
plt.show()


# **6. Purchase vs. Stay_In_Current_City_Years**

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Stay_In_Current_City_Years',y='Purchase', data = df,
            width=0.4,palette='hls',order=['1','2','3','4+'])
plt.show()


# **7. Purchase vs. Marital_Status**

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Marital_Status',y='Purchase', data = df,
            width=0.4,palette='hls')
plt.show()


# In[ ]:


df.groupby('Marital_Status').agg({'Purchase':['min','max','mean']}).round(3)


# Both married as well as unmarried people spend about the same amount of money on shopping.

# In[ ]:




