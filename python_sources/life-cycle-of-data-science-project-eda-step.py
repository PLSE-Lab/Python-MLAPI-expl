#!/usr/bin/env python
# coding: utf-8

# # COMPLETE LIFE-CYCLE OF A DATA SCIENCE PROJECT
# 
# **1. Exploratory Data Analysis
# **2. Feature Engg and Selection 
# **3. Best Model Selection using Sklearn, AUTOML TPOT, Tensorflow, and Keras
# **4. Hyper-paramter tuning and modelling 

# # Importing Necessary Libraries and Packages

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the datasets into memory

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# .head() function helps to print top 5 rows of the datasets 

# In[ ]:


train.head()


# In[ ]:


test.head()


# Test and Train dataset visual analysis shows that the target variable or parameter is 'Saleprice'! Thus, one is going to predict the **Selling price** of house under sale.

# Let's check out the dependent variable[SalePrice] and other independent variables!

# In[ ]:


train.columns


# train.info() code is highly useful to capture missing values, datatype of each and every columns.  

# In[ ]:


train.info()


# Inferences: PoolQC, Fence,MiscFeature,Alley, and FireplaceQu are few columns or features that are predominantly composed with missing values!

# Do visualize the dataset to explore insights from it!

# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


parameter_na=[i for i in train.columns if train[i].isnull().sum()>1]

for i in parameter_na:
    print(i, np.round(train[i].isnull().mean(),2),' % missing values')


# Accept the Challlenge! Lol!!
# Let's clean the data!! However one should not drop columns based on presence of missing values! Here come into play statistics in DS projects. But we have SimpleImputer as a weapon to combat with missing values!!

# In[ ]:


numerical_features = [feature for feature in train.columns if train[feature].dtypes != 'O']

print('Number of numerical datatype parameters:',len(numerical_features))


# In[ ]:


train[numerical_features].head()


# There are lot of temporal variables in this dataset. Temporal variables are those columns that comprises years/data/time related data. Let's check if these temporal variables contribute to our target variable or the parameter we are predict (SalePrice). If there is no good realtionship b/w these two paramters let's eliminate them and get rid from the 'CURSE OF DIMENSIONALITY'! 

# In[ ]:


year_feature = [f for f in numerical_features if 'Yr' in f or 'Year' in f]
year_feature


# In[ ]:


train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year the House was Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs Year House was Sold")


# NOTE:
# * The **Prices** of houses decreases as years increase i.e olden day houses costed more than 2010. The graph clearly states that 2006 and 2007 were the year of highest house sales and sellers would have had happy times, unlike owners sellers today!  
# * Also, the curve slants down linearly from the later half of 2007 and still decreasing!! Alert for Business officials, and Investors. Careful!!

# In[ ]:


# Let's check out other type sof variables like Numerical variables.
# As, all know it is further subdivided into two types: 1) Continous, and 2) Discrete variables

discrete_feature = [p for p in numerical_features if len(train[p].unique())<25 and p not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[ ]:


# Names of columns in train datsets with discrete variables
print(discrete_feature)


# In[ ]:


# Let's check out continuous varables:
cont_feature=[f for f in numerical_features if f not in discrete_feature+year_feature+['Id']]
print("Continuous feature paramters are Count {}".format(len(cont_feature)))


# In[ ]:


print(cont_feature)


# # Feature Engineering

# In[ ]:


corrmat = train.corr()
top=corrmat.index
plt.figure(figsize=(30,30))
graph = sns.heatmap(train[top].corr(),annot=True,cmap='RdYlGn')


# WOW! Huge and colorful right? But don'worry we are looking into the last row only as this shows how independent features correlate to our target feature 'SalesPrice'.
# NOTE: Don't eliminate columns as it shows negative correlation. Negative corrleation is nothing but inverse correlation that indicates that the two variables move in the opposite direction. It simply means that if feature X increases then feature Y decreases. If there is a good negative correlation ranging b/w -0.9 to -1.0, you are lucky. 

# Again, we can eliminate only Id column as it is unnecessary in this application!But the temporal variable spaly an important role as visualized in the above graph!

# Simple Imputer does not work when the dataset has categorical values i.e non-numerical values. eg..Male-Female in gender column that needs to be converted to 0s and 1s.  So, officially moving into Feature Engg step of Data Science!

# # **Do, UPVOTE this kernel if you LIKE!** 

# Check out my Second stage in Complete Data Science project: Feature engg and selection stage

# In[ ]:




