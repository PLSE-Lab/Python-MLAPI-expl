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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Read from CSV file

# In[ ]:


train_data = pd.read_csv("../input/train.csv")


# In[ ]:


train_data.head


# # Describe dataframe

# In[ ]:


train_data.describe()


# # Display first five row of dataframe

# In[ ]:


train_data.head()


# # Display last five row of dataframe

# In[ ]:


train_data.tail()


# # Some operation with dataframe 

# #### Install seaborn library for data visualization

# In[ ]:


import seaborn as sns


# #### Get sales price zone wise 

# In[ ]:


train_data.groupby(['MSZoning']).mean()['SalePrice'].reset_index()


# In[ ]:


zone_sale_data = train_data.groupby(['MSZoning']).mean()['SalePrice'].reset_index()


# #### Data visualization - Bar Chart

# In[ ]:


sns.barplot(x='MSZoning', y='SalePrice', hue='MSZoning', data=zone_sale_data)


# #### Data visualization - Scatter Chart

# In[ ]:


sns.scatterplot(x='MSZoning', y='SalePrice', hue='SalePrice', data=zone_sale_data)


# #### Get sale price corresponding years

# In[ ]:


year_sale_price = train_data[['YearBuilt', 'SalePrice']]


# In[ ]:


year_sale_price.head()


# #### Data Visualization - Box Chart

# In[ ]:


sns.boxplot(x='SalePrice', y='YearBuilt', data=year_sale_price)


# In[ ]:


train_data.groupby(['YearBuilt']).mean()['SalePrice'].head().reset_index()


# #### Data Visualization - Scatter Chart

# In[ ]:


mean_year_sale = train_data.groupby(['YearBuilt']).mean()['SalePrice'].reset_index()


# In[ ]:


sns.scatterplot(x='YearBuilt', y='SalePrice', hue='YearBuilt', data=mean_year_sale)


# # Data Exploration

# #### Get average sale price based on HouseStyle and MSZoning

# In[ ]:


houseStyle_zone_sale = train_data.groupby(['HouseStyle', 'MSZoning']).mean()['SalePrice'].reset_index()
houseStyle_zone_sale


# In[ ]:


sns.barplot(x='HouseStyle', y='SalePrice', data=houseStyle_zone_sale)


# In[ ]:


sns.barplot(x='MSZoning', y='SalePrice', hue='HouseStyle', data=houseStyle_zone_sale)


# # Binning in Pandas
# 
# Binning :- To create a new meaning column based on existinf coloumn.
# 

# In[ ]:


def getNewHouseStyle(x) :
    return x + ' Apartment';

train_data['NewHouseStyle'] = train_data['HouseStyle'].map(lambda x : getNewHouseStyle(x))

train_data.head()


# In[ ]:


binning_home_style = train_data.groupby(['HouseStyle', 'NewHouseStyle']).mean()['SalePrice'].reset_index() 
binning_home_style


# In[ ]:


sns.barplot(x='HouseStyle', y='SalePrice', hue='NewHouseStyle', data=binning_home_style)

