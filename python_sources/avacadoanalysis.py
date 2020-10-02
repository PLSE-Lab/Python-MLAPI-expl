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


# In[ ]:


import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


os.getcwd()


# In[ ]:


import pandas as pd


# In[ ]:


avo = pd.read_csv('../input/avocado.csv')
sns.pairplot(avo, x_vars='AveragePrice',y_vars='Total Volume')
#Pair plot is used to understand 
#or to explain a relationship between two variables 
#sns.pairplot(avo)


# 

# In[ ]:


avo.head()


# In[ ]:


#How many rows and columns are there
avo.shape


# In[ ]:


#What is the average Purchase Price of Avocado in last 4 years
avo['AveragePrice'].mean()


# In[ ]:


#What is the minimum Price of Avocado in last 4 years
avo['AveragePrice'].min()


# In[ ]:


#What is the maximum Price of Avocado in last 4 years
avo['AveragePrice'].max()


# In[ ]:


#Find maximum no of Avacados are sold in last 4 years and which year it is?
avo['Total Volume'].max()


# In[ ]:


#How many people made the purchase of Organic and Conventional in last 4 years
avo['type'].value_counts()


# In[ ]:


#Is there a correlation between Total bags and Total volume
avo[['Total Volume','Total Bags']].corr()


# In[ ]:


sns.heatmap(avo[['Total Volume','Total Bags']].corr())


# In[ ]:


#'What was the average volume of Avocado per year? (2015-2018) ?')
avo.groupby('year')['Total Volume'].mean()


# In[ ]:


#Find Avg Price of Avocado in the different region in last 4 years
avo.groupby('region')['AveragePrice'].mean()


# In[ ]:


#mean of Average Price in last 4 years
avo['AveragePrice'].mean()


# In[ ]:


#Which year Avacodo costed more than average in among the diff PLU(Price look-up codes)

avo[avo['AveragePrice'] > avo['AveragePrice'].mean()][['AveragePrice','4046','4225','4770']].sort_values(by = ['AveragePrice'])


# In[ ]:


#Find Average price of Organic Avocado in last years of which regions?
#Subset of data from complete dataset 
organic = avo[avo['type'] == 'organic']
organic


# In[ ]:


#Find the average Organic Avacado cost in last 4 years
organic.groupby('year')['AveragePrice'].mean()


# In[ ]:


#Find the list of average Organic Avacado cost in different regions in last 4 years
organic.groupby('region')['AveragePrice'].mean().sort_values(ascending=False)


# In[ ]:


#Find sales by regions and later build it by year
avo.groupby('region')['Total Volume'].mean().sort_values()


# In[ ]:


#Find each year in last 4 years how many types of avacado got sold.
avo.groupby('year')['type'].value_counts()


# In[ ]:


#Find the total no of sales in each year of last 4 years
avo.groupby('year')['Total Volume'].count()


# In[ ]:


#Data avialable for each year..in the dataset for last 4 years(size of the dataframe)
avo.groupby('year')['Total Volume'].size()


# In[ ]:


#Correlation:Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together. 
avo[['4046','Small Bags','Total Volume','Total Bags']].corr()


# In[ ]:


sns.heatmap(avo[['4046','Small Bags','Total Volume','Total Bags']].corr())


# In[ ]:


sns.heatmap(avo[['4225','Large Bags','Total Volume','Total Bags']].corr())


# In[ ]:


sns.heatmap(avo[['4770','XLarge Bags','Total Volume','Total Bags']].corr())


# In[ ]:


sns.heatmap(avo[['4046','4770','4225','Small Bags','Large Bags','XLarge Bags','Total Volume','Total Bags']].corr())


# In[ ]:


#Average Sales of Avacados by month 
avo['Month'] = avo['Date'].apply(lambda date:pd.Period(date, freq='M'))


# In[ ]:


avo.head(2)


# In[ ]:


avg_monthly_sales = avo.groupby(avo['Month'])['Total Volume'].mean()
avg_monthly_sales


# In[ ]:


sns.distplot(avg_monthly_sales,bins=10, kde=False)


# In[ ]:


#Average Sales of Avacados by Quater of each year in last 4 years
avo['Quater'] = avo['Date'].apply(lambda date:pd.Period(date, freq='Q'))


# In[ ]:


avo.head(6)


# In[ ]:


avg_Q_sales = avo.groupby(avo['Quater'])['Total Volume'].mean()


# In[ ]:


avg_Q_sales


# In[ ]:


#Distribution plot showing the Average price in four years(Avg Price of Avacado ranged between $1.0 and $1.5)
sns.distplot(avo['AveragePrice'],bins=10, kde=False)


# In[ ]:


#Avg Price of Avacados greater than the Average Price of Avacados in last 4 years
# Looks like 2016 and 2017 first quater the the price was greater than average
sns.jointplot(x='AveragePrice', y='year', data=avo[avo['AveragePrice'] > avo['AveragePrice'].mean()], kind='hex', 
              gridsize=20)


# In[ ]:


#Average sales of PLU 4225 by region.
#avo.groupby('region')['4225'].mean()
sns.distplot(avo.groupby('region')['4225'].mean(),bins=10, kde=False)


# In[ ]:


#Average sales of PLU 4770 by region.
avo.groupby('region')['4770'].mean()
sns.distplot(avo.groupby('region')['4770'].mean(),bins=10, kde=False)


# In[ ]:


#Average sales of PLU 4026 by region.
#avo.groupby('region')['4046'].mean()
## Looks like PLU 4046 and PLU 4225 are preffered same across All S regions.
sns.distplot(avo.groupby('region')['4046'].mean(),bins=10, kde=False)


# In[ ]:





# In[ ]:





# In[ ]:




