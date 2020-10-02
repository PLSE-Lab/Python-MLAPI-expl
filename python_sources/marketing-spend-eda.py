#!/usr/bin/env python
# coding: utf-8

# ### Hello! This is my first Kaggle contribution. This notebook contains a descriptive analysis using the Marketing Spend dataset.

# In[ ]:


# Libraries
import numpy as np
import pandas as pd 


# In[ ]:


# Importing data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/marketing-spend/marketing data.csv')


# In[ ]:


# First 5 rows
data.head(5)


# In[ ]:


# How many rows and columns does the DF have?
print("Rows:",data.shape[0], "\n","Columns:", data.shape[1], "\n", sep='')
# Variables name
print("Variables:", list(data.columns), "\n", sep='')
# Variable types
print("Variable types:", "\n",data.dtypes, sep='')


# ### Radio and Other media have 75% of the observations with null values. We will treat null values with 0.

# In[ ]:


print('% of null values per column:\n',100*(data.isna().sum()/data.shape[0]),sep='')
# Treating NaN
data.fillna(0, inplace=True)


# ### Our dataset has 12 registers of digital and offline media spend. Sponsorship has the highest average spend. Radio and Other media have the lowest spend.

# In[ ]:


# Main statistical measures
data.describe()


# ### We can see that most of the money were spend in 2015, in absolute and average values. 

# In[ ]:


print("Yearly spend: \n",data.groupby('Year')['Total Investment'].sum(), "\n", sep='')
print("Average monthly spend: \n", data.groupby('Year')['Total Investment'].mean(), sep='')


# In[ ]:


monthly_spend = data.groupby(['Year','Month'])['Total Investment'].sum()
print("Monthly spend: \n",monthly_spend, sep='')

monthly_perc_spend = (monthly_spend/monthly_spend.groupby(level=0).sum())*100
print("\nMonthly percentual spend: \n", monthly_perc_spend, sep='')


# In[ ]:


def calculate_percent(col,total='Total Investment'):
    return (col/data[total])*100

percent_of_total = data.iloc[:,3:].apply(calculate_percent, axis=0).fillna(0)
percent_of_total = pd.concat([data[['Year','Month']],percent_of_total], axis=1)
print("Percentage of total investment: \n", percent_of_total)


# ### Practicing some dataviz, we can see that Sponsorship spend has some spikes and two periods of a really low investment. The other channels have a more constant monthly investment.

# In[ ]:


data['Period'] = data["Month"].astype(str) + "/" + data["Year"].astype(str)
data.drop(columns=['Year','Month','Total Investment']).plot(x='Period', figsize=(13,8))


# ### Other channels, Radio and Content Marketing received a really low percentual investment along the reported period. 

# In[ ]:


percent_of_total['Period'] = percent_of_total["Month"].astype(str) + "/" + percent_of_total["Year"].astype(str)
percent_of_total.drop(columns=['Year','Month']).plot(x='Period', figsize=(13,8))

