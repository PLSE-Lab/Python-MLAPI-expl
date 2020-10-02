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


data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()


# In[ ]:


import matplotlib as mlp
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# The following section is exploratory analysis on the price distribution of the top 5 neighbourhoods in each neighbourhood group

# In[ ]:


data['neighbourhood_group'].unique()


# In[ ]:


manhattan = data[data['neighbourhood_group'] == 'Manhattan']
brooklyn = data[data['neighbourhood_group'] == 'Brooklyn']
staten_island = data[data['neighbourhood_group'] == 'Staten Island']
queens = data[data['neighbourhood_group'] == 'Queens']
bronx = data[data['neighbourhood_group'] == 'Bronx']


# In[ ]:


manhattan['neighbourhood'].value_counts().head()


# In[ ]:


manhattan_5 = manhattan[manhattan['neighbourhood'].isin(['Harlem','Upper West Side',"Hell's Kitchen",'East Village','Upper East Side'])]


# In[ ]:


m_plot = sns.boxplot(data = manhattan_5, x = 'neighbourhood', y = 'price')
m_plot.set_title("Price distribution of the top 5 neighbourhoods in Manhattan")


# In[ ]:


brooklyn['neighbourhood'].value_counts().head()


# In[ ]:


brooklyn_5 = brooklyn[brooklyn['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Bushwick','Crown Heights','Greenpoint'])]
b_plot = sns.boxplot(data = brooklyn_5, x = 'neighbourhood', y = 'price')
b_plot.set_title("Price distribution of the top 5 neighbourhoods in Brooklyn")


# In[ ]:


staten_island['neighbourhood'].value_counts().head()


# In[ ]:


staten_5 = staten_island[staten_island['neighbourhood'].isin(['St. George', 'Tompkinsville','Stapleton','Concord','Arrochar'])]
b_plot = sns.boxplot(data = staten_5, x = 'neighbourhood', y = 'price')
b_plot.set_title('Price distribution of the top 5 neighbourhoods in Staten Island')


# In[ ]:


queens['neighbourhood'].value_counts().head()


# In[ ]:


queens_5 = queens[queens['neighbourhood'].isin(['Astoria','Long Island City','Flushing','Ridgewood','Sunnyside'])]
b_plot = sns.boxplot(data = queens_5, x = 'neighbourhood', y = 'price')
b_plot.set_title('Price distribution of the top 5 neighbourhoods in Queens')


# In[ ]:


bronx['neighbourhood'].value_counts().head()


# In[ ]:


bronx_5 = bronx[bronx['neighbourhood'].isin(['Kingsbridge','Fordham','Longwood','Mott Haven','Concourse'])]
b_plot = sns.boxplot(data = bronx_5, x = 'neighbourhood', y = 'price')
b_plot.set_title('Price distribution of the top 5 neighbourhoods in Bronx')


# These boxplots show us that Bronx has the cheapest rentals of the 5 neighbourhood groups, followed by Staten Island, with Manhattan far and away having the most expensive rentals. Let's double check to be sure.

# In[ ]:


data['price'].groupby(data['neighbourhood_group']).describe().round(2)


# There does not seem to be a linear relationship amongst most of the variables provided in the dataset. Further analysis would require non-linear, or polynomial methods.

# In[ ]:


sns.pairplot(data)


# In[ ]:


plt.figure(figsize = (10,10))

sns.distplot(manhattan['price'], color="skyblue", label="Manhattan")
sns.distplot(bronx['price'], color = 'pink', label = "Bronx")
sns.distplot(staten_island['price'], color = 'green', label = 'Staten Island')
sns.distplot(queens['price'], color = 'orange', label = 'Queens')
sns.distplot(brooklyn['price'], color = 'red', label = 'Brooklyn')

plt.legend()


# The price distribution of all 5 neighbourhood groups is skewed left, so it might be a good idea to conduct a log transformation on the price data to normalize it.
