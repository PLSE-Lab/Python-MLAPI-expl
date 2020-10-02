#!/usr/bin/env python
# coding: utf-8

# ## This is a dataset describing some important indicators of a country. We will try to understand this data and discover some interesting things. So let's dive into the analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns             # for visualisation
import matplotlib.pyplot as plt   # for visualisation

# for showing plot in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# loading and reading data


data = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')


# In[ ]:


# seeing the data

data.head()


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# ## Check for missing values

# In[ ]:



for col in data.columns:
    print(col  , (data[col].isnull().sum()/len(data[col])*100 ))
    


# ## As we can see that maximum percentage of missing values is 9.7 

# In[ ]:


data = data.fillna(0)


# In[ ]:


for col in data.columns:
    print(col  , (data[col].isnull().sum()/len(data[col])*100 ))


# ## As the names of some columns are not favourable with the typing purpose, so let's make them favourable

# In[ ]:


new_column_name = {'Area (sq. mi.)':'Area' , 'Pop. Density (per sq. mi.)':'Pop_density' , 
                  'Coastline (coast/area ratio)':'Coastline' , 
                  'Infant mortality (per 1000 births)':'Infant_mortality' , 'GDP ($ per capita)':'GDP_per_capita' ,
                  'Literacy (%)':'Literacy_percent' , 'Phones (per 1000)':'Phones_per_k' , 'Arable (%)':'Arable' ,
                   'Crops (%)':'Crops' ,'Other (%)':'Other'}
data = data.rename(columns = new_column_name )


# In[ ]:


data.head()


# ## Now we can see there are many numeric values in various columns which are not in the correct format. So let's make them correct.

# In[ ]:


def replace_commas(columns):
    for col in columns:
        data[col] = data[col].astype(str)
        dat = []
        for val in data[col]:
            val = val.replace(',' , '.')
            val = float(val)
            dat.append(val)

        data[col] = dat
    return(data.head())


# In[ ]:


columns = data[['Pop_density' , 'Coastline' , 'Net migration' , 'Infant_mortality' , 
                   'Literacy_percent' , 'Phones_per_k' , 'Arable' , 'Crops' , 'Other' , 'Birthrate' , 'Deathrate' , 'Agriculture' ,
                   'Industry' , 'Service']]
replace_commas(columns)


# ## Now let's see that which are the top 5 countries with high GDP per capita

# In[ ]:


data.sort_values(by = 'GDP_per_capita' , ascending = False).Country.iloc[0:5]


# ## Infant mortality rate ->It is the number of deaths per 1,000 live births of children under one ## year of age. 

# In[ ]:


sns.distplot(data.Infant_mortality)


# ## As we can see that most of the countries have IMR below 100

# ## Now let's see which countries have the highest IMR and which region they belong to?

# In[ ]:


data.sort_values(by = 'Infant_mortality' , ascending = False).Country.iloc[0:5]


# In[ ]:


data.sort_values(by = 'Infant_mortality' , ascending = False).Region.iloc[0:5]


# ## Top 4 countries with highest IMR belong to sub-saharan african region

# ## Now let's try to find the relationship between IMR and GDP per capita of that country

# In[ ]:


sns.lmplot(x = 'GDP_per_capita' , y = 'Infant_mortality' , data = data)


# ## As it can be inferred by looking at the scatterplot that there is a good enough relation between the IMR and per capita GDP of the country. And the relation is Negative.

# ## Let's try to understand this relation more with the help of hexplot

# In[ ]:


sns.jointplot(x = 'GDP_per_capita' , y = 'Infant_mortality' , kind = 'hex' , data = data)


# ## From the hexplot we can see that many lower IMR values corresponds to lower per capita GDP. Hence we can conclude that negative relation between the above two variables is not that strong.
