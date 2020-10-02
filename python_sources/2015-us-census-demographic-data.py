#!/usr/bin/env python
# coding: utf-8

# ### This is an assignment of BAx452 Machine Learning. 
# #### Assignment requirement:
# - Create a faceted plot in ggplot using size, shape and color as well as facets. 
# - Create a Correlation Heatmap in Seaborn . 
# - Create your own Test and Training sets.

# ##### *Table of Content*
# ### - Import Package and Load Dataset
# ### - County Count
# ### - Aggregate Data By State
# ### - Subset Data 
# ### - Facted Plot
# ### - Correlation Heatmap
# ### - Create Test and Training Datasets
# 

# ## Import Packages and Load Dataset

# In[ ]:


# import packages
import numpy as np
import pandas as pd
import ggplot
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load dataset
df = pd.read_csv('../input/acs2015_county_data.csv')
df.head()


# In[ ]:


# get column names
list(df)


# ## Let's see how many counties does each state have

# In[ ]:


# get unique values in "State" 
print(set(df['State']))


# In[ ]:


# county counts for each state
county = df[['State','County']].groupby('State').agg('count')
county.sort_values('County', ascending = 0)


# ## Aggregate Data By State

# In[ ]:


# group by state 
agg_df = df.groupby('State').sum()

# order by total population 
agg_df.sort_values('TotalPop', ascending = 0)


# ## Subset Data Using Data of 6 States

# In[ ]:


# Create a new dataset which contains data of states with top 5 total population
ca = df.groupby('State').get_group('California')
tx = df.groupby('State').get_group('Texas')
ny = df.groupby('State').get_group('New York')
fl = df.groupby('State').get_group('Florida')
il = df.groupby('State').get_group('Illinois')
pn = df.groupby('State').get_group('Pennsylvania')
df_6 = pd.concat([ca,tx,ny,fl,il,pn])
df_6.head()


# ## Faceted Plot

# In[ ]:


from ggplot import *

ggplot(df_6, aes(x='Unemployment', y='Poverty',color='Black')) +       geom_point(shape = 6) +       xlim(0, 18) +       xlab('Unemployment Rate') + ylab('Poverty Rate') + ggtitle('Poverty') +       facet_wrap("State")


# ## Correlation Heatmap

# In[ ]:


# Correlation heatmap
plt.subplots(figsize=(20,15))
ax = plt.axes()
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap = "Blues")


# ## Create Test and Training Datasets

# In[ ]:


#create test and trainning datasets
from sklearn.model_selection import train_test_split
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)


# 
