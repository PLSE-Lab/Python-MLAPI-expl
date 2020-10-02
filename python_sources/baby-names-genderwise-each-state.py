#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_cell_magic('sh', '', '# location of data files\nls /kaggle/input')


# In[ ]:


# imports
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore", message="axes.color_cycle is deprecated")
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sqlite3


# In[ ]:


# explore sqlite contents
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# In[ ]:


# helper method to load the data
def load(what='NationalNames'):
    assert what in ('NationalNames', 'StateNames')
    cols = ['Name', 'Year', 'Gender', 'Count']
    if what == 'StateNames':
        cols.append('State')
    df = pd.read_sql_query("SELECT {} from {}".format(','.join(cols), what),
                           con)
    return df


# State-by-state data

# In[ ]:


df2 = load(what='StateNames')
df2.head(5)


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Kerry" and State=="TX"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Kerry" and State=="CA"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Kerry" and State=="NY"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Kerry" and State=="AZ"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Robbie" and State=="TX"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Robbie" and State=="CA"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Robbie" and State=="NY"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:


# Male and female babies with given name. 
tmp = df2.query('Name=="Jackie" and State=="TX"')[['Gender','Year', 'Count']].groupby(['Gender', 'Year']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'") 
        .set_index("Year").sort_index()
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'") 
          .set_index("Year").sort_index()
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join.plot()

# This result is good, the tails are much smaller.


# In[ ]:




