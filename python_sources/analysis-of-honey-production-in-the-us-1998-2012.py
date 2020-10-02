#!/usr/bin/env python
# coding: utf-8

# **Introduction**:
# According to https://www.statista.com/statistics/191996/top-10-honey-producing-us-states/ the US is one of the top 5 honey producing nations. The top 5 states for honey production are N. Dakota, S. Dakota, California, Montana and Florida, this data is for 2017. The US is also amongst the world's top four honey consuming nations along with Austria, Germany and Switzerland. The US does import a lot of honey but it also exports honey to countries like Japan, Yemen and Canada. For a decade or more bee keepers have been worried by declining yield, a google search for something like 'decline in honey production' shows concern from many countries in the industrialised world. The exact cause or causes are still debated but may include the use of pesticides by farmers, more intensive farming, parasites and climate change. In this kernel I will use the data to get a better picture of how overall honey production in the US has changed between 1998 and 2012 also how yield per colony has changed by state over the time period and finally which states have seen a decline in yield and if any have seen an increase.

# **Section 1: import data**

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/honeyproduction.csv')
df.head()


# **Section 2: Plotting total production against total colonies:**
# 
# Is there a simple relationship between the number of colonies and total production?

# In[3]:


df_copy = df.copy()
df_copy['total'] = df_copy['totalprod'].groupby(df_copy['year']).transform('sum')
df_copy['total_col'] = df_copy['numcol'].groupby(df_copy['year']).transform('sum')
drop_cols=['state','numcol','yieldpercol','totalprod','stocks','priceperlb','prodvalue']
df_by_year = df_copy.drop(drop_cols,1)
df_by_year = df_by_year.drop_duplicates(keep='first')


# In[4]:


sns.regplot(df_by_year['total_col'],df_by_year['total'],scatter_kws={"color": "orange"}, line_kws={"color": "black"})


# The above graph shows that as the total number of colonies increases the total production increases. The next step is to look at production and the number of colonies over the time period.

# In[5]:


df_by_year.plot(x='year',y='total',color='orange',title='total production by year')
df_by_year.plot(x='year',y='total_col',color='orange', title='total colonies by year')


# Total nuber of colonies shows a sharp recovery from 2008 but overall production failed to show the same level of improvement. Unfortunately the data only goes to 2012, it would be interesting to see if production did improve as the number of colonies recovered.
# 
# **Section 3: Visualise the production by state and time**
# 
# Over time, which states produce the most honey?
# 
# The next step is to visualise the yeild per colony by state in 1998 and again in 2012. I want to see which states had the most productive colonies and if there have been any changes in this over the 14 years. Some states are missing from the data set and not all states have the full 14 years of data.

# In[7]:


df_1998 = df[df.year==1998]
print('number of states in dataset for 1998: ' + str(len(df_1998)))


# In[8]:


df_2012 = df[df.year==2012]
print('number of states in dataset for 2012: ' + str(len(df_2012)))


# In[9]:


import plotly.offline as py
py.init_notebook_mode(connected=True)

scl = [[0.0, 'rgb(255, 204, 128)'],[0.2, 'rgb(255, 184, 77)'],[0.4, 'rgb(255, 153, 0)'],            [0.6, 'rgb(255, 153, 0)'],[0.8, 'rgb(204, 122, 0)'],[1.0, 'rgb(128, 77, 0)']]

labels = df_1998['state']
values = df_1998['yieldpercol']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "yield per colony (in pounds)")
        ) ]

layout = dict(
        title = 'honey yield per colony by state in the US in 1998',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_honey_yield_per_bee_colony_1998.html' )


# In[12]:


py.init_notebook_mode(connected=True)

scl = [[0.0, 'rgb(255, 204, 128)'],[0.2, 'rgb(255, 184, 77)'],[0.4, 'rgb(255, 153, 0)'],            [0.6, 'rgb(255, 153, 0)'],[0.8, 'rgb(204, 122, 0)'],[1.0, 'rgb(128, 77, 0)']]

labels = df_2012['state']
values = df_2012['yieldpercol']


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "yield per colony (in pounds)")
        ) ]

layout = dict(
        title = 'honey yield per colony by state in the US in 2012',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='US_honey_yield_per_bee_colony_2012.html' )


# Yield per colony is down in CA and other states such as North and South Dakota but is up in MS. So it is not a simple picture of declining production across all states. Remember this is not total production, MS is still relatively small in terms of honey production but in terms of yield per colony the story in MS seems to be a positive one over the time period represented by the data.
# 
# **Section 4: Investigate honey production in CA and MS**

# In[14]:


df_ca = df[df['state']=='CA']
df_ms = df[df['state']=='MS']


# **Changes over time in yield per colony in CA**

# In[16]:


sns.regplot(df_ca['year'],df_ca['yieldpercol'],scatter_kws={"color": "orange"}, line_kws={"color": "black"}).set_title('California')


# **Changes over time in yield per colony in MS**

# In[17]:


sns.regplot(df_ms['year'],df_ms['yieldpercol'],scatter_kws={"color": "orange"}, line_kws={"color": "black"}).set_title('Mississippi')


# **Section 5: Changes in the price of honey over the period 1998 - 2012**

# In[18]:


print(df_1998.priceperlb.mean())


# In[19]:


print(df_2012.priceperlb.mean())


# The average price increased by a factor of almost 3.
