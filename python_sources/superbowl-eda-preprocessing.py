#!/usr/bin/env python
# coding: utf-8

# # Superbowl Ads Analysis (From 1967 to 2020)

# ![SuperBowl-Ad](https://www.koamnewsnow.com/content/uploads/2020/04/NFL-logo.jpg)

# ### Let the games begin!
# 
# > "It's not the SIZE of the dog in the fight, but the size of the FIGHT in the dog". - Archie Griffin

# In[ ]:


get_ipython().system('pip install chart-studio')

# Data processing libraries
import numpy as np 
import pandas as pd 

# Visualization libraries
import datetime
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Plotly visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
from plotly.graph_objs import *
from IPython.display import Image
pd.set_option('display.max_rows', None)

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Read data
temp = pd.read_csv('../input/superbowlads/superbowl-ads.csv', delimiter='^',quotechar='"')

# Line 561 , the 'Year' is missing
temp.iloc[561]='2020,'+temp.iloc[561]

# Replace quotes with spaces
new = temp.iloc[:,0].str.split(',', n = 3, expand = True) 
new[3]=new[3].str.replace('"','')
new[1]=new[1].str.replace('"','')
new[2]=new[2].str.strip('"')
new2=new[2].str.split('""', n = 1, expand = True) 
new = pd.concat([new.iloc[:,[0,1]],new2,new.iloc[:,3]],axis=1).values

# Build the final dataframe
ads_df=pd.DataFrame(new,columns=["Year","Type","Product","Title","Notes"])
ads_df=ads_df.loc[ads_df['Type']!='Product type']
del temp , new , new2
ads_df.head()


# ### Group the products based on types
# In order to scale the data, bring the data in log scale

# In[ ]:


records = ads_df.groupby(['Type']).size()
records = records.sort_values()

grouped_df = pd.DataFrame(records)

grouped_df['Count'] = pd.Series(records).values
grouped_df['Type'] = grouped_df.index
grouped_df['Log Count'] = np.log(grouped_df['Count'])
grouped_df.head()


# **Bar Chart**

# In[ ]:


fig = go.Figure(go.Bar(
    x = grouped_df['Type'],
    y = grouped_df['Count'],
    text=['Bar Chart'],
    name='Product Categories',
    marker_color=grouped_df['Count']
))

fig.update_layout(
    height=800,
    title_text='Distribution of Superbowl ads across product categories',
    showlegend=True
)

fig.show()


# **Line Chart**

# In[ ]:


fig = go.Figure(go.Scatter(
    x = grouped_df['Type'],
    y = grouped_df['Log Count'],
    text=['Line Chart observing the type'],
    name='Type',
    mode='lines+markers',
    marker_color='#56B870'
))

fig.update_layout(
    height=800,
    title_text='Superbowl Ads Distribution Across Type',
    showlegend=True
)

fig.show()


# ### Observe the distribution of Ads across product categories
# Let's create a pie chart visualization to observe the distribution

# **Observe the types of products in the Superbowl ads**

# In[ ]:


type_df = ads_df.groupby(['Type'], as_index=False, sort=False)['Product'].count()
type_df.head()


# **Pie Chart**

# In[ ]:


fig = px.pie(type_df, values='Product', names='Type')
fig.show()


# **Observe the Product distribution across types**

# In[ ]:


product_df = ads_df.groupby(['Product'], as_index=False, sort=False)['Type'].count()
product_df.head()


# In[ ]:


fig = px.pie(product_df, values='Type', names='Product')
fig.show()


# ## Visualize the distribution of Ads over time
# Group the ads by year and type across the fields

# In[ ]:


year_df = ads_df.groupby(['Year','Type'], as_index=False, sort=False).count()
year_df.head()


# In[ ]:


# Create traces
fig = go.Figure()

fig.add_trace(go.Scatter(
    x = year_df['Year'],
    y = year_df['Type'],
    text=['Line Chart with lines and markers'],
    name='Type',
    mode='markers',
    
))

fig.update_layout(
    height=800,
    title_text='Superbowl Ads Trends Over Time',
    showlegend=True
)

fig.show()

