#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
from plotly.graph_objs import *

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()



# In[ ]:


df=pd.read_csv('../input/us-charities.csv') 


# In[ ]:


plotly.tools.set_credentials_file(username='vochiphat', api_key='aYSd242sROhvQX4Qbfgb')


# #### descriptive analysis

# In[ ]:


df.info()


# In[ ]:


print(df.columns) 


# In[ ]:


df.head(5)


# In[ ]:


df.describe()


# In[ ]:


df.groupby('State')


# In[ ]:


df.groupby('State').groups


# In[ ]:


df.groupby(['State'])['Charity name'].count()


# In[ ]:


grouped = df.groupby('State')

for name, group in grouped:
    print(name)
    print(group)


# In[ ]:


for title, group in df.groupby('State'):
    group.hist(x='Charity name', y='Net assets', title=title)


# In[ ]:


texas=grouped.get_group('TX')


# #### visualization

# In[ ]:


df['Net assets'].max()


# In[ ]:


data = [Bar(x=df["Charity name"],
            y=df["Net assets"])]

py.iplot(data,filename='bar_units sold')


# In[ ]:


df['Fundraising expenses'].max()


# In[ ]:


data = [Histogram(x=df['Fundraising expenses'])]


py.iplot(data,filename='histogram')


# In[ ]:


g= sns.FacetGrid(df, col='State')
g.map(plt.hist,'Administrative expenses');
g.add_legend
sns.set(rc={'figure.figsize':(110.7,80.27)})


# In[ ]:


sns.set_style("whitegrid")
sns.boxplot(x='Organization type', y="Net assets", data=df, palette="dark")
sns.despine(left=True)


# In[ ]:





# In[ ]:





# In[ ]:




