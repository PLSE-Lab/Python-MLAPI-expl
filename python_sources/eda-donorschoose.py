#!/usr/bin/env python
# coding: utf-8

# ## Detailed explainations Yet to be Added Stay tuned !

# ### Import libraries

# In[1]:


import difflib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import nxviz as nz
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))


# ### Read Files

# In[2]:


donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False, parse_dates=["Teacher First Project Posted Date"])
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)


# ### Display File content

# In[3]:


resources.head()


# In[4]:


teachers.head()


# In[5]:


donors.head()


# In[6]:


projects.head()


# ### Merging with Donations table

# In[7]:


projects = projects.merge(teachers[['Teacher ID', 'Teacher Prefix']], on=['Teacher ID'])
donations = donations.merge(projects[['Project ID', 'Project Type', 'Project Subject Category Tree', 'Project Resource Category', 'Project Grade Level Category', 'Teacher Prefix', 'Project Cost']], on=['Project ID'])
donations = donations.merge(donors[['Donor ID', 'Donor State', 'Donor City']], on=['Donor ID'])
donations = donations.merge(resources[['Resource Unit Price', 'Project ID']], on=['Project ID'])
donations['Project Cost'] = donations['Project Cost'].replace('[\$,]', '', regex=True).astype(float)
donations['Donation Received Date'] =  pd.to_datetime(donations['Donation Received Date'], format='%Y-%m-%d %H:%M:%S')
donations['year'] = pd.DatetimeIndex(donations['Donation Received Date']).year
donations = donations.dropna()


# In[10]:


donations.head()


# ### Funding distribution among different project type

# In[11]:


plt.figure(figsize=(12,8))
temp = donations.groupby('Project Type')['Donation Amount'].sum()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Funding distribution among different project type')


# ### Percentage of Funds Distributed among different teacher prefixes

# In[13]:


plt.figure(figsize=(12,8))
temp = donations['Teacher Prefix'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Percentage of Funds Distributed among different teacher prefixes')


# ### Funding Amount distribution among different teacher prefixes

# In[14]:


plt.figure(figsize=(12,8))
temp = donations.groupby(['Teacher Prefix'])['Donation Amount'].sum()
temp.iplot(kind='bar', yTitle='Total amount', title='Funding Amount distribution')


# ### Projection of sum of Funding recieved by different project types

# In[18]:


df = donations.groupby(['Project Type', 'year']).agg({'Donation Amount':sum})
df = df.reset_index(level=[0,1])
df = df.pivot(index="year", columns='Project Type', values='Donation Amount')
py.iplot([{
    'x': df.index,
    'y': df[col],
    'name': col
}  for col in df.columns])


# ### Projection of Number of Fundings Recieved by project types in each year from 2012-2018

# In[19]:


df = donations.groupby(['Project Type', 'year'])['Donation Amount'].count()
df = df.reset_index(level=[0,1])
df = df.pivot(index="year", columns='Project Type', values='Donation Amount')
py.iplot([{
    'x': df.index,
    'y': df[col],
    'name': col
}  for col in df.columns])


# ### sum of Funding recieved by every donor state 

# In[20]:


df = donations.groupby(['Donor State', 'year']).agg({'Donation Amount':sum})
df = df.reset_index(level=[0,1])
df = df.pivot(index="year", columns='Donor State', values='Donation Amount')
py.iplot([{
    'x': df.index,
    'y': df[col],
    'name': col
}  for col in df.columns])


# ### stacked projection of sum of funding recieved by  each Project Resource Category by year

# In[21]:


df = donations.groupby(['Project Resource Category', 'year']).agg({'Donation Amount':sum})
df = df.reset_index(level=[0,1])
df = df.pivot(index="year", columns='Project Resource Category', values='Donation Amount')
df.iplot(kind='area', fill=True)


# ## Fnding measurements based on this realtion:
#     size of bubble represents Project Cost
#     x axis denote Resource unit price
#     y axis denote Donation amount

# ### Based on Project Resource Category

# In[27]:


df = donations.groupby('Project Resource Category').agg({'Donation Amount':sum, 'Project Cost':sum, 'Resource Unit Price':sum})
df = df.reset_index(level=[0])
df.iplot(kind='bubble', y='Resource Unit Price', x='Donation Amount', size='Project Cost', text='Project Resource Category',
             xTitle='Donation Amount', yTitle='Resource Unit Price')


# ### Based on Project Subject Category Tree

# In[24]:


df = donations.groupby('Project Subject Category Tree').agg({'Donation Amount':sum, 'Project Cost':sum, 'Resource Unit Price':sum})
df = df.reset_index(level=[0])
df.iplot(kind='bubble', y='Resource Unit Price', x='Donation Amount', size='Project Cost', text='Project Subject Category Tree',
             xTitle='Donation Amount', yTitle='Resource Unit Price')


# ### Based on Project Grade Level Category

# In[26]:


df = donations.groupby('Project Grade Level Category').agg({'Donation Amount':sum, 'Project Cost':sum, 'Resource Unit Price':sum})
df = df.reset_index(level=[0])
df.iplot(kind='bubble', y='Resource Unit Price', x='Donation Amount', size='Project Cost', text='Project Grade Level Category',
             xTitle='Donation Amount', yTitle='Resource Unit Price')


# ### Based on Donor State

# In[25]:


df = donations.groupby('Donor State').agg({'Donation Amount':sum, 'Project Cost':sum, 'Resource Unit Price':sum})
df = df.reset_index(level=[0])
df.iplot(kind='bubble', y='Resource Unit Price', x='Donation Amount', size='Project Cost', text='Donor State',
             xTitle='Donation Amount', yTitle='Resource Unit Price')


# In[ ]:


#df.iplot(subplots=True, subplot_titles=True, legend=False)


# ## Network Map depecting states which have same donation pattern

# In[29]:


df = donations.groupby(['Donor State', 'Project Resource Category']).agg({'Donation Amount':sum})
df = df.reset_index(level=[0,1])
df = df.sort_values(by=['Donation Amount'], ascending=[False]).groupby('Donor State').agg(lambda x: x.tolist()).reset_index().values.tolist()


# In[30]:


temp = {}
for i in range(len(df)):
    temp[df[i][0]] = df[i][1]

df = {}
for i in temp.keys():
    for j in temp.keys():
        sm = difflib.SequenceMatcher(None, temp[i], temp[j])
        if sm.ratio()>0.80:
            df[(i, j)] = 1
        else:
            df[(i, j)] = 0
            
temp = []
for k, v in df.items():
    if v == 1:
        temp.append(k)


# In[32]:


plt.figure(figsize=(20,20))
G = nx.Graph()
G.add_edges_from(temp)
options = {
    'node_color': 'green',
    'node_size': 800,
}
nx.draw_random(G, with_labels=True, font_weight='bold', width=0.5, color="yellow", **options)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




