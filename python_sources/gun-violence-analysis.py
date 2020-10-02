#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


import os


# In[ ]:


import matplotlib.pyplot as plt 


# In[ ]:


import seaborn as sns 


# In[ ]:


df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")


# In[ ]:


df.shape
df.index.values
df.columns.values
df.dtypes


# In[ ]:


df['killed_or_not'] = np.where(df['n_killed'] > 0, 'YES', 'NOT')
df['killed_or_not'].value_counts()


# In[ ]:


df.isna().sum()
df.isnull().sum()


# In[ ]:


df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year


# In[ ]:


gp_state = df.groupby(by=['state']).agg({'n_killed':'sum','n_injured':'sum','n_guns_involved':'sum'}).reset_index()


# In[ ]:


gp_state['n_killed'] =  (gp_state['n_killed'] - min(gp_state['n_killed']))/(max(gp_state['n_killed']) - min(gp_state['n_killed']))
gp_state['n_injured'] =  (gp_state['n_injured'] - min(gp_state['n_injured']))/(max(gp_state['n_injured']) - min(gp_state['n_killed']))
gp_state['n_guns_involved'] =  (gp_state['n_guns_involved'] - min(gp_state['n_guns_involved']))/(max(gp_state['n_guns_involved']) - min(gp_state['n_guns_involved']))


# In[ ]:


gp_state


# In[ ]:


g = sns.jointplot("n_injured",
              "n_killed",
              gp_state
              )

g.ax_joint.plot(                # Plot y versus x as lines and/or markers.
               np.linspace(0, 1),
               np.linspace(0, 1)
               )


# In[ ]:


varsforgrid = ['n_killed', 'n_injured','n_guns_involved']
g = sns.PairGrid(gp_state,
                 vars=varsforgrid,  # Variables in the grid
                 hue='state'       # Variable as per which to map plot aspects to different colors.
                 )
g = g.map_diag(plt.hist)                   
g.map_offdiag(plt.scatter)
g.add_legend();


# In[ ]:


g = sns.distplot(df['n_killed'], kde=True);
g.axvline(0, color="red", linestyle="--");
plt.xlim(0,5)
plt.ylim(0,1)


# In[ ]:


gp_year = df.groupby(by=['year']).agg({'n_killed':'sum','n_injured':'sum','n_guns_involved':'sum','killed_or_not':'count'}).reset_index()
gp_year.killed_or_not


# In[ ]:


gp_year['n_killed'] =  (gp_year['n_killed'] - min(gp_year['n_killed']))/(max(gp_year['n_killed']) - min(gp_year['n_killed']))
gp_year['n_injured'] =  (gp_year['n_injured'] - min(gp_year['n_injured']))/(max(gp_year['n_injured']) - min(gp_year['n_killed']))
gp_year['n_guns_involved'] =  (gp_year['n_guns_involved'] - min(gp_year['n_guns_involved']))/(max(gp_year['n_guns_involved']) - min(gp_year['n_guns_involved']))


# In[ ]:


varsforgrid = ['n_killed', 'n_injured','n_guns_involved']
g = sns.PairGrid(gp_year,
                 vars=varsforgrid,  # Variables in the grid
                 hue='year'       # Variable as per which to map plot aspects to different colors.
                 )
g = g.map_diag(plt.hist)                   
g.map_offdiag(plt.scatter)
g.add_legend();


# In[ ]:


sns.boxplot(x='year',y='killed_or_not',data = df)


# In[ ]:


sns.violinplot("year", "killed_or_not", data=df );     # x-axis has categorical variable

