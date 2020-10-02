#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import (absolute_import,division,print_function,unicode_literals)
import warnings
warnings.simplefilter('ignore')

get_ipython().run_line_magic('pylab', 'inline')
from pylab import rcParams
rcParams['figure.figsize']=8,5
import seaborn as sns


# In[2]:


df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.info()


# Some faeture need to set type manually

# In[5]:


df.head()


# In[6]:


df = df.dropna()
df['User_Score'] = df.User_Score.astype('float64')
df['Year_of_Release'] = df.Year_of_Release.astype('int64')
df['User_Count'] = df.User_Count.astype('int64')
df['Critic_Count'] = df.Critic_Count.astype('int64')


# In[8]:


df.shape


# So, we have 6825 objects & 16 features for them. Let's look as some useful features

# In[10]:


useful_cols = ['Name','Platform','Year_of_Release','Genre','Global_Sales',
              'Critic_Score','Critic_Count','User_Score','User_Count',
              'Rating']
df[useful_cols].head()


# First, lets try the most easy way to visualize -- DataFrame.plot

# In[12]:


sales_df = df[[x for x in df.columns if 'Sales' in x]+['Year_of_Release']]
sales_df.groupby('Year_of_Release').sum().plot()


# Not bad! But if we add parameter *kind* we can change type of plot.  Also we can rotate text under X axes with function *rot*

# In[13]:


sales_df.groupby('Year_of_Release').sum().plot(kind='bar',rot=45)


# Looks nice! 
# Now lets try ** Seaborn**
# *Seaborn* can make more difficult pictures than *matplotlib*. For example such difficult graph will be *pair plot* (scatter plot matrix). And we can see, how different features are corelated between each other

# In[15]:


cols = ['Global_Sales','Critic_Score', 'Critic_Count', 'User_Score','User_Count']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')


# With *Seaborn* we can draw *dist plot*. For example lets look at distribution of *Critic_Score*

# In[16]:


sns.distplot(df.Critic_Score)


# Also except *pair plot* we can research correlation between 2 features with function *joint plot*. This function is the hibrid of *scatter plot* & *histogram*.  Lets look how User_score is corelated with Critic_Score

# In[17]:


sns.jointplot(df['Critic_Score'],df['User_Score'])


# Another useful type of graphs is *boxplot*

# In[19]:


top_platforms = df.Platform.value_counts().sort_values(ascending=False).head(5).index.values
sns.boxplot(y='Platform',x='Critic_Score',data=df[df.Platform.isin(top_platforms)],orient='h')


# Lets analyse this picture... So.. *BOX PLOT* consist of the **box** and **mustache**. The Box shows us interquartil distribution = 25% (Q1) and 75% (Q2) percentiles. 
# Line inside the box is the median of feature distribution.
# That was a box, now mustache :)). Mustache shows the whole dispersion of feature values besides ejections, i.e. minimal and maximal values which inside of the interval 
# (Q1 - 1.5*IQR, Q3+1.5*IQR), where IQR=Q3-Q1 - interquartile dispersion. The dots on graph are ejections.
# This picture from Wikipedia
# ![https://commons.wikimedia.org/w/index.php?curid=14524285)](http://)
# 

# And the last in seaborn let it be *Heat map*. 

# In[20]:


platform_genre_sales = df.pivot_table(index='Platform',columns='Genre',
                                     values='Global_Sales',
                                     aggfunc=sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales,annot=True,fmt=".1f",linewidth=.5)


# <H1>**Plotly**</H1>
# We saw visualisations was based on *matplotlib*. But of course this is not the only library for graphs in *python*. Another is *Plotly* - this is open sourse library which can draw interactive graphs in Jupyter Notebook without java-script knowledge.
# Lets try it!

# In[21]:


from plotly.offline import download_plotlyjs,init_notebook_mode,iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)


# In[23]:


# Count number of released games & sold copies in years
years_df = df.groupby('Year_of_Release')[['Global_Sales']].sum().join(df.groupby('Year_of_Release')[['Name']].count()
                                                                    )
years_df.columns = ['Global_Sales','Number_of_Games']

# Create line for number of sold copies
trace0 = go.Scatter(x=years_df.index,y=years_df.Global_Sales,name='Global Sales')

# Create line for number of released games
trace1 = go.Scatter(x=years_df.index,y=years_df.Number_of_Games,name='Number of Games Released')

# Set data list and title
data = [trace0,trace1]
layout = {'title':'Statistics of video games'}

# Create object Figure and visualize it
fig = go.Figure(data=data,layout=layout)
iplot(fig,show_link=False)


# We can save this graph in html-file:

# In[24]:


plotly.offline.plot(fig,filename='years_stats.html',show_link=False)


# Now lets try to look at market percent of gaming platforms, calculated and based on count of games released and on sum sales.  We will use *bar chart*

# In[26]:


# Count the number of sold and released games in platforms
platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(df.groupby('Platform')[['Name']].count())
platforms_df.columns = ['Global_Sales','Number_of_Games']
platforms_df.sort_values('Global_Sales',ascending=False,inplace=True)

# Create traces for visualizasion
trace0 = go.Bar(x=platforms_df.index,y=platforms_df.Global_Sales,name='Global Sales')
trace1 = go.Bar(x=platforms_df.index,y=platforms_df.Number_of_Games,name='Number of games released')

# Create list with data and title
data = [trace0,trace1]
layout = {'title':'Share of platforms','xaxis':{'title':'platform'}}

# Create object FIgure
fig = go.Figure(data=data,layout=layout)
iplot(fig,show_link=False)


# In **Plotly** we can draw *boxplots*. Let's see distribution between Critic_Counts and Genre

# In[28]:


# Create Box trace for each genre
data = []
for genre in df.Genre.unique():
    data.append(go.Box(y=df[df.Genre==genre].Critic_Score,name=genre))
# Visualize data
iplot(data,show_link=False)
                


# That's all, my friend.
