#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/serp_flights.csv")
data.head()


# In[ ]:


data.describe()


# #### This is for visualizing missing data
# Here's some noise in the data

# In[ ]:


import missingno as msno
msno.matrix(data)
plt.show()


# In[ ]:


data.isnull().sum()


# In[ ]:


new_data = data[['searchTerms','title','snippet','displayLink','queryTime','formattedSearchTime']]
new_data.head()


# ### Noisy data or missing data is removed by removing the column of it 
# 

# In[ ]:


import missingno as msno
msno.matrix(new_data)
plt.show()


# In[ ]:


new_data.isnull().sum()


# ### Counting the number of different Websites used in the dataset

# In[ ]:


websites_count = data['displayLink'].value_counts()
websites_count


# ### Adding new column of 'WebsiteCounts'

# In[ ]:


new_data['WebsiteCounts'] = new_data.groupby(['displayLink'])['formattedSearchTime'].transform('count')
new_data.head()


# ### Renaming the name of the column

# In[ ]:


new_data = new_data.rename(columns={'displayLink':'Website'})
new_data.head()


# ## Visualization

# In[ ]:


from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image


# In[ ]:


colors = {
    "Bug": "#A6B91A",
    "Dark": "#705746",
    "Dragon": "#6F35FC",
    "Electric": "#F7D02C",
    "Fairy": "#D685AD",
    "Fighting": "#C22E28",
    "Fire": "#EE8130",
    "Flying": "#A98FF3",
    "Ghost": "#735797",
    "Grass": "#7AC74C",
    "Ground": "#E2BF65",
    "Ice": "#96D9D6",
    "Normal": "#A8A77A",
    "Poison": "#A33EA1",
    "Psychic": "#F95587",
    "Rock": "#B6A136",
    "Steel": "#B7B7CE",
    "Water": "#6390F0",
}


# ## Most Searched Website with its count while booking a ticket

# In[ ]:


data = go.Bar(
    x=new_data['Website'][:20],
    y=new_data['WebsiteCounts'][:20],
    marker=dict(
        color=list(colors.values())
    ),

)

layout = go.Layout(
    title='Webiste Counts',
    xaxis=dict(
        title='Webistes'
    ),
    yaxis=dict(
        title='Webistes Count'
    )
)

fig = go.Figure(data=[data], layout=layout)
iplot(fig, filename='Webiste Counts')


# In[ ]:


new_data['title'].value_counts()


# In[ ]:


new_data['searchTerms'].value_counts()


# #### Do Upvote and share if you like this
# ### Thank You

# In[ ]:




