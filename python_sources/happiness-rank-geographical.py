#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import plotly.plotly as py 
import plotly.graph_objs as go


# In[ ]:





# In[ ]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


df = pd.read_csv('../input/2015.csv')


# In[ ]:


df.head()


# In[ ]:


data = dict(type='choropleth',
           locations = df['Country'],
           locationmode='country names',
           z = df['Happiness Score'],
           text = df['Country'],
           colorbar= {'title' : 'Happiness Score'})


# In[ ]:


layout = dict(title='2015 Happiness Rank',
            geo = dict(showframe = False,
             projection = {'type' : 'kavrayskiy7'}))


# In[ ]:


choromap = go.Figure(data=[data],layout = layout)


# In[ ]:


iplot(choromap)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


i


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




