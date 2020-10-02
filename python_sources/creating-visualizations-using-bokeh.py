#!/usr/bin/env python
# coding: utf-8

# In[25]:


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


# In[26]:


datacity = pd.read_csv('../input/Indian_cities.csv')


# In[27]:


datacity.head(2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


from bokeh.models import ColumnDataSource
#from bokeh.plotting import figure,output_notebook,show
from bokeh.io import output_notebook,show
from bokeh.plotting import figure
from bokeh.transform import dodge


# In[29]:


group = datacity[['population_total','population_male','population_female',
                                  'literates_total','literates_male',
          'literates_female']].groupby(datacity['state_name']).agg(
    {'population_total':np.sum,'population_male': np.sum,'population_female':np.sum,
      'literates_total':np.sum,'literates_male': np.sum,'literates_female':np.sum})
group = group.sort_values('population_total')
source12 = ColumnDataSource(group)


# In[ ]:


group.tail(2)


# In[ ]:


source12.column_names


# In[71]:


p = figure(plot_width=900,plot_height=500, x_range=group.index.values)
#p.vbar(x='state_name', top='population_male', width=1,line_color="white",color="#718dbf",legend='male', source=source12)
p.vbar(x=dodge('state_name', -0.25, range=p.x_range), top='population_male', width=0.2, source=source12,
       color="#718dbf", legend='male')

p.vbar(x=dodge('state_name',  0.0,  range=p.x_range), top='population_female', width=0.2, source=source12,
       color="#e84d60", legend='female')
p.xaxis.major_label_orientation = 1.2
p.x_range.range_padding = 0.1
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
output_notebook()
show(p)


# In[47]:


group1=group[['population_male','population_female']]


# In[48]:


group1.head(2)


# In[49]:


group1.index.values


# In[50]:


group1.reset_index(level=0, inplace=True)


# In[55]:


group1=pd.melt(group1,id_vars=['state_name'],value_vars=['population_male','population_female'])


# In[56]:


group1.head(5)


# In[59]:


group1 = group1.sort_values('state_name')
source1 = ColumnDataSource(group1)


# In[64]:


p = figure(plot_width=900,plot_height=500, x_range=group.index.values)
p.vbar(x='state_name', top='value', width=1,line_color="white",color="#718dbf", source=source1)
p.xaxis.major_label_orientation = 1.2
output_notebook()
show(p)


# In[68]:


print(source1.column_names)
print(source12.column_names)

