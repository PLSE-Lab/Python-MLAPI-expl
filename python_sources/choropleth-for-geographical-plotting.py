#!/usr/bin/env python
# coding: utf-8

# In this kernel we will cover the basics of how to use choropleth map using python.Here we will be plotting maps on country and global scale.This kernel is a work on prrocess if you like my work please vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing python Module**

# In[ ]:


"""import plotly.plotly as py
import plotly.graph_obj as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)


# **Choropleth for USA**

# In[ ]:


data = dict(type='choropleth',
           locations = ['AZ','CA','NY'],
           locationmode = 'USA-states',
           colorscale = 'Portland',
           text = ['text 1','text 2','text 3'],
           z = [1.0,2.0,3.0],
           colorbar = {'title':'Colorbar Title Goes Here'})        # In place of Portland we can use Greens 


# In[ ]:


layout = dict(geo={'scope':'usa'})


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# ### US Agricultural Export Year 2011

# In[ ]:


df= pd.read_csv('../input/2011_US_AGRI_Exports')
df.head()


# So we have the statewise Agricultural Produce Export for USA.

# In[ ]:


data1 = dict(type='choropleth',
           locations = df['code'],
           locationmode = 'USA-states',
           colorscale = 'Portland',
           text = df['text'],
           z = df['total exports'],
           colorbar = {'title':'Millions USD'})    


# In[ ]:


layout1 = dict(title = '2011 US Agriculture Exports by State',
             geo = dict(scope = 'usa',showlakes = True,lakecolor = 'rgb(85,173,240)'))


# In[ ]:


layout1


# In[ ]:


chromap2 = go.Figure(data = [data1],layout =layout1)


# In[ ]:


iplot(chromap2)


# We can see that the agricultural exports are more from the states of California,Iowa and Illinois.

# ### Choropleth for Internation Level

# In[ ]:


df2= pd.read_csv('../input/2014_World_GDP')
df2.head()


# We have the GDP Data of different countries for the year 2014.We will be plotting this data on a world map.

# In[ ]:


data = dict(type = 'choropleth',
           locations = df2['CODE'],
           z=df2['GDP (BILLIONS)'],
           text = df2['COUNTRY'],
           colorbar = {'title':'GDB in Billions USD'})


# In[ ]:


layout = dict(title='2014 Global GDP',
             geo = dict(showframe= False,
                       projection = {'type': 'mercator'}))


# In[ ]:


choromap3 = go.Figure(data=[data],layout=layout)
iplot(choromap3)


# So the big economies of the world USA and China are quite distinctly evident in the Choropleth.

# ### World Power Consumption

# In[ ]:


df3= pd.read_csv('../input/2014_World_Power_Consumption')
df3.head()


# In[ ]:


data3 = dict(type = 'choropleth',
           locations = df3['Country'],
           locationmode = 'country names',
           z=df3['Power Consumption KWH'],
           text = df3['Country'],
           colorbar = {'title':'Power Consumption KWH'})


# In[ ]:


layout3 = dict(title='2014 Power Consumption',
             geo = dict(showframe= False,
                       projection = {'type': 'mercator'}))


# In[ ]:


choromap3 = go.Figure(data=[data3],layout=layout3)
iplot(choromap3,validate = False)


# So we can see that highest power consuming country id China Followed by USA.

# **Changing the color scale to improve vizualization**

# In[ ]:


data3 = dict(type = 'choropleth',
           locations = df3['Country'],
           colorscale = 'Viridis',
           reversescale = True,
           locationmode = 'country names',
           z=df3['Power Consumption KWH'],
           text = df3['Country'],
           colorbar = {'title':'Power Consumption KWH'})


# In[ ]:


layout3 = dict(title='2014 Power Consumption',
             geo = dict(showframe= False,
                       projection = {'type': 'mercator'}))


# In[ ]:


choromap3 = go.Figure(data=[data3],layout=layout3)
iplot(choromap3,validate = False)


# Now we can see that the above plot has a better color scale than the previous plot.This helps us to identify the countries more easily based on the power the consume.

# ### USA 2012 Election Data

# In[ ]:


df4= pd.read_csv('../input/2012_Election_Data')
df4.head()


# In[ ]:


data4 = dict(type = 'choropleth',
           colorscale = 'Viridis',
           reversescale = True,
           locations = df4['State Abv'],
           z=df4['Voting-Age Population (VAP)'],                  
           locationmode = 'USA-states',           
           text = df4['State'],
           colorbar = {'title':'Voting Age Population'})


# In[ ]:


layout4 = dict(title='2012 Election Data',
              geo = dict(scope='usa',showlakes=True,lakecolor='rgb(85,173,240)'))


# In[ ]:


choromap4 = go.Figure(data=[data4],layout=layout4)
iplot(choromap4,validate = False)


# In[ ]:




