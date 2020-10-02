#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

Avocado = pd.read_csv('../input/avocado.csv')

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#print(os.listdir("../input/avocado.csv"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Avocado.head()


# After importing the dataset as Avocado, I read the top five rows

# In[ ]:


Avocado.info()


# In[ ]:


Avocado.describe()


# This line gives mathematical information about the dataset 

# In[ ]:


sns.heatmap(Avocado.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# Avocado.isnull() returns boolean values of the dataframe. Therefore, if there were any null value in the dataframe, it would return False values and make differences on the heatmap. 

# In[ ]:


regions = Avocado['region'].unique()
tot_vol_by_region = {}
for region in regions:
    if region != 'TotalUS':
        avg_vol = sum(Avocado[Avocado['region']==region]['Total Volume'])/list(Avocado['region']==region).count(True)
        tot_vol_by_region[region] = avg_vol 
tot_vol_by_region

Avg_vol = pd.DataFrame({'Region':list(tot_vol_by_region),'avg_vol':list(tot_vol_by_region.values())})
Avg_vol.sort_values(by = 'avg_vol',inplace = True,ascending=False)


# ###

# In[ ]:


plt.figure(figsize=(12,10))
sns.barplot(y= 'avg_vol',x='Region',data=Avg_vol,palette='pastel')
plt.tight_layout()
plt.xticks(rotation='vertical')
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('Average Volume',{'fontsize':'large'})
plt.title("Average Volume in Each Region",{'fontsize':20})


# In[ ]:


regions = Avocado['region'].unique()
avg_price_by_region = {}
for region in regions:
    if region != 'TotalUS':
        avg_price = sum(Avocado[Avocado['region']==region]['AveragePrice'])/list(Avocado['region']==region).count(True)
        avg_price_by_region[region] = avg_price #.append({region: avg_vol})
avg_price_by_region

Avg_price = pd.DataFrame({'Region':list(avg_price_by_region.keys()),'avg_price':list(avg_price_by_region.values())})#,columns=['a','b'])
Avg_price.sort_values(by = 'avg_price',inplace = True,ascending=False)


# In[ ]:


plt.figure(figsize=(12,10))
sns.barplot(y= 'avg_price',x='Region',data=Avg_price,palette='Set2')
plt.xticks(rotation='vertical')
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('Average Price',{'fontsize':'large'})
plt.title("Average Price in Each Region",{'fontsize':20})


# In[ ]:


plt.figure(figsize=(12,10))
sns.set_style('whitegrid')
sns.pointplot(x='AveragePrice',y='region',data=Avocado, hue='year',join=False)
plt.xticks(np.linspace(1,2,5))
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('Average Price',{'fontsize':'large'})
plt.title("Yearly Average Price in Each Region",{'fontsize':20})


# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.plotly as py


# In[ ]:


import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


trace0 = go.Box(y=Avocado[Avocado['type']=='conventional']['AveragePrice'],name='Converntional')
trace1 = go.Box(y=Avocado[Avocado['type']=='organic']['AveragePrice'],name='Organic')
data = [trace0, trace1]
layout = go.Layout(title = 'Average Price by Type', xaxis = dict(title='Type',titlefont=dict(size=20,color='black'))
                  , yaxis = dict(title = 'Average Price',titlefont=dict(size=20,color='black')))

fig = go.Figure(data=data,layout=layout)
#py.iplot(fig)
iplot(fig)


# In[ ]:


New_Dict = {}
for region in regions:
    Avg_Price_by_Type = Avocado[(Avocado['region'] == region) & (Avocado['type']=='organic')]
    New_Dict[region] = Avg_Price_by_Type['AveragePrice'].mean()

Organic_region = pd.DataFrame({'Region':list(New_Dict.keys()),'avg_price':list(New_Dict.values())})

trace0 = go.Scatter(
    x = Organic_region['Region'],
    y = Organic_region['avg_price'],
    name = 'Organic'
)

New_Dict1 = {}
for region in regions:
    Avg_Price_by_Type = Avocado[(Avocado['region'] == region) & (Avocado['type']=='conventional')]
    New_Dict1[region] = Avg_Price_by_Type['AveragePrice'].mean()

Con_region = pd.DataFrame({'Region':list(New_Dict1.keys()),'avg_price':list(New_Dict1.values())})
    
trace1 = go.Scatter(
    x = Con_region['Region'],
    y = Con_region['avg_price'],
    name = 'Conventional'
)

layout = go.Layout(title = 'Average Price by Type in Each Region',
                   xaxis = dict(title='Region',titlefont=dict(size=20,color='black')),
                   yaxis = dict(title = 'Average Price',titlefont=dict(size=20,color='black')),
                   )

data = [trace0,trace1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)

