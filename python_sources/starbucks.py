#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('pip install pyforest')


# In[ ]:


from pyforest import *
import plotly
#import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
import os
import warnings
warnings.filterwarnings('ignore')
import mpl_toolkits
from IPython.display import Image
Image(url='https://rde-stanford-edu.s3.amazonaws.com/Hospitality/Images/starbucks-header.jpg')


# In[ ]:


df = pd.read_csv('../input/directory.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


def update_column(column):
    return column.replace(' ', '_').lower()


# In[ ]:


starbucks = df.copy()


# In[ ]:


starbucks.columns = starbucks.columns.map(update_column)


# In[ ]:


starbucks.info()


# In[ ]:


starbucks.isnull().sum()


# In[ ]:


starbucks.ownership_type.unique()


# In[ ]:


starbucks.info()


# In[ ]:


starbucks.country.unique()


# In[ ]:


country_indices, country_labels = starbucks.country.factorize()
country_labels


# In[ ]:


country_indices


# In[ ]:


starbucks['country_indice'] = country_indices


# In[ ]:


starbucks['country'].value_counts()


# In[ ]:


sns.set(style='dark', context='talk')
sns.countplot(x='ownership_type', data=starbucks, palette='BuGn_d')


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set(title = 'Top 10 Countries with most number of Starbucks outlets')
starbucks.country.value_counts().head(10).plot(kind='bar', color='blue')


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set(title='Top 10 Store Names with most Outlets')
starbucks.store_name.value_counts().head(10).plot(kind='bar', color='orange')


# In[ ]:


starbucks['state/province'].value_counts()


# In[ ]:


usa_states = starbucks[starbucks['country'] == 'US']
usa_states['state/province'].value_counts().head(1)


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set(title='Top 10 States of USA with most outlets')
usa_states['state/province'].value_counts().head(10).plot(kind='bar', color='purple')


# In[ ]:


starbucks.brand.value_counts()


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set(title='Brand under which Starbucks operate')
starbucks.brand.value_counts().head(10).plot(kind='bar', color='pink')


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\HITARTH SHAH\\Anaconda3\\ANA_NAV\\Library\\share'
from mpl_toolkits.basemap import Basemap


# In[ ]:


get_ipython().system('pip install --upgrade pip')


# In[ ]:


get_ipython().system('conda install -c conda-forge basemap-data-hires --yes')


# In[ ]:


plt.figure(figsize=(12,9))
m=Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='h')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()
x, y=m(list(starbucks['longitude'].astype(float)), list(starbucks['latitude'].astype(float)))
m.plot(x,y,'bo', markersize=5, alpha=0.6, color='blue')
plt.title('Starbucks Stores Across the World')
plt.show()


# In[ ]:


plt.figure(figsize=(12,9))
m=Basemap(projection='mill', llcrnrlat=20, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-60, resolution='h')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()
x, y=m(list(starbucks['longitude'].astype(float)), list(starbucks['latitude'].astype(float)))
m.plot(x,y,'bo', markersize=5, alpha=0.6, color='blue')
plt.title('Extinct and Endangered Languages in USA')
plt.show()


# In[ ]:




