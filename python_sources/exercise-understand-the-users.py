#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Understand Users

# In[ ]:


#User-Defined-Functions
def get_location(x,i):
    try:
        return x[i]
    except:
        return 'Unknown'




def get_mode(x):
    data = Counter(x)
    return data.most_common(1)[0][0]


# In[ ]:


#read in raw data
page_views = pd.read_csv("../input/page_views_sample.csv")
events = pd.read_csv("../input/events.csv")


# In[ ]:


#Filter page_views based on user-id and time of event leading to clicks
events_filter = events[['uuid','timestamp']]
page_views_filter = page_views.merge(events_filter,how='inner',left_on='uuid',right_on='uuid')
page_views_filter = page_views_filter[page_views_filter['timestamp_x'] <= page_views_filter['timestamp_y']]


# In[ ]:


#Processing Location variable
page_views_filter['geo_location'] = page_views_filter['geo_location'].fillna("Unknown")
page_views_filter['geo_location'] = page_views_filter['geo_location'].apply(lambda x: x.split(">"))
page_views_filter['geo_country'] = page_views_filter['geo_location'].apply(lambda x: get_location(x,0))
page_views_filter['geo_state'] = page_views_filter['geo_location'].apply(lambda x: get_location(x,1)) 
page_views_filter['geo_dma'] = page_views_filter['geo_location'].apply(lambda x: get_location(x,2)) 
page_views_filter=page_views_filter.drop('geo_location',axis=1)


# In[ ]:


#Processing platform and traffic source
#platform (desktop = 1, mobile = 2, tablet =3)
page_views_filter['platform_desktop'] = page_views_filter['platform'].apply(lambda x: 1 if x == 1 else 0)
page_views_filter['platform_mobile'] = page_views_filter['platform'].apply(lambda x: 1 if x == 2 else 0)
page_views_filter['platform_tablet'] = page_views_filter['platform'].apply(lambda x: 1 if x == 3 else 0)
#traffic_source (internal = 1, search = 2, social = 3)
page_views_filter['source_internal'] = page_views_filter['traffic_source'].apply(lambda x: 1 if x == 1 else 0)
page_views_filter['source_search'] = page_views_filter['traffic_source'].apply(lambda x: 1 if x == 2 else 0)
page_views_filter['source_social'] = page_views_filter['traffic_source'].apply(lambda x: 1 if x == 3 else 0)


# In[ ]:


#Get rid of columns that are not useful
page_views_filter=page_views_filter.drop(['platform','traffic_source'],axis=1)


# In[ ]:


t=page_views_filter[0:10]


# In[ ]:


page_views_fa=page_views_filter.groupby(['uuid']).agg({
    'timestamp_x': [max,min],
    'document_id':len,
    'geo_country':get_mode,
    'geo_state':get_mode,
    'geo_dma':get_mode,
    'platform_desktop':sum,
    'platform_mobile':sum,
    'platform_tablet':sum,
    'source_internal':sum,
    'source_search':sum,
    'source_social':sum
    })


# In[ ]:


page_views_fa.columns = [''.join(col).strip() for col in page_views_fa.columns.values]


# In[ ]:


page_views_fa=page_views_fa.reset_index()


# In[ ]:


page_views_fa['time_diff'] = page_views_fa['timestamp_xmax'] - page_views_fa['timestamp_xmin']


# In[ ]:


page_views_fa['time_diff_by_min'] = page_views_fa['time_diff'].apply(lambda x: round(x/(1000*60),0))


# In[ ]:


page_views_fa[0:100]


# Exploratory Analysis And Visualization
# --------------------------------------

# In[ ]:


#Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig,ax = plt.subplots()
ax.set_title("The distribution of Time Lag")
sns.distplot(page_views_fa['time_diff_by_min'],ax=ax)


# In[ ]:


fig,ax = plt.subplots()
ax.set_title("The distribution of Time Lag (Number of activities > 1)")
sns.distplot(page_views_fa['time_diff_by_min'][page_views_fa['time_diff_by_min'] > 0],ax=ax)


# In[ ]:


doc_len_rebin = pd.cut(page_views_fa['document_idlen'],10,retbins=True)

xlabel = doc_len_rebin[0].value_counts().index.tolist()
y = doc_len_rebin[0].value_counts().values.tolist()
x = np.arange(len(y))

# add some text for labels, title and axes ticks
fig, ax = plt.subplots()
ax.bar(x,y)
ax.set_ylabel('Number of users')
ax.set_title('Number of document interacted')
ax.set_xticklabels(xlabel)


# In[ ]:


doc_len_rebin = pd.cut(page_views_fa['document_idlen'][page_views_fa['document_idlen']<=20],10,retbins=True)

xlabel = doc_len_rebin[0].value_counts().index.tolist()
y = doc_len_rebin[0].value_counts().values.tolist()
x = np.arange(len(y))
width = 0.35       # the width of the bars

# add some text for labels, title and axes ticks
fig, ax = plt.subplots()
ax.bar(x,y,width)
ax.set_ylabel('Number of users')
ax.set_title('Number of document interacted(<20)')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(xlabel,fontsize='small')

