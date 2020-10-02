#!/usr/bin/env python
# coding: utf-8

# ### 0.0 Load libraries

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        continue
        print(os.path.join(dirname, filename))
        
import colorlover as cl
from IPython.display import HTML

import collections


# ### 0.1 Load dataset

# In[ ]:


train = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_1.csv')


# In[ ]:


print(train.shape)


# In[ ]:


train.head(10)


# In[ ]:


train.columns


# In[ ]:


train.iloc[34436].Page


# ### Missing values

# I have a function that computes the number of missing values per column. There appear to be quite a few of those in this dataset, and they'll need to be dealt with!

# In[ ]:


def missingData(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum())/df.isnull().count().sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
    return missing_data


# In[ ]:


missingData(train).head(20)


# In[ ]:


missingData(train).tail(10)


# ## Individual Webpages

# The dataset consists of time series for a decently large number of pages. The first thing I did to explore the data was look at random at the time series for particular entries. 

# In[ ]:


sns.distplot(np.log1p(train.drop(columns='Page').sum(axis=1)), rug=True, kde=False)


# In[ ]:


data = [
    go.Histogram(
        x=np.log1p(train.drop(columns='Page').sum(axis=1))/np.log(10),
        histnorm='probability'
    )
]


layout = dict(
            title='Distribution of page views',
            autosize= True,
            bargap= 0.015,
            height= 400,
            width= 600,       
            hovermode= 'x',
            xaxis=dict(
            autorange= True,
            zeroline= False,
            tickvals=[0,1,2, 3, 4,5, 6, 7,8, 9,10],
            ticktext=['10$^0$', '10$^1$', '10$^2$', '10$^3$', '10$^4$', '10$^5$','10$^6$', '10$^7$','10$^8$','10$^9$', '10$^{10}$',]),
            yaxis= dict(
            autorange= True,
            showticklabels= True,
           ))

fig1 = dict(data=data, layout=layout)


iplot(fig1)


# In[ ]:


fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x = np.log1p(train.drop(columns='Page').sum(axis=1)),
        histnorm='probability',
        name = 'Training set')
)


fig.update_layout(height=450, width=900, title = 'Distribution of total no. views')

fig.show()


# In[ ]:


webpage = train.iloc[11214]
webpage_name = webpage['Page']
webpage = webpage.drop(labels = ['Page', 'lang'])
domnhall_gleeson = we

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = webpage.index,
        y = webpage,
        name='signal', 
        line = dict(color='crimson', width=4)
        )
)

"""
fig.add_trace(
    go.Scatter(
        x = webpage.index,
        y = webpage.rolling(14).mean(),
        name='2-week rolling average', 
        line = dict(color='crimson', width=4)
        )
)
"""    
    
    
fig.update_layout(
    height=600, 
    width=1400, 
    title=go.layout.Title(
        text=webpage_name,
        xref="paper",
        font=dict(
                size=24,
                #color="#7f7f7f"
            ),
        x=0
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Daily Traffic",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    )
)
    
    
fig.show()


# In[ ]:


webpage = train.iloc[34436]
webpage_name = webpage['Page']
webpage = webpage.drop(labels = ['Page'])

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = webpage.index,
        y = webpage,
        name='signal', 
        line = dict(color='crimson', width=4)
        )
)

"""
fig.add_trace(
    go.Scatter(
        x = webpage.index,
        y = webpage.rolling(14).mean(),
        name='2-week rolling average', 
        line = dict(color='crimson', width=4)
        )
)
"""    
    
    
fig.update_layout(
    height=600, 
    width=1400, 
    title=go.layout.Title(
        text=webpage_name,
        xref="paper",
        font=dict(
                size=24,
                #color="#7f7f7f"
            ),
        x=0
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Daily Traffic",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    )
)
    
    
fig.show()


# In[ ]:


webpage = train.iloc[4436]
webpage_name = webpage['Page']
webpage = webpage.drop(labels = ['Page'])

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = webpage.index,
        y = webpage,
        name='signal', 
        line = dict(color='crimson', width=4)
        )
)

"""
fig.add_trace(
    go.Scatter(
        x = webpage.index,
        y = webpage.rolling(14).mean(),
        name='2-week rolling average', 
        line = dict(color='crimson', width=4)
        )
)
"""    
    
    
fig.update_layout(
    height=600, 
    width=1400, 
    title=go.layout.Title(
        text=webpage_name,
        xref="paper",
        font=dict(
                size=24,
                #color="#7f7f7f"
            ),
        x=0
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Daily Traffic",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    )
)
    
    
fig.show()


# In[ ]:


npages = 5
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    print('\n\n')


# ## Global features: Page Language

# I'm reusing [muonneutrino's function](https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration) for computing the language of a webpage.

# In[ ]:


def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res[0][0:2]
    return 'na'

train['lang'] = train.Page.map(get_language)

from collections import Counter

languages = pd.DataFrame.from_dict(dict(Counter(train.lang)), orient='index', columns=['Count'])

fig = go.Figure([go.Bar(x=languages.index, y=languages.Count, marker_color='crimson')])

fig.update_layout(
    title=go.layout.Title(
        text="Wikipage total counts per language",
        xref="paper",
        font=dict(
                size=24,
                #color="#7f7f7f"
            ),
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Language",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Number of Webpages",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    )
)

fig.update_xaxes( tickfont=dict(size=16))
fig.update_yaxes( tickfont=dict(size=16))


fig.show()


# In[ ]:


lang_sets = {}

for language in languages.index:
    print(language)
    lang_sets[language] = train[train.lang==language].iloc[:,0:-1]
    lang_sets[language].index = pd.to_datetime(lang_sets[language].index)
    
sums = {}
for language in lang_sets:
    sums[language] = lang_sets[language].iloc[:,1:].sum(axis=0)/lang_sets[language].shape[0]

offset = collections.defaultdict(int)
offset['en'] = 1000
offset['ru'] =  3000
offset['es'] = 1800
offset['de'] = 1300
offset['ja'] = 1000
offset['fr'] = 600
offset['zh'] = 300


# In[ ]:


lang_dict = {'zh': 'Chinese', 'fr': 'French', 'en': 'English', 'ru': 'Russian', 'de': 'German', 'ja': 'Japanese', 'es': 'Spanish', 'na': 'Other'}


# In[ ]:


#cl.scales['7']
#HTML(cl.to_html( cl.scales['8'] )) # All scales with 11 colors


# In[ ]:


fig = go.Figure()

colorscale = cl.scales['8']['qual']['Set1']
i=0
for language in sums.keys():
    if offset[language]: name=lang_dict[language] + '(+' + str(offset[language])+')'
    else: name = lang_dict[language] 
    fig.add_trace(
        go.Scatter(
        x = sums[language].index,
        y = sums[language] + offset[language],
        name=name , 
        line = dict(color=colorscale[i], width=4)
        )
    )
    i+=1
    
    
fig.update_layout(
    height=600, 
    width=1400, 
    title=go.layout.Title(
        text="Time Series of Webpage Traffic in different languages (offsets for clarity)",
        xref="paper",
        font=dict(
                size=24,
                #color="#7f7f7f"
            ),
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Language",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Aggregate Number of Views",
            font=dict(
                size=18,
                #color="#7f7f7f"
            )
        )
    )
)
    
    
fig.show()


# ## Most popular webpages

# In[ ]:


npages = 5
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    print('\n\n')


# ### Periodicities with FFT

# In[ ]:


from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf


# In[ ]:





# In[ ]:





# In[ ]:




