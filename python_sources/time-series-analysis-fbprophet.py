#!/usr/bin/env python
# coding: utf-8

# ## General Information
# 
# ## Data Description
# 
# The training dataset consists of approximately 145k time series. Each of these time series represent a number of daily views of a different Wikipedia article, starting from July, 1st, 2015 up until December 31st, 2016. The leaderboard during the training stage is based on traffic from January, 1st, 2017 up until March 1st, 2017.
# 
# The second stage will use training data up until September 1st, 2017. The final ranking of the competition will be based on predictions of daily views between September 13th, 2017 and November 13th, 2017 for each article in the dataset. You will submit your forecasts for these dates by September 12th.
# 
# For each time series, you are provided the name of the article as well as the type of traffic that this time series represent (all, mobile, desktop, spider). You may use this metadata and any other publicly available data to make predictions. Unfortunately, the data source for this dataset does not distinguish between traffic values of zero and missing values. A missing value may mean the traffic was zero or that the data is not available for that day.
# 
# To reduce the submission file size, each page and date combination has been given a shorter Id. The mapping between page names and the submission Id is given in the key files.
# 
# ## File descriptions
# 
# Files used for the first stage will end in '_1'. Files used for the second stage will end in '_2'. Both will have identical formats. The complete training data for the second stage will be made available prior to the second stage.
# 
# **train_*.csv** - contains traffic data. This a csv file where each row corresponds to a particular article and each column correspond to a particular date. Some entries are missing data. The page names contain the Wikipedia project (e.g. en.wikipedia.org), type of access (e.g. desktop) and type of agent (e.g. spider). In other words, each article name has the following format: 'name_project_access_agent' (e.g. 'AKB48_zh.wikipedia.org_all-access_spider').
# 
# **key_*.csv** - gives the mapping between the page names and the shortened Id column used for prediction
# 
# **sample_submission_*.csv** - a submission file showing the correct format

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # search in texts as scrap
from collections import Counter
from fbprophet import Prophet
import matplotlib.pyplot as plt # graph
from plotly.offline import init_notebook_mode, iplot # graph
from plotly import graph_objs as go # graph

# Initialize plotly
init_notebook_mode(connected=True)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Importing Kaggle's Files

# In[ ]:


# importing and seeing the kaggle's file
k1=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_1.csv')
k1.head()


# In[ ]:


# importing and seeing the kaggle's file
k2=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_2.csv')
k2.head()


# In[ ]:


# importing and seeing the kaggle's file
t2=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_2.csv')
t2.head()


# In[ ]:


# importing and seeing the kaggle's file
t1=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_1.csv')
t1.head()


# In[ ]:


# importing and seeing the kaggle's file
ss2=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/sample_submission_2.csv')
ss2.head()


# In[ ]:


# importing and seeing the kaggle's file
ss1=pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/sample_submission_1.csv')
ss1.head()


# In[ ]:


# view shapes of files 

print('shape t1')
print(t1.shape)
print('shape t2')
print(t2.shape)
print('shape k1')
print(k1.shape)
print('shape k2')
print(k2.shape)
print('shape ss1')
print(ss1.shape)
print('shape ss2')
print(ss2.shape)


# ## WORKING 

# In[ ]:


# Verify general behavior

#Total sum per column: 
t1.loc['Total',:]= t1.mean(axis=0)


# In[ ]:


x = t1.tail(1)
x = x.T
x


# In[ ]:


x.drop('Page', axis=0, inplace=True)
x.drop('lang', axis=0, inplace=True)


# In[ ]:


x


# In[ ]:


def plotly_df(df, title=''):
    """Visualize all the dataframe columns as line plots."""
    common_kw = dict(x=df.index, mode='lines')
    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    iplot(fig, show_link=False)


# In[ ]:


plotly_df(x, 'Total behavior')


# In[ ]:


# TO use prophet is necessary 2 columns ds (with date ) and y with information
df = x.reset_index()
df.columns = ['ds', 'y']
df.tail(n=3)


# In[ ]:


# ds MUST be date
df['ds'] = pd.to_datetime(df['ds']).dt.date


# In[ ]:


# to try, last month will be predict 

prediction_size = 30
train_df = df[:-prediction_size]
train_df.tail(n=3)


# In[ ]:


# It's time to predict with prophet 

m = Prophet()
m.fit(train_df)
future = m.make_future_dataframe(periods=prediction_size)
forecast = m.predict(future)


# In[ ]:


forecast.tail(n=3)


# In[ ]:


m.plot(forecast)


# In[ ]:


m.plot_components(forecast)


# In[ ]:





# In[ ]:


# minimizing outliers effects

outliers = []

for i in range(0,len(df)-1):
    if (df['y'][i] > forecast['yhat_upper'][i]):
        outliers.append({'info':'max', 'index':i, 'date':df['ds'][i] ,'val':df['y'][i], 'forecast val': forecast['yhat_upper'][i], 'factor': df['y'][i]/forecast['yhat'][i] })
    if (df['y'][i] < forecast['yhat_lower'][i]):
        outliers.append({'info':'min', 'index':i, 'date':df['ds'][i] ,'val':df['y'][i], 'forecast val': forecast['yhat_lower'][i], 'factor': df['y'][i]/forecast['yhat'][i]})
        
outliers = pd.DataFrame(outliers)


# In[ ]:


df_new = df.copy()
df_new


# In[ ]:


for i in range (0, len(outliers)-1 ):    
    df_new['y'][outliers['index'][i]] = df_new['y'][outliers['index'][i]] / outliers['factor'][i]


# In[ ]:


# Prophet AGAIN with NEW df

# ds MUST be date
df_new['ds'] = pd.to_datetime(df_new['ds']).dt.date


# In[ ]:


# to try, last month will be predict 

prediction_size = 30
train_df = df_new[:-prediction_size]
train_df.tail(n=3)


# In[ ]:


# It's time to predict with prophet 

m = Prophet()
m.fit(train_df)
future = m.make_future_dataframe(periods=prediction_size)
forecast = m.predict(future)


# In[ ]:


m.plot(forecast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Function to find page language

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page) # search text like mask
    if res:
        return res[0][0:2] # 2 first letters
    return 'na'

t1['lang'] = t1.Page.map(get_language) # new collumn
# see: map () relation between series = https://www.geeksforgeeks.org/python-pandas-map/


# In[ ]:


plt.figure(figsize=(12, 6))
plt.title("Number of sites by languages", fontsize="18")
t1['lang'].value_counts().plot.bar(rot=0);


# In[ ]:


t1.values[0]


# In[ ]:


t1.T.index.values


# In[ ]:





# In[ ]:





# In[ ]:


t1['mean'] = 


# In[ ]:



#str_ = '2NE1_zh.wikipedia.org_all-access_spider'
str_[0:str_.find('.wikipedia.org')-3]

def get_subject(page):
    res = page[0:page.find('.wikipedia.org')-3]
    if res:
        return res
    return 'na'


# In[ ]:


t1['subject'] = t1.Page.map(get_subject) # new collumn


# In[ ]:


t1.Page


# In[ ]:


#Split datasets in languages

lang_sets = {}
lang_sets['en'] = t1[t1.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = t1[t1.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = t1[t1.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = t1[t1.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = t1[t1.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = t1[t1.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = t1[t1.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = t1[t1.lang=='es'].iloc[:,0:-1]


# In[ ]:


# access means for languages
sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]


# In[ ]:


days = [r for r in range(sums['en'].shape[0])]


# In[ ]:


################################ TESTS 



#re.search('{}.wikipedia.org','2NE1_zh.wikipedia.org_all-access_spider')
#[0][0:2]
# t1.Page.map(get_language)
# lang_sets
#lang_sets['es']
#len(lang_sets)
#japan = pd.DataFrame()
#japan['date_']= ja.index
#japan['count_'] = list(sums['ja'])
#japan
#pd.DataFrame(sums['ja'])

#pd.DataFrame(sums['ja']).index



# lang_sets['en'].iloc[:,1:].sum(axis=0)
#lang_sets['en'].shape[0]


# [](http://)

# In[ ]:




