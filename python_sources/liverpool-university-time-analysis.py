#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fastai.tabular import *
import gc
from tqdm import tqdm,tnrange
from math import *
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sns.set_style('darkgrid')


# In[ ]:


color_list= ['red','green','blue','gold','red','green','blue','gold']


# In[ ]:


df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
ts = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


ts.head()


# In[ ]:


ts.info()


# In[ ]:


plt.figure(1,figsize=(18,6))
sns.countplot(df.open_channels);


# In[ ]:



sns.pairplot(df.sample(frac=0.15,random_state=64),hue='open_channels',height=8)


# In[ ]:


df['frac_part'] = [abs(modf(x)[0]) for x in np.array(df.signal.values)]
df['int_part'] = [modf(x)[1] for x in np.array(df.signal.values)]


# In[ ]:


ts['frac_part'] = [abs(modf(x)[0]) for x in np.array(ts.signal.values)]
ts['int_part'] = [modf(x)[1] for x in np.array(ts.signal.values)]


# In[ ]:


df.head()


# # Group by open_channels

# In[ ]:


ch = df.groupby(['open_channels'])


# # Density plot of each open_channel

# In[ ]:


plt.figure(1,figsize=(20,8))
for i in range(11):
    sns.kdeplot(ch.get_group(i).signal,label=str(i),shade=True,legend=True)


# # Mean of each open_channel

# In[ ]:


ch_m=ch.mean()
ch_m.reset_index(inplace=True)
ch_m


# In[ ]:


plt.figure(1,figsize=(20,8))
plt.subplot(221)
sns.lineplot(data=ch_m,x='open_channels',y='int_part')
sns.scatterplot(data=ch_m,x='open_channels',y='int_part',label='int_part')
sns.lineplot(data=ch_m,x='open_channels',y='signal',)
sns.scatterplot(data=ch_m,x='open_channels',y='signal',label='signal')
plt.subplot(222)
sns.lineplot(data=ch_m,x='open_channels',y='frac_part')
sns.scatterplot(data=ch_m,x='open_channels',y='frac_part',label='frac_part')
plt.subplot(223)
sns.kdeplot(df.frac_part,shade=True,label='frac_part');


# # Scatter plot of a fraction of whole dataset

# In[ ]:


px.scatter(df.sample(frac=0.15,random_state=42),x='time',y='signal',color='open_channels',trendline='ols')


# # Signal of each open_channel

# In[ ]:


plt.figure(1,figsize=(20,20))
for i in range(4):
    plt.subplot(411+i)
    sns.lineplot(data=ch.get_group(i)[:15000],x='time',y='signal',color=color_list[i]);
    plt.title('open_channel '+str(i))


# In[ ]:


plt.figure(1,figsize=(20,20))
n=0
for i in range(4,8):
    plt.subplot(411+n)
    sns.lineplot(data=ch.get_group(i)[:15000],x='time',y='signal',color=color_list[i]);
    plt.title('open_channel '+str(i))
    n+=1


# In[ ]:


plt.figure(1,figsize=(20,18))
n=0
for i in range(8,11):
    plt.subplot(411+n)
    sns.lineplot(data=ch.get_group(i)[:15000],x='time',y='signal',color=color_list[i%8]);
    plt.title('open_channel '+str(i))
    n+=1


# # Rolling mean of window sizes [10,50,100,1000]

# In[ ]:


win_sz = [10, 50, 100, 1000]
for window in win_sz:
    df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
    df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()


# In[ ]:


mean_df = df.iloc[3200000:3210000]


# In[ ]:


col = ['rolling_mean_10', 'rolling_std_10', 'rolling_mean_50',
       'rolling_std_50', 'rolling_mean_100', 'rolling_std_100',
       'rolling_mean_1000', 'rolling_std_1000']


# In[ ]:



for i in range(0,len(col),2):
  plt.figure(1,figsize=(20,8))
  plt.subplot(211)
  sns.lineplot(y=mean_df[col[i]],x=list(range(3200000,3210000)),color=color_list[i])
  plt.title(col[i])
  plt.subplot(212)
  sns.lineplot(y=mean_df[col[i+1]],x=list(range(3200000,3210000)),color=color_list[i+1])
  plt.title(col[i+1])
  plt.show()


# In[ ]:





# # Open_channel distribution in each batch

# In[ ]:


plt.figure(1,figsize=(20,4))
n=0
for i in range(3):
  plt.subplot(131+n)
  data = df.iloc[(i * 500000):((i+1) * 500000 + 1)]
  sns.countplot(x=data.open_channels)
  plt.title('batch '+str(i))
  n+=1
plt.show()
plt.figure(2,figsize=(20,4))
n=0
for i in range(3,6):
  plt.subplot(131+n)
  data = df.iloc[(i * 500000):((i+1) * 500000 + 1)]
  sns.countplot(x=data.open_channels)
  plt.title('batch '+str(i))
  n+=1
plt.show()
plt.figure(3,figsize=(20,4))
n=0
for i in range(6,10):
  plt.subplot(141+n)
  data = df.iloc[(i * 500000):((i+1) * 500000 + 1)]
  sns.countplot(x=data.open_channels)
  plt.title('batch '+str(i))
  n+=1
plt.show()
  


# # open_channel distribution in each batch

# # batch 0

# In[ ]:


b = 0
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,4))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 1

# In[ ]:


b = 1
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,4))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # bacth 2

# In[ ]:


b = 2
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,5))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 3

# In[ ]:


b = 3
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,6))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set2')
plt.title('batch  :  '+str(b));


# # batch 4

# In[ ]:


b = 4
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,8))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 5

# In[ ]:


b = 5
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,8))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 6

# In[ ]:


b = 6
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,6))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 7

# In[ ]:


b = 7
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,6))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 8

# In[ ]:


b = 8
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,8))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# # batch 9

# In[ ]:


b = 9
data1 = df.iloc[(b * 500000):((b+1) * 500000 + 1)]
plt.figure(1,figsize=(20,8))
sns.scatterplot(data=data1.sample(frac=0.2,random_state=42), x='time',y='signal',hue='open_channels',palette='Set1')
plt.title('batch  :  '+str(b));


# In[ ]:




