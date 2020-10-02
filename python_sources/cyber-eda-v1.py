#!/usr/bin/env python
# coding: utf-8

# # This is a work under progress stay tuned for upcoming versions
# ![Enter the wolves](http://www.gstatic.com/tv/thumb/v22vodart/9991602/p9991602_v_v8_ab.jpg)

# In[ ]:


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


# In[ ]:


# official way to get the data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


os.listdir("../input")


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.dtypes


# In[ ]:


market_train_df.head()


# In[ ]:


news_train_df.head()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(market_train_df)


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("white")


# In[ ]:


sns.distplot(market_train_df.volume)


# In[ ]:


sns.distplot(market_train_df.open)


# In[ ]:


sns.distplot(market_train_df.close)


# In[ ]:


market_train_df.describe()


# In[ ]:


sns.distplot(market_train_df.returnsClosePrevRaw1)


# In[ ]:


sns.distplot(market_train_df.returnsOpenPrevRaw1)


# In[ ]:


sns.distplot(market_train_df.universe)


# In[ ]:


sns.distplot(market_train_df.returnsOpenNextMktres10)


# In[ ]:


sns.lineplot(x="time", y="open",
             hue="universe",
             data=market_train_df)


# In[ ]:


sns.lineplot(x="time", y="returnsOpenNextMktres10",
             hue="universe",
             data=market_train_df)


# In[ ]:


days = env.get_prediction_days()


# In[ ]:


print(days)


# In[ ]:


dir(env)


# In[ ]:


filtered_ = market_train_df.loc[market_train_df['universe'] == 1.0 ]


# In[ ]:


filtered_.head()


# In[ ]:


filtered_.shape

