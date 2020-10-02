#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()


# First, let's filter market data by previous 10 day market adjusted return over 300% for open and close price. There are huge values on the open side but not the close side. The closed price and return seem to be correct and there is something wrong with the open price and return.

# In[ ]:


market_train_df[abs(market_train_df['returnsOpenPrevMktres10'])>=3]


# In[ ]:


market_train_df[abs(market_train_df['returnsClosePrevMktres10'])>=3]


# We can see that there are many assets those intra-day "price drop" is more than 50%. Of course, some of the open price does not make sense.

# In[ ]:


market_train_df[abs(market_train_df['close']/market_train_df['open'])<=0.5]


# We can look at a particular asset "EXH.N". Pay particular attention to the 21 days window around open price 0.01 and 999.99.
# 
# For 0.01 as and example,  'returnsOpenPrevRaw1' on next day and 'returnsOpenPrevRaw10' on the 10th day is very high. I am not sure how market adjusted returns are calculated, but it seems like there is a propagation of error lasting well into the futurn (there could be other assets with incorrect entries on those days and if the market return is a calculated field based on these entries, there could be error as well). Notice 'returnsOpenPrevMktres10' has the same value 'returnsOpenNextMktres10' 11 rows ahead.

# In[ ]:


market_train_df[market_train_df['assetCode'] == "EXH.N"][['time', 'open','returnsOpenPrevRaw1','returnsOpenPrevRaw10','returnsOpenPrevMktres1', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10',
                                                         'close','returnsClosePrevRaw1','returnsClosePrevRaw10','returnsClosePrevMktres1', 'returnsClosePrevMktres10']].head(50)


# Since the competition is to predict the open price return, I hope there is no error in recording open future price or otherwise the score will be significantly impacted.
