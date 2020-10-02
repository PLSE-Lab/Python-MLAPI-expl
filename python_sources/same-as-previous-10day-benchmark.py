#!/usr/bin/env python
# coding: utf-8

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


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


days = env.get_prediction_days()


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    market_obs_df['returnsOpenPrevMktres10']=market_obs_df['returnsOpenPrevMktres10'].fillna(0)
    predictions_template_df['confidenceValue'] = np.tanh(market_obs_df['returnsOpenPrevMktres10'])
    env.predict(predictions_template_df)
print('Done!')


# In[ ]:


env.write_submission_file()


# In[ ]:




