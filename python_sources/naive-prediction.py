#!/usr/bin/env python
# coding: utf-8

# # Naive prediction kernel
# 
# In this kernel, we will predict the returns for the next 10 days by taking the returns for the previous 10 days.

# In[ ]:


import numpy as np
import pandas as pd
import os
from kaggle.competitions import twosigmanews


# In[ ]:


# Create the environment
env = twosigmanews.make_env()


# In[ ]:


def naive_predict(market_obs_df, predictions_template_df):
    market_obs_df = market_obs_df.set_index('assetCode')
    predictions_template_df['confidenceValue'] = predictions_template_df.assetCode.apply(lambda x: market_obs_df.loc[x].returnsOpenPrevMktres10)
    
    # replace NAs with zeros
    predictions_template_df['confidenceValue'] = predictions_template_df['confidenceValue'].fillna(0.0)
    # there are supposed to be inf values in the data
    predictions_template_df['confidenceValue'] = predictions_template_df['confidenceValue'].replace([np.inf, -np.inf], 0.0)
    # clip values to the required range
    predictions_template_df['confidenceValue'] = predictions_template_df['confidenceValue'].clip(-1.0, 1.0)
    return predictions_template_df


# In[ ]:


days = env.get_prediction_days()


# In[ ]:


for (market_obs_df, _, predictions_template_df) in days:
    predictions_df = naive_predict(market_obs_df, predictions_template_df)
    env.predict(predictions_df)

env.write_submission_file()


# In[ ]:




