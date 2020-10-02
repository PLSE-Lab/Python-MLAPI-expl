#!/usr/bin/env python
# coding: utf-8

# # The purpose of this kenel
# 
# First, don't be afraid. This kernel only contain less than 20 lines of code
# 
# The purpose of this kernel is to show a **benchmark** about the score of a random guess. This Kernel doesn't contain any modelling, I just randomly guess the output, so that so that you can know if your model is actually **"learning something"**.  If you can get a much higher score than this one, that means you are doing it correctly, congrat!

# ##  initial Environment

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# (market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


import random
import numpy as np


# ## generate output

# In[ ]:


def make_predictions(predictions_template_df, market_obs_df, news_obs_df):
    mu, sigma = 0,0.3
    pred = np.random.normal(mu, 0.3,len(predictions_template_df)) # generate result in Guassian Distribution
    pred = np.clip(pred,-1,1) # limit number in range -1 to 1
    predictions_template_df.confidenceValue = pred


# ## write output

# In[ ]:


days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(predictions_template_df, market_obs_df, news_obs_df)
    env.predict(predictions_template_df)
    
print('Done!')

env.write_submission_file()

