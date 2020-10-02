#!/usr/bin/env python
# coding: utf-8

# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# ---
# 

# # Introduction
# 
# In this exercise, you will develop a baseline model for predicting if a customer will buy an app after clicking on an ad. With this baseline model, you'll be able to see how your feature engineering and selection efforts improve the model's performance.

# In[ ]:


# # Set up code checking
# from learntools.core import binder
# binder.bind(globals())
# from learntools.feature_engineering.ex1 import *

# import pandas as pd

# # click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv',
# #                          parse_dates=['click_time'])
# # click_data.head(10)


# In[ ]:


import pandas as pd
ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched']).drop("ID",axis=1)
print(ks.shape)
ks.head(10)


# In[ ]:


# Drop live projects
ks = ks.query('state != "live"')
# ks = ks.query('state != "undefined"')


# In[ ]:


# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))


# In[ ]:


ks.to_csv("featEng_kickstarterProj_v1.csv.gz",index=False,compression="gzip")

