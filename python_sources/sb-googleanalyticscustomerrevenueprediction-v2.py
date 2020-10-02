#!/usr/bin/env python
# coding: utf-8

# **Google Analytics Customer Revenue Prediction Contest**
# 
# **-----  !!ALL ZEROS!! -----**
# 
# * [http://www.kaggle.com/c/ga-customer-revenue-prediction](http://www.kaggle.com/c/ga-customer-revenue-prediction)
# 
# Created by Steve Black
# 
# Started on Oct.05.2018
# 
# This is a second try. 
# 
# My first try was about exploring how to connect to data and exploring what it is. 
# 
# The goal of this second Kernel-Submission is simple. I'm going to submit an answer where ALL users have zero revenue!
# 
# Why?
# * I believe the majority of users have zero revenue, thus predicting zero is acutally most accurate for most users.
# * 9,996 out of 714, 167 or 1.40% of users have revenue in the Train data set
# * This 'zero-answer' will then just test the error for this most basic case.
# * Of course, it should not win any prizes, but will be a great baseline.
# 
# Results:
# * My RMS score after submitting is : 1.7804
# * This serves as a great baseline... If you don't do better than this, then your better off guessing zeros!
# 
# HA! after submitting I can see many others with this exact same score... so my zeros idea is hardly unique!
# Great thinkers, think alike?

# In[ ]:


# ----- CREATED UPON CREATION OF KAGGLE PYTHON KERNEL -----
# ----- KEPT AS IS -----
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


# Load data using BigQueryHelper:
import bq_helper
print("Loading Training Data Set...")
ga_bq_train = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
print("Loading Test Data Set... ")
ga_bq_test = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_test_set")
print("Data Loaded: done")


# In[ ]:


# Test query estimate
queryy =     """
    SELECT  fullVisitorId
    FROM `kaggle-public-datasets.ga_test_set.ga_sessions_*` 
    GROUP BY fullVisitorId
    ORDER BY fullVisitorId
    """
print("size = " + str( ga_bq_test.estimate_query_size(queryy) * 1000 ) + " MB" )


# In[ ]:


# Test query final
print('starting queryy...')
test_as_pandas_data = ga_bq_test.query_to_pandas_safe(queryy)
print('done')


# In[ ]:


test_as_pandas_data.head(5)


# In[ ]:


# The predicted output needs to be the natural log of the sum + 1
# which is all zeros as ln(1) = 0
test_as_pandas_data['PredictedLogRevenue'] = np.log(0+1)
test_as_pandas_data.head(5)


# In[ ]:


test_as_pandas_data.describe()


# In[ ]:


test_as_pandas_data.info()


# In[ ]:


# Let's save it:
print('saving csv file...')
test_as_pandas_data.to_csv('submission_SteveBlack_v002_date_2018_1005_0959.csv', index = False)
print('done')

