#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # *Sample Submission*

# In[ ]:


df = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')


# In[ ]:


df.head(10)


# In[ ]:


df.to_csv('submission.csv', index = False)


# # *Data Leakage Exercise*

# Most people find target leakage very tricky until they've thought about it for a long time.
# 
# So, before trying to think about leakage in the housing price example, we'll go through a few examples in other applications. 
# 
# Things will feel more familiar once you come back to a question about house prices.

# reference: https://www.kaggle.com/alexisbcook/data-leakage

# In[ ]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex7 import *
print("Setup Complete")


# ### *1. The Data Science of Shoelaces*

# Nike has hired you as a data science consultant to help them save money on shoe materials. 
# 
# Your first assignment is to review a model one of their employees built to predict how many shoelaces they'll need each month. 
# 
# The features going into the machine learning model include:
# - The current month (January, February, etc)
# - Advertising expenditures in the previous month
# - Various macroeconomic features (like the unemployment rate) as of the beginning of the current month
# - The amount of leather they ended up using in the current month
# 
# The results show the model is almost perfectly accurate if you include the feature about how much leather they used. 
# 
# But it is only moderately accurate if you leave that feature out. 
# 
# You realize this is because the amount of leather they use is a perfect indicator of how many shoes they produce, which in turn tells you how many shoelaces they need.
# 
# Do you think the _leather used_ feature constitutes a source of data leakage? 
# 
# If your answer is "it depends," what does it depend on?
# 
# After you have thought about your answer, check it against the solution below.

# In[ ]:


q_1.solution()


# ### *2. Return of the Shoelaces*

# You have a new idea. 
# 
# You could use the amount of leather Nike ordered (rather than the amount they actually used) leading up to a given month as a predictor in your shoelace model.
# 
# Does this change your answer about whether there is a leakage problem? 
# 
# If you answer "it depends," what does it depend on?

# In[ ]:


q_2.solution()


# ### *3. Getting Rich With Cryptocurrencies?*

# You saved Nike so much money that they gave you a bonus. 
# 
# Congratulations.
# 
# Your friend, who is also a data scientist, says he has built a model that will let you turn your bonus into millions of dollars. 
# 
# Specifically, his model predicts the price of a new cryptocurrency (like Bitcoin, but a newer one) one day ahead of the moment of prediction. 
# 
# His plan is to purchase the cryptocurrency whenever the model says the price of the currency (in dollars) is about to go up.
# 
# The most important features in his model are:
# - Current price of the currency
# - Amount of the currency sold in the last 24 hours
# - Change in the currency price in the last 24 hours
# - Change in the currency price in the last 1 hour
# - Number of new tweets in the last 24 hours that mention the currency
# 
# The value of the cryptocurrency in dollars has fluctuated up and down by over \$100 in the last year, and yet his model's average error is less than \$1. 
# 
# He says this is proof his model is accurate, and you should invest with him, buying the currency whenever the model says it is about to go up.
# 
# Is he right? 
# 
# If there is a problem with his model, what is it?

# In[ ]:


q_3.solution()


# ### *4. Preventing Infections*

# An agency that provides healthcare wants to predict which patients from a rare surgery are at risk of infection, so it can alert the nurses to be especially careful when following up with those patients.
# 
# You want to build a model. 
# 
# Each row in the modeling dataset will be a single patient who received the surgery, and the prediction target will be whether they got an infection.
# 
# Some surgeons may do the procedure in a manner that raises or lowers the risk of infection. 
# 
# But how can you best incorporate the surgeon information into the model?
# 
# You have a clever idea. 
# 1. Take all surgeries by each surgeon and calculate the infection rate among those surgeons.
# 2. For each patient in the data, find out who the surgeon was and plug in that surgeon's average infection rate as a feature.
# 
# Does this pose any target leakage issues?
# 
# Does it pose any train-test contamination issues?

# In[ ]:


q_4.solution()


# ### *5. Housing Prices*

# You will build a model to predict housing prices.  
# 
# The model will be deployed on an ongoing basis, to predict the price of a new house when a description is added to a website.  
# 
# Here are four features that could be used as predictors.
# 1. Size of the house (in square meters)
# 2. Average sales price of homes in the same neighborhood
# 3. Latitude and longitude of the house
# 4. Whether the house has a basement
# 
# You have historic data to train and validate the model.
# 
# Which of the features is most likely to be a source of leakage?

# In[ ]:


q_5.hint()


# In[ ]:


q_5.solution()


# ### *Conclusion*

# Leakage is a hard and subtle issue. 
# 
# You should be proud if you picked up on the issues in these examples.
# 
# There is still a lot of room to build knowledge and experience. 
# 
# Try out 'ASHRAE - Great Energy Predictor III': https://www.kaggle.com/c/ashrae-energy-prediction 
# 
# Again, Congratulations!
