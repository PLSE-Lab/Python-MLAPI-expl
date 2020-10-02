#!/usr/bin/env python
# coding: utf-8

# I'm seeing a lot of models using variables like week of year and month to train their models, and so I build this kernel to show what variables of date can be used and what variables don't add to the model.

# In[8]:


import numpy
import pandas 

import os
print(os.listdir("../input"))


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train = pandas.read_csv('../input/train.csv')
test = pandas.read_csv('../input/test.csv')


# In[7]:


for i in [train,test]:
    i['activation_date'] = pandas.to_datetime(i['activation_date'])
    i['month'] = i['activation_date'].dt.month
    i['day'] = i['activation_date'].dt.day
    i['day_of_week'] = i['activation_date'].dt.dayofweek
    i['weekofyear'] = i['activation_date'].dt.weekofyear


# We can see that we have the same month in train and test, so there is no need of using this variable in the model.
# 
# The day variable have a small overlap, but we have to be cautious when using it because this diference will not apear on our cross-validation.
# 
# Day of week is the only variable that I see using in models, cause we have a full overlap in train and test.
# 
# In week of year we have 0 overlap. If we build our models with week of year, we may see improvment on our cross-validation, but not on the test set
# 

# In[10]:


### Thanks SRK for this script for venn plots https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito

from matplotlib_venn import venn2

for i in ['month','day','day_of_week','weekofyear']:

    plt.figure(figsize=(10,7))
    venn2([set(train[i].unique()), set(test[i].unique())], set_labels = ('Train set', 'Test set') )
    plt.title("Number of " + i + " in train and test", fontsize=15)
    plt.show()


# In[ ]:




