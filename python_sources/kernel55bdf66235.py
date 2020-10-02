#!/usr/bin/env python
# coding: utf-8

# # Show or No Show? Visualization and Prediction
# A beginner's attempt at using logistic regression to determine whether or not a patient is going to show up to an appointment.
# 
# ## Questions to Consider
# - What kinds of patients are we working with?
# - Depending on a patient's characteristics, will he or she show up or not?
# - What factors correlate the most to whether or not a patient shows up?
# - If it's possible to accurately predict who the no-shows are to an appointment, what incentives can we offer these people to encourage them to show up to appointments?

# **Let's get started!**
# ## Check out the data
# We've been able to get some data from your neighbor for housing prices as a csv set, let's get our environment ready with the libraries we'll need and then import the data!
# ### Import Libraries

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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


USAhousing = pd.read_csv('../input/public-house/USA_Housing.csv')


# In[ ]:


USAhousing.head()

