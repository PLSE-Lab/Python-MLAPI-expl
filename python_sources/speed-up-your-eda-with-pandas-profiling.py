#!/usr/bin/env python
# coding: utf-8

# # Speed up your EDA with Pandas Profiling

# ## What is Pandas Profiling?

# Pandas profiling allows you to generate an **entire EDA** with a few lines of code. 
# 
# Basically, pandas profiling allows you to view all sorts of things that would take a lot of time, so why not use the simplest way?

# The only drawback is that it takes 30 seconds to generate one profile report.

# In[ ]:


import pandas as pd
import pandas_profiling
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


gender_submission.profile_report()


# In[ ]:


train.profile_report()


# In[ ]:


test.profile_report()

