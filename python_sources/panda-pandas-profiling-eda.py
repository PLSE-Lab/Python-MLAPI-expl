#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas_profiling import ProfileReport


# In[ ]:


train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


profile = ProfileReport(train, title='Train Profiling Report')
# profile = ProfileReport(train, title='Titanic Train Minimal Profiling Report', minimal=True)


# In[ ]:


profile.to_file(output_file="ProfileReport_train.html")


# In[ ]:


profile.to_widgets()


# In[ ]:


profile.to_notebook_iframe()


# In[ ]:


test = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')


# In[ ]:


profile_test = ProfileReport(test, title='Test Profiling Report')
# profile_test = ProfileReport(train, title='Titanic Test Minimal Profiling Report', minimal=True)


# In[ ]:


profile_test.to_file(output_file="ProfileReport_test.html")


# In[ ]:


profile_test.to_widgets()


# In[ ]:


profile_test.to_notebook_iframe()


# In[ ]:




