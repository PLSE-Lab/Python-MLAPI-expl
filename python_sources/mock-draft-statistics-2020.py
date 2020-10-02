#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
md = pd.read_csv('/kaggle/input/mock-drafts-2020-features/MockDraftStats.csv')
pd.set_option('display.max_rows', None)


# # Mock Draft Statistics 2020
# 
# The following numbers are descriptive statistics of 48 NFL mock drafts that I scrapped. The original dataset can be found here: [Dataset](https://www.kaggle.com/sherkt1/mock-drafts-2020)
# 
# Enjoy.

# In[ ]:


md = md.sort_values(by=['Mean'])
md

