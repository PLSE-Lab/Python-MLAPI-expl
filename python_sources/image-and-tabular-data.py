#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


image_data=pd.read_csv('../input/melanomaimage-and-tabular-data/imagedata.csv')
tabular_data=pd.read_csv('../input/melanomaimage-and-tabular-data/submission (1).csv')


# In[ ]:


sub = image_data.copy()

sub.target = 0.9 * image_data.target.values + 0.1 * tabular_data.target.values

sub.to_csv('submission.csv',index=False)


# In[ ]:


sub.head()

