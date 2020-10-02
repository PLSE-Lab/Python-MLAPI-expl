#!/usr/bin/env python
# coding: utf-8

# **On the initial basis, I conducted some tests and got good results.**

# In[ ]:


import numpy as np 
import pandas as pd 
import os
os.listdir('../input/')


# In[ ]:


sub1 = pd.read_csv('../input/average-efficientnet/submission.csv')
sub1.head()


# In[ ]:


sub2 = pd.read_csv('../input/classification-densenet201-efficientnetb7/submission.csv')
sub2.head()


# In[ ]:


sub3 = pd.read_csv('../input/tf-zoo-models-on-tpu/submission.csv')
sub3.head()


# In[ ]:


sub4 = pd.read_csv('../input/fork-of-plant-2020-tpu-915e9c/submission.csv')
sub4.head()


# In[ ]:


sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
sub.head()


# In[ ]:


sub.healthy = ( sub1.healthy  + sub3.healthy + sub4.healthy)/3


# In[ ]:


sub.multiple_diseases = (sub1.multiple_diseases  + sub3.multiple_diseases + sub4.multiple_diseases)/3


# In[ ]:


sub.rust = (sub1.rust  + sub3.rust + sub4.rust)/3


# In[ ]:


sub.scab = (sub1.scab  + sub3.scab + sub4.scab)/3


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)

