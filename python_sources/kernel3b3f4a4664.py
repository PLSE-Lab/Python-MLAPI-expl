#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sub1=pd.read_csv('/kaggle/input/tensorflow-hub-inception-resnet-v2-submission/submission.csv')
sub1.head()


# In[ ]:


sub=sub1
sub.to_csv('submission.csv',index=None)

