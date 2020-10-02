#!/usr/bin/env python
# coding: utf-8

# I am sorry to see that there are more and more 1.000 in LB. Kaggle is a platform to help us learning from competitions and learning from other excellent competitors. It's not a difficult palyground competition. We can ensemble 3 top models in public kernels to get a great score without training (0.9781).

# In[ ]:


import os
import numpy as np
import pandas as pd
import scipy.special

sigmoid = lambda x:scipy.special.expit(x)


# In[ ]:


os.listdir('../input')


# In[ ]:


sub1 = pd.read_csv('../input/blending-power-0-9754/sub.csv')
sub2 = pd.read_csv('../input/tta-power-densenet169/submission_tta_64.csv')
sub3 = pd.read_csv('../input/you-really-need-attention-pytorch/sub_tta.csv')


# In[ ]:


sub1.head()


# In[ ]:


sub2.head()


# In[ ]:


sub3.head()


# In[ ]:


sub1.label = (sub1.label + sub2.label + sub3.label) / 3


# In[ ]:


sub1.to_csv('ensemble.csv', index=False)

