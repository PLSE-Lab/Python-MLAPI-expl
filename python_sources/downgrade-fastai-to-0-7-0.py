#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################################
# IMPORTANT: Change the "Internet" settings in the right panel to "Internet Connected"
#########################################

get_ipython().system('pip install fastai==0.7.0 --no-deps')
# fastai depends also on an older version of torch
get_ipython().system('pip install torch==0.4.1 torchvision==0.2.1')


# In[ ]:


# Importing a fastai 0.7.0 package now works!
from fastai.transforms import *

