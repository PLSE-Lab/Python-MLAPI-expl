#!/usr/bin/env python
# coding: utf-8

# #              <font color=green>Vote</font> <font color=orange>Early and Vote</font> <font color=green>Often!</font>
# 

# **Since this is a blend of stacks and other blends(?), I would like to acknowledge many kernels that have contributed to this truly amazying blend. Below are the refs (also see refs inside those kernels).**
# 
# ## **<font color=green>Enjoy!</font>**
# 
# **References: (let me know if I forgot somebody)**
# 
# ['Stacking Higher and Higher' with fresh stack of models and refs](https://www.kaggle.com/stocks/stacking-higher-and-higher)
# 
# [My kernel](https://www.kaggle.com/stocks/under-sample-with-multiple-runs).
# 
# [IEEE - LGB + Bayesian opt.](https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt)
# 
# [Stackers Blend](https://www.kaggle.com/rajwardhanshinde/stackers-blend-top-4)
# 
# ['Stacking?'](https://www.kaggle.com/rajwardhanshinde/stacking)
# #### Info from the parent of this clone:
# >* Based on https://www.kaggle.com/lpachuong/statstack
# >* Thanks to <br>
# https://www.kaggle.com/jazivxt/safe-box<br>
# https://www.kaggle.com/artgor/eda-and-models<br>
# https://www.kaggle.com/stocks/under-sample-with-multiple-runs<br>
# https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb

# In[ ]:


import os
import numpy as np 
import pandas as pd 


# In[ ]:


sub_path = "../input/ieeesubmissions4"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


all_blends = pd.read_csv(os.path.join(sub_path, 'All_Blends_9430.csv'), index_col=0)
stack_median = pd.read_csv(os.path.join(sub_path, 'stack_median_9424.csv'), index_col=0)
sub = stack_median.copy()
sub.head()
sub['isFraud'] = stack_median['isFraud'].values*0.156+all_blends['isFraud'].values*0.844
sub['isFraud'].head()
sub.head()
sub.to_csv('blend_of_blends.csv', float_format='%.6f')

