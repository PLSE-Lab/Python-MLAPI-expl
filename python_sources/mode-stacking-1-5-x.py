#!/usr/bin/env python
# coding: utf-8

# ## Stacking the Best Models by Mode
# 
# Mode = the most frequently observed data value. <br>
# If two submissions "vote" for the same result, then the result is correct. Probably.  <br>
# Or the both submissions are wrong ;)
# 
# <pre><b>
# This Kernel shows how the scores can be improved using Stacking Method.
# Credit Goes to the following kernels
# ref:
# https://www.kaggle.com/roydatascience/cellular-stacking-1-5
# 

# In[ ]:


import os
import numpy as np 
import pandas as pd 
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import glob

all_files = glob.glob("../input/cellstack/*.csv")
all_files


# In[ ]:


outs = [pd.read_csv((f), index_col=0)['sirna'].values for f in all_files]
collected = np.array(outs)

# getting the mode
m = stats.mode(collected)[0][0]

submission = pd.read_csv('../input/recursion-cellular-image-classification/sample_submission.csv')
submission['sirna'] = m
submission.to_csv('ModeStacker.csv', index=False)

