#!/usr/bin/env python
# coding: utf-8

# <pre><b>Credits to the Experts (Please like their kernels)
# https://www.kaggle.com/roydatascience/aggregating-the-light-gbm-models-0-9460
# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
# </b></pre>

# **UPVOTE WILL BE APPRECIATED! THANK YOU!

# In[ ]:


from scipy.stats import rankdata

LABELS = ["isFraud"]


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/lgmodels'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


predict_list = []
predict_list.append(pd.read_csv("../input/stsckerrr2/AggStacker.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/stsckerrr/submission (4).csv")[LABELS].values)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(1):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('AggStacker.csv', index=False)


# In[ ]:


submission.head()


# AggStacker.csv is the best score (0.9471)
