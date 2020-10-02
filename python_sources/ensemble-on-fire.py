#!/usr/bin/env python
# coding: utf-8

# Credits:
# datasets:
# <br>
# https://www.kaggle.com/stocks/ieeesubmissions4
# <br>
# and kernals:
# <br>
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt
# <br>
# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
# <br>
# https://www.kaggle.com/andrew60909/lgb-starter-r
# <br>
# https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


sub1 = pd.read_csv('../input/ensemble/ensemble/All_Blends_9430.csv')
sub2 = pd.read_csv('../input/ensemble/ensemble/stack_median_9427.csv')
sub3 = pd.read_csv('../input/ensemble/ensemble/submission_IEEE_9417.csv')
sub4 = pd.read_csv('../input/lgb-single-model-lb-0-9419/ieee_cis_fraud_detection_v2.csv')
sub5 = pd.read_csv('../input/lgb-starter-r/sub.csv')
sub6 = pd.read_csv('../input/lgbm-baseline-small-fe-no-blend/lgb_sub.csv')
temp=pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

temp['isFraud'] =  0.22 * sub3['isFraud'] + 0.23 * sub4['isFraud'] + 0.27 * sub5['isFraud'] + 0.28 * sub6['isFraud']
temp.to_csv('submission.csv', index=False )

