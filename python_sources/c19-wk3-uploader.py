#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

filename = 'submission_c19_wk3_FINAL_MODEL-agg_dff-50-low_joined_UPDATED_PS.csv'


sub = pd.read_csv('/kaggle/input/c19week3/{}'.format(filename))
sub.to_csv('submission.csv', index=False)


# In[ ]:


print(sub[sub.ForecastId == 11438])
print(sub[sub.ForecastId == 7625])


# In[ ]:


print(sub.head())
print(sub.tail())


# In[ ]:




