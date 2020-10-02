#!/usr/bin/env python
# coding: utf-8

# # About
# 
# This kernel was built for you in case you're running your work elsewhere and want to try out your submissions directly.

# <font size=4 color='red'> If you find this kernel useful, please don't forget to upvote. Thank you. </font>

# # Reminder
# 
# Please note that in order to be eligible for this competition prizes, your submission can't be generated from this kernel but from machine learning code fully run on TPUs for both training and inference. That code must be made public following the end of the competition. Prizes are subject to fulfillment of Winners Obligations as specified in the competition's rules.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


submission = pd.read_csv('/kaggle/input/104-flowers-submissions/0.96523.csv')
submission


# In[ ]:


submission.to_csv('submission.csv', index=False)

