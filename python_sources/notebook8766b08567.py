#!/usr/bin/env python
# coding: utf-8

# KazAnova has shared a leak (https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31870 ) based on the timestamp of the forlders containing images.  This notebooks shows some reason why this leak provides useful informaiton to decision tree based methods.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
### Seaborn style
sns.set_style("whitegrid")


# Loading data and merging

# In[ ]:


train_data = pd.read_json('../input/train.json')


listing_image_time = pd.read_csv('../data/listing_image_time.csv')
listing_image_time.columns = ['listing_id', 'image_time']
listing_image_time.head()


# In[ ]:




