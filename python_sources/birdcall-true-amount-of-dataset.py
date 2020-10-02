#!/usr/bin/env python
# coding: utf-8

# # **True Amount of Dataset**
# Hi all. I'm new to kaggle. It's my first (public)notebook, please read and comment. 
# 
# we have almost same amount of dataset;100 files for each birds, but the duration is different.  The difference means "imbalanced data" for machine learnng . So, we should attention the "true" amount of data.
# 
# Here's my question: should we use additional dataset (like [these](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-n-z) [datas](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-a-m)--thanks to [@rohanrao](https://www.kaggle.com/rohanrao)) to balance the amount of data? 
# please comment your solution! thank you.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')


# In[ ]:


duration_list = [sum(train["duration"][train["ebird_code"]==bird_l]) for bird_l in train.ebird_code.unique().tolist()]


# In[ ]:


dur_dict = {"ebird_code":train.ebird_code.unique().tolist(),
               "sum_of_duration":duration_list}


# In[ ]:


duration_df = pd.DataFrame(dur_dict)
duration_df.head(10)


# In[ ]:


plt.figure(figsize=(15,4))
duration_df["sum_of_duration"].plot()
print("min:{} max:{} mean:{}".format(min(duration_list),max(duration_list),np.mean(duration_list).astype(int)))
plt.plot(np.arange(0,264),[np.mean(duration_list).astype(int)]*264)##### plot mean ######


# In[ ]:




