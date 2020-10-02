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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.text import *


# In[ ]:


from pathlib import Path


# In[ ]:


folder_path = Path("/kaggle/input/fake-and-real-news-dataset")
os.listdir(folder_path)


# In[ ]:


folder_path.ls()


# In[ ]:


true_data = pd.read_csv(folder_path/'True.csv')
fake_data = pd.read_csv(folder_path/'Fake.csv')


# In[ ]:


true_data.head()


# In[ ]:


fake_data.head()


# So now you have pandas dataframe with the subset of true and fake news

# In[ ]:


print("Shape of the true data df is: ", true_data.shape)
print("Shape of the fake data df is: ", fake_data.shape)


# 21,000+ --> TRUE
# 
# 23,000+ --> FAKE
# 
# ~45,000 total. So 20% for validation would be ~9,000 samples
# 

# Now let's add a column to label fake vs real before combining the frames

# In[ ]:


true_data = true_data.assign(is_fake=0);
fake_data = fake_data.assign(is_fake=1);


# In[ ]:


true_data.head()


# In[ ]:


fake_data.head()


# In[ ]:


full_data = true_data.append(fake_data)


# In[ ]:


full_data.head()


# In[ ]:


full_data.shape


# In[ ]:


data = (TextList.from_df(df=full_data, path=folder_path, cols=1)
       .split_by_rand_pct(0.2)
       .label_from_df(cols=4)
       .databunch())


# let's run two models (before utilizing a pretrained language model) between title and text to see if one does better -- who knows if they will do anything of value without ULMFiT applied

# In[ ]:


data.show_batch()


# In[ ]:


learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.5)


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


data_title = (TextList.from_df(df=full_data, path=folder_path, cols=0)
       .split_by_rand_pct(0.2)
       .label_from_df(cols=4)
       .databunch())


# In[ ]:


learn = text_classifier_learner(data_title, AWD_LSTM, drop_mult=0.5,
                                model_dir='/tmp/models')


# In[ ]:


learn.fit_one_cycle(1)


# * At first glance -- the full text has MUCH better predictive power
