#!/usr/bin/env python
# coding: utf-8

# Update: June.18.2020  
# 
# As in [this discussion](https://www.kaggle.com/c/birdsong-recognition/discussion/159123#890189) or [host comments](https://www.kaggle.com/c/birdsong-recognition/discussion/159123#890675), `background` column or `secondary_labels` column in train.csv have multi-label information.  
# I think both columns are almost same, after processing like below I did here. But using `secondary_labels` will be preferable based on host's comment. (Here I stick to using `background`)
# 
# [host](https://www.kaggle.com/stefankahl) comments:
# > Overlapping vocalizations are a major issue and Xeno-canto recordings may or may not contain background species and they may or may not have an appropriate label (typically primary and secondary labels in the metadata). 
# 
# I'm not sure using secondary labels makes our score better or not, but for curiosity I made an dataframe for multi-label task. Please let me know, if I'm wrong.

# In[ ]:


import warnings, re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')


# In[ ]:


train = pd.read_csv('../input/birdsong-recognition/train.csv')
print(train.shape)
train.head(3)


# In[ ]:


train['background'].value_counts(dropna=True, sort=True).to_frame().head(20).plot.bar(
    color='deeppink', figsize=(15, 5)
);


# Make a neet function to extract multi label information from train.background column.

# In[ ]:


def get_multi_label(s, bird_l):
    if type(s) != str: s = str(s)
    return [b in re.sub(r' \([^()]*\)', '', s).split('; ') for b in bird_l]


# In[ ]:


bird_l = train.species.unique().tolist()


# In[ ]:


# default label
label_df = pd.get_dummies(train.species).set_index(train.xc_id)
label_df.head(3)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'background_arr = np.array([get_multi_label(r, bird_l) for r in train.background])\nbackground_arr.shape')


# In[ ]:


label_df.iloc[:, :] = label_df.values + background_arr
label_df['label_n'] = label_df.sum(axis=1)


# In[ ]:


label_df.label_n.value_counts().plot.bar(figsize=(10, 5), color='deeppink');


# In[ ]:


label_df.drop('label_n', axis=1).to_csv('multi-label.csv')


# As a result, almost of all instances are still mono-labeled. But one third of all train data is multi-labeled.
# 
# One strategy may be, first we train our models with mono-labeled data, then fine-tune with multi-labeled data.
# I'm not sure this may be good or not. But hope this information will help you.  
# 
# Happy Kaggling!!!
# 
# <img src="https://storage.googleapis.com/kaggle-avatars/images/2080166-kg.png" width=100 align='left'>
