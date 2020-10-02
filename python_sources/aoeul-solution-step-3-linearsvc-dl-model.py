#!/usr/bin/env python
# coding: utf-8

# Combining results from [LinearSVC](https://www.kaggle.com/szelee/aoeul-solution-step-1-linearsvc) + [Deep Learning Model](https://www.kaggle.com/szelee/aoeul-solution-step-2-deep-learning-model) gives quite a decent score boost. 
# 
# - (0.45500 + 0.46533) -> 0.46814 (Public LB)
# - (0.45353 + 0.46393) -> 0.46673 (Private LB)
# 
# My actual final submissions gain a few extra points because of further fine tuning on the hyperparameters of the LinearSVC and DL models + some extra feature engineering on test data (matching unknown words to the  closest word in train data, finding exact matched train title's label in train data as prediction).

# In[ ]:


from pathlib import Path
import json
import re
import sys
import warnings
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook


# In[ ]:


dl_df = pd.read_csv('../input/aoeul-solution-step-2-deep-learning-model/Deep_Learning_predictions.csv')
svc_df = pd.read_csv('../input/aoeul-solution-step-1-linearsvc/LinearSVC_predictions.csv')


# In[ ]:


# deep learning model prediction with 2 predictions for each test data
dl_df.head(10)


# In[ ]:


# LinearSVC model prediction with 1 prediction for each test data
svc_df.head(10)


# In[ ]:


# sanity check
len(dl_df), len(svc_df)


# In[ ]:


# merging both dataframe
new_df = pd.merge(dl_df, svc_df, on='id')
new_df.head()


# In[ ]:


# Iterating through the results. 
# If both DL's first prediction == LinearSVC prediction, use DL's original prediction
# If not match, LinearSVC become first prediction, and DL's first become second.

new_tag = []
i = 0
for _, row in tqdm_notebook(new_df.iterrows()):
    x1, x2 = row.tagging_x.split(' ') 
    x1 = int(x1)
    x2 = int(x2)
    y = row.tagging_y
    
    if x1 == y:
        new_tag.append(row.tagging_x)
        i = i + 1
    else:
        new_tag.append(str(y)+ ' '+ str(x1))


# In[ ]:


# percentage of matches between DL and LinearSVC
i/len(new_df)*100 


# In[ ]:


# save to file
new_df['tagging']= np.array(new_tag)
new_df = new_df[['id', 'tagging']]
new_df.to_csv(f'Ensembled_SVC_DL_predictions.csv', index=None)

