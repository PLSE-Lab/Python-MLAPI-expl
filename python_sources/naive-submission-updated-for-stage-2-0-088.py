#!/usr/bin/env python
# coding: utf-8

# **updated for Stage 2**
# 
# tl;dr 
# - tuning table might have similar distribution of labels to submission.csv
# - tuning table is part of submission.csv

# # Predictions

# In[ ]:


import pandas as pd

top_n = 3

tuning_labels = pd.read_csv('../input/tuning_labels.csv', names=['id', 'labels'], index_col=['id'])

# calculate top_n most popular labels
predicted = ' '.join(
    tuning_labels['labels']
    .str
    .split()
    .apply(pd.Series)
    .stack()
    .value_counts()
    .head(top_n)
    .index
    .tolist()
)

print(f'{top_n} most popular labels are: {predicted}')


# # Stage 1

# In[ ]:


submission = pd.read_csv('../input/stage_1_sample_submission.csv', index_col='image_id')

# tuning table is part of submission.csv
submission.index.isin(tuning_labels.index).sum()


# In[ ]:


# use most popular labels as a prediction unless the correct labels are provided
submission['labels'] = predicted
submission.update(tuning_labels)


# In[ ]:


submission.to_csv('naive.csv')


# # Stage 2
# one-liner update for Stage 2

# In[ ]:


(
    pd.read_csv('https://storage.googleapis.com/inclusive-images-stage-2/stage_2_sample_submission.csv', index_col='image_id')
    .assign(labels=predicted)
    .to_csv('stage_2_submission.csv')
)

