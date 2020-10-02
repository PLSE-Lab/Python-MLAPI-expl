#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


sub1 = pd.read_csv('../input/classification-densenet201-efficientnetb7/submission.csv')
sub2 = pd.read_csv('../input/tf-zoo-models-on-tpu/submission.csv')
sub3 = pd.read_csv('../input/fork-of-plant-2020-tpu-915e9c/submission.csv')
sub4 = pd.read_csv('../input/plant-pathology-2020-in-pytorch-0-979-score/submission.csv')

# To to get a higher score you can change weights or add new submissions.
submissions = [sub1, sub2, sub3, sub4]
sub_weights = [0.35, 0.10, 0.15, 0.40]

print(sum(sub_weights))


# In[ ]:


sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
sub.iloc[:, 1:] = np.average([df.iloc[:, 1:].values for df in submissions], weights=sub_weights, axis=0)


# In[ ]:


# https://www.kaggle.com/c/plant-pathology-2020-fgvc7/discussion/149845
# Try changing alpha, it may lead to a higher score.
def LabelSmoothing(encodings , alpha=0.01):
    K = encodings.shape[1]
    y_ls = (1 - alpha) * encodings + alpha / K
    return y_ls


preds = sub.iloc[:, 1:].values
LabelSmoothing(preds, alpha=0.01)
sub.iloc[:, 1:] = preds


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# ### Please upvote if you like it.
# ### Feel free to leave comments below!
