#!/usr/bin/env python
# coding: utf-8

# How about the performance of BERT?  
# I made a little changes to the official code, feed the raw data and got the LB score 0.91216  
# the repo is here https://github.com/EliasCai/bert-toxicity-classification  
# there is a lot of word to do  
# you can train you own model and upload the prediction
# 

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

submission = pd.read_csv('../input/bertpred3/sub.csv')

print(submission.head())
submission.to_csv('submission.csv', index=False)

