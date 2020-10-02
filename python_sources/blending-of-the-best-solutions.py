#!/usr/bin/env python
# coding: utf-8

# Let's blend good and stable solutions for this competition

# https://www.kaggle.com/gkoundry/bayesian-logistic-regression-with-pystan/output <br/>
# https://www.kaggle.com/featureblind/robust-lasso-patches-with-rfe-gs/output<br/>
# https://www.kaggle.com/aantonova/851-logistic-regression/output<br/>
# https://www.kaggle.com/melondonkey/bayesian-spike-and-slab-in-pymc3<br/>

# # Import libraries and files

# In[ ]:


# import libraries
import pandas as pd

# load in the submissions
sub1 = pd.read_csv('../input/overfitting-dataset/submission1.csv')
sub2 = pd.read_csv('../input/overfitting-dataset/submission2.csv')
sub3 = pd.read_csv('../input/overfitting-dataset/submission3.csv')
sub4 = pd.read_csv('../input/overfitting-dataset/submission4.csv')


# # Blend

# Let's use the same coefficients for all files because we don't want overfit to public LB.

# In[ ]:


# create blend of submissions
submission = pd.DataFrame()
submission['id'] = sub1['id']
submission['target'] = 0.25*sub1['target']+0.25*sub2['target']+0.25*sub3['target']+0.25*sub4['target']


# # Submission

# In[ ]:


submission.to_csv('submission.csv', index=False)

