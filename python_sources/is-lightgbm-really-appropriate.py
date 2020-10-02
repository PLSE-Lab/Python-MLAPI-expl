#!/usr/bin/env python
# coding: utf-8

# # Is LightGBM really appropriate?
# by @marketnetural
# 
# LightGBM is the go-to model for Kagglers when dealing with structured data. Given the nature of the data in this competition, I have some doubts about the appropriateness of using LightGBM (or XGBoost, or any out-of-the-box scikit-learn model). As most public kernels use LightGBM, I would love to hear community thoughts on this.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


plt.rcParams['figure.figsize'] = 14, 8


# In[ ]:


# Make environment and get data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()


#  I was hoping to make a more thorough kernel to explore this but ran out of time. In lieu of that, I thought I would just post some thoughts here.
#  
#  I suspect that LightGBM and other out-of-the-box ML models are not appropriate for this competition and will just lead to overly optimistic models... Some LGB model feature importance plots show high importance for features like `assetCode`, `month`, etc. This is a huge red flag as there is no economic reason (all other things being equal) why this should be. However, that's **not** what I am worried about here. Let's assume you remove features like that, there is still a problem.
#  
#  Generally, LightGBM et al. assume the data are "identically and independently distributed" (IID). It is somewhat anathema for Kagglers to worry about such things, but I think it is very important here (the "I" for "independently" being key). One important aspect of this competition is that the target labels **are overlapping in time** :
#  
#  - on t=0, the target label is the return from t+1 to t+11
#  - on t=1, the target label is the return from t+2 to t+12;
#  - as such, the target labels for t=0 and t=1 overlap with 9 daily returns.
#  
#  
#  Let's look at the (rank) correlation of the raw target labels vs those labels shifted by 1 day and 5 days. What do we see?

# In[ ]:


from scipy.stats import spearmanr

def sp(group, col1_name, col2_name):
    x = group[col1_name]
    y = group[col2_name]
    return spearmanr(x, y)[0]


# In[ ]:


market_train_df['target_shift_1'] = (
    market_train_df.
    groupby('assetCode')['returnsOpenNextMktres10'].
    shift(-1)
)

market_train_df['target_shift_5'] = (
    market_train_df.
    groupby('assetCode')['returnsOpenNextMktres10'].
    shift(-5)
)


# In[ ]:


rc_1 = (
    market_train_df.
    sort_values(['time', 'assetCode']).
    dropna().
    groupby('time').
    apply(sp, 'returnsOpenNextMktres10', 'target_shift_1')
)

rc_5 = (
    market_train_df.
    sort_values(['time', 'assetCode']).
    dropna().
    groupby('time').
    apply(sp, 'returnsOpenNextMktres10', 'target_shift_5')
)


# In[ ]:


rc_1.plot()
rc_5.plot(title='Rank Correlation Between Target and Target Shifted: 1 Day and 5 Days');


# From the plot we can see that there is significant correlation between the targets, not just at 1 day, but even at 5 days shift. Why does this matter? Well, from the original Brieman paper, [*Random Forests*](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf):
# 
# > Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled **independently** and with the **same distribution** for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The **generalization error** of a forest of tree classifiers depends on the strength of the individual trees in the forest **and the correlation between them.** [my emphasis added]
# 
# Of course LightGBM is not simply a Random Forest model, however Brieman's insight is applicable to all ensembling techniques: the errors of the individual models must be uncorrelated; uncorrelated errors diversity away leaving the true signal. If the errors are correlated (becuase the target labels are correlated), then you are not really gaining much true incremental value at each boosting round. I suspect that this is one reason why many have reported difficulty in model validation versus Public Leaderboard scores. 
# 
# Anyhow, thanks for considering this idea and **please do leave comments.**
# 
# *This post was inspired by Chapter 4, "Sample Weights", of Advances in Financial Machine Learning by Lopez de Prado.*
# 
# 
