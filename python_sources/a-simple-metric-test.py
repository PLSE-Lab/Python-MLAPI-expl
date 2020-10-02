#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel is a test to extract some simple but interesting conclusions about the metric function of this competition.

# In[ ]:


import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews


# ## Proposed metric function

# In[ ]:


def sigma_score(val_y, pred_y, time, univ):

    pred_y = 2.0 * np.clip(pred_y,0,1) - 1.0
    
    x_t = pd.DataFrame(data={'extra_time': time, 'val': pred_y * val_y * univ})
    x_t_sum = x_t.groupby(by='extra_time').sum()
    if x_t_sum.val.std() == 0:
        score = 0
    else:
        score = float(x_t_sum.val.mean() / x_t_sum.val.std())
    return score


# ## Maximum possible training score

# In[ ]:


env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df["newTarget"] = (market_train_df["returnsOpenNextMktres10"] > 0).astype(int)
score = sigma_score(market_train_df["returnsOpenNextMktres10"].values,market_train_df["newTarget"].values, market_train_df["time"].factorize()[0],market_train_df["universe"].values )
print(score)


# ## First test
# Random predictions

# In[ ]:


val_y = [0.1, 1.5, -5.6, 7.8, -3.3, -0.6]
pred_y = np.random.rand(len(val_y))
time = [1,1,2,2,3,3]
univ = [1,1,1,1,1,1]
score = sigma_score(val_y, pred_y, time, univ)
print(score)


# ## Second test
# Predict perfectly the greatest return:

# In[ ]:


pred_y = [0,0,0,1,0,0]
score = sigma_score(val_y, pred_y, time, univ)
print(score)


# ### Conclusions
# Predicting perfectly the greatest return value seems to be really important!

# ## Third test
# Predict all correctly but with small confident values:

# In[ ]:


pred_y = [0.1,0.1,-0.1,0.1,-0.1,-0.1]
score = sigma_score(val_y, pred_y, time, univ)
print(score)


# ### Conclusions
# Predicting all correctly but with a small confident value seems to be a really bad scenario.

# # Final conclusions
# From this simple tests, it seems that predicting perfectly the greatest return value is more important that predicting all correctly but with a small confident value!
