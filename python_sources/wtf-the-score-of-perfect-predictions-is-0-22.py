#!/usr/bin/env python
# coding: utf-8

# The score of the perfect classifier is just 0.22. It means that in this competition our goal is to get not a perfect model but profitable and low volatile model. 
# 
# Below is the code getting the score of the perfect classifier. If you have found a mistake in the code please comment about it. 

# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kaggle.competitions import twosigmanews


# In[ ]:


env = twosigmanews.make_env()


# In[ ]:


TARGET_COLUMN = "returnsOpenNextMktres10"
is_toy = False


# # Get the data

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# # Calculate the score of perfect classifier 

# In[ ]:


market_train_df["time"] = market_train_df["time"].dt.date
market_train_df.set_index(["time", "assetCode"], inplace=True)


# In[ ]:


labels = market_train_df[TARGET_COLUMN]
market_train_df.drop([TARGET_COLUMN], axis=1, inplace=True)


# In[ ]:


def calculate_perfect_scores(market_data, labels):
    predictions = labels.copy()
    predictions = predictions > 0
    predictions = predictions.astype(int)
    predictions[predictions == 0] = -1
    return_from_every_prediction = (labels * predictions * market_data["universe"]).groupby("time").sum()
    score = return_from_every_prediction.mean() / return_from_every_prediction.std()
    
    return score, return_from_every_prediction


# In[ ]:


score, return_from_every_prediction = calculate_perfect_scores(market_train_df, labels)


# In[ ]:


print("Score with perfect prediction", score)


# In[ ]:


values = return_from_every_prediction.values

plt.plot(values);

