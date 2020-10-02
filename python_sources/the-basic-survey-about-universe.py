#!/usr/bin/env python
# coding: utf-8

# # The basic survey about universe
# 
# 
# There is the description below about universe. (quotation from Data Description)
# ```
# universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. 
# This value is not provided outside of the training data time period. The trading universe on a given date is the 
# set of instruments that are avilable for trading (the scoring function will not consider instruments that are not 
# in the trading universe). The trading universe changes daily.
# ```
# 
# The result of the basic survey about universe:
# 
# * `market_train_df` has 4,072,956 records.
# * 2,423,150 records are `universe` = 1, and they are belong to 2,466 `assetCode`.
# * 3,780 `assetCode` are included in `market_train_df`. So, 35% of `assetCode` have `universe` = 0 records.
# * 698 `assetCode` have redords, all of which are `universe` = 1.
# * Vice versa, 1,314 `assetCode` have redords, all of which are `universe` = 0.
# * `market_train_df` has 2,498 days.
# * There are 522 `assetCode`, which has 2,498 records (full year).
# * **278** `assetCode` of 522 have all `universe` = 1 records.
# 
# 
# * Future data (test run result of `submission.csv`) has 1,157,953 records.
# * There are 2,458 `assetCode` in the period.
# * Future data has 639 days, and 1,329 `assetCode` are full year.
# * **269** `assetCode` of 1,329 are in **278**, which train records are all `universe` = 1 and full year.
# 
# 
# The unilization of this result for the analysis is next work.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

import tqdm
import os

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# ## market_train_df

# In[ ]:


market_train_df.head(1)


# In[ ]:


market_train_df.shape


# In[ ]:


market_train_df.query('universe ==1').shape


# In[ ]:


market_train_df.groupby('assetCode').count().shape


# In[ ]:


market_train_df.query('universe ==1').groupby('assetCode').count().shape


# In[ ]:


market_train_df.groupby('time').count().shape


# In[ ]:


market_train_df.query('universe ==1').groupby('time').count().shape


# In[ ]:


market_train_df_time_count = market_train_df.groupby('time').count()
market_train_df_time_count.head(1)


# In[ ]:


market_train_df_univ1_time_count = market_train_df.query('universe ==1').groupby('time').count()
market_train_df_univ1_time_count.head(1)


# In[ ]:


plt.figure(figsize=(20, 10))
plt.plot(market_train_df_time_count.assetCode, label='all')
plt.plot(market_train_df_univ1_time_count.assetCode, label='univ=1')
plt.title('market_train_df : count')
plt.legend()


# In[ ]:


#For example
market_train_df.query('assetCode == "AAPL.O" & universe ==1').shape


# In[ ]:


market_train_df_assetCode_mean = market_train_df.groupby('assetCode').mean()


# In[ ]:


plt.figure(figsize=(20, 5))
plt.scatter(np.arange(len(market_train_df_assetCode_mean.universe)), market_train_df_assetCode_mean.universe)
plt.title('market_train_df_assetCode : universe_mean')


# In[ ]:


market_train_df_assetCode_mean_univ1 = market_train_df_assetCode_mean.query('universe == 1')
market_train_df_assetCode_mean_univ1.shape


# In[ ]:


market_train_df_assetCode_mean_univ1.index


# In[ ]:


market_train_df_univ1all = market_train_df[market_train_df['assetCode'].isin(market_train_df_assetCode_mean_univ1.index)]
market_train_df_univ1all.shape


# In[ ]:


market_train_df_univ1all.groupby('assetCode').mean().shape


# In[ ]:


market_train_df_assetCode_mean_univ0 = market_train_df_assetCode_mean.query('universe == 0')
market_train_df_assetCode_mean_univ0.shape


# In[ ]:


# 2498days
market_train_df_assetCode_countall = market_train_df.groupby('assetCode').count().query('time == 2498')
market_train_df_assetCode_countall.shape


# In[ ]:


market_train_df_assetCode_countall_univ1all = market_train_df_assetCode_countall    [market_train_df_assetCode_countall.index.isin(market_train_df_assetCode_mean_univ1.index)]
market_train_df_assetCode_countall_univ1all.shape


# In[ ]:


# reference
market_train_df_assetCode_mean_univX = market_train_df_assetCode_mean.query('0 < universe < 1')


# In[ ]:


market_train_df_assetCode_countall_univ1X = market_train_df_assetCode_countall    [market_train_df_assetCode_countall.index.isin(market_train_df_assetCode_mean_univX.index)]
market_train_df_assetCode_countall_univ1X.shape


# In[ ]:


y = market_train_df[market_train_df['assetCode'].isin(market_train_df_assetCode_countall_univ1X.index)]    .groupby('time').sum().universe
plt.figure(figsize=(20, 10))
plt.plot(y)


# ## Test run

# In[ ]:


days = env.get_prediction_days()


# In[ ]:


#from tutorial
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')


# In[ ]:


env.write_submission_file()


# In[ ]:


print([filename for filename in os.listdir('.') if '.csv' in filename])


# ## Furetue data

# In[ ]:


test_sub_df = pd.read_csv('submission.csv')
test_sub_df.shape


# In[ ]:


test_sub_df.head(1)


# In[ ]:


test_sub_df.groupby('assetCode').count().shape


# In[ ]:


test_sub_df.groupby('time').count().shape


# In[ ]:


test_sub_df_assetCode_count = test_sub_df.groupby('assetCode').count()
test_sub_df_assetCode_countall = test_sub_df_assetCode_count.query('confidenceValue == 639')
test_sub_df_assetCode_countall.shape


# In[ ]:


test_sub_df_assetCode_countall_univ1all = test_sub_df_assetCode_countall[test_sub_df_assetCode_countall.    index.isin(market_train_df_assetCode_countall_univ1all.index)]
test_sub_df_assetCode_countall_univ1all.shape


# In[ ]:




