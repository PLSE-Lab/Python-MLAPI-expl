#!/usr/bin/env python
# coding: utf-8

# The main objective for this kernel is an optimization of memory usage for the future kernels.
# Do not use this date for real kernels - it may break your score if data will be changed.

# In[ ]:


from kaggle.competitions import twosigmanews

# Read data but just once
if 'env' not in globals():
    env = twosigmanews.make_env()
    (market_df, news_df) = env.get_training_data()


# In[ ]:


market_df.to_pickle('market_df.pkl')
news_df.to_pickle('news_df.pkl')


# # How to use this data in your notebook?
# 1. Fork this script and run it
# 
# 2. Click **Add Data** on the right side of your kernel
# 
# 3. Choose **Kernel output files**  and **Your work** in the new window
# 
# 4. Find **Data preparation / Memory optimization** and click **Add**

# After downloading use the following code to read train data:
#     
#     import pandas as pd
#     
#     market_df = pd.read_pickle('../input/data-preparation-memory-optimization/market_df.pkl')
#     news_df = pd.read_pickle('../input/data-preparation-memory-optimization/news_df.pkl')

# In[ ]:




