#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl test data
# 
# * This notebook is just for everyone to download test data more conveniently for offline develop model.
# 

# In[ ]:


from kaggle.competitions import nflrush
import pandas as pd

# You can only call make_env() once, so don't lose it!
env = nflrush.make_env()


# ### Training data is in the competition dataset as usual

# In[ ]:


iter_test = env.iter_test()
test = pd.DataFrame()
sample_prediction = pd.DataFrame()
for (test_df, sample_prediction_df) in iter_test:
    test = pd.concat([test_df,test])
    sample_prediction = pd.concat([sample_prediction_df,sample_prediction])
    env.predict(sample_prediction_df)
test.to_csv('test.csv',index=False)
sample_prediction.to_csv('sample_prediction.csv',index=False)


# In[ ]:


print(test.head())


# In[ ]:


print(sample_prediction.head())

