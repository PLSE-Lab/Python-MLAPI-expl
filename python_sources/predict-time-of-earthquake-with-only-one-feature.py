#!/usr/bin/env python
# coding: utf-8

# # Predict time of earthquake with only one feature

# This kernel aims at finding a solution that makes sense by using basic models and features.
# I find it useful to understand roughly how the 'acoustic_data' signal is behaving.
# We are using here the standard deviation of 'acoustic_data' as the only predictive feature.

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's check the format of the train data.

# In[ ]:


df_train_10M = pd.read_csv('../input/train.csv', nrows=200000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
df_train_10M.head()


# As the submission consists of samples of 150,000 data points, we divide the train set into samples of 150,000 data points as well. 
# For each of these small samples, we calculate the standard deviation of the input data "acoustic_data".
# Let's assume that the "time_to_failure" can be considered as nearly constant within 150,000 data points, we calculate the mean of "time_to_failure" for each of these smaller samples.

# In[ ]:


df_train_10M["index_obs"] = (df_train_10M.index.astype(float)/150000).astype(int)


# In[ ]:


train_set = df_train_10M.groupby('index_obs').agg({'acoustic_data': 'std', 'time_to_failure': 'mean'})


# In[ ]:


train_set.columns = ['acoustic_data_std', 'time_to_failure_mean']
train_set.head()


# We delete the original training dataset to free up some memory.

# In[ ]:


del df_train_10M


# We check at which indices the earthquakes happen. These are the indices when the difference between 2 'time_to_failure' is positive.

# In[ ]:


train_set[train_set['time_to_failure_mean'].diff() > 0].head()


# We smooth out a bit the STD of "acoustic_data" using rolling mean.

# In[ ]:


train_set["acoustic_data_transform"] = train_set["acoustic_data_std"].clip(-20, 12).rolling(10, min_periods=1).median()


# We plot the STD of 'acoustic_data' over time, the 3 red vertical lines correspond to the 3 first earthquakes.
# At first glance, you can see that the (smoothed) STD of 'acoustic_data' approximately increases linearly until the time at which the earthquakes happen.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(20,12))
ax2 = ax1.twinx()
ax1.plot(train_set["acoustic_data_transform"])
plt.axvline(x=38, color='r')
plt.axvline(x=334, color='r')
plt.axvline(x=697, color='r')
plt.title('Smoothed standard deviation of acoustic_data vs. time', size=20)
plt.show()


# Following our first intuition, if "acoustic_data_std" is a linear function of  "time_to_failure_mean", as we can see on the graph, then "time_to_failure_mean" is also a linear function of "acoustic_data_std".
# 
# Let's use a simple linear regressor to predict the "time_to_failure_mean" from the smooth version of "acoustic_data_std".

# In[ ]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(train_set[["acoustic_data_transform"]], train_set["time_to_failure_mean"])

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)


# Let's predict the results on test set and submit the results.

# In[ ]:


submission_file = pd.read_csv('../input/sample_submission.csv')
submission_file.head()


# In[ ]:


for index, seg_id in enumerate(submission_file['seg_id']):
    seg = pd.read_csv('../input/test/' + str(seg_id) + '.csv')
    x = seg['acoustic_data'].values
    std_x = max(-20, min(12, np.std(x)))
    submission_file.loc[index, "time_to_failure"] = max(0, regr.intercept_ + regr.coef_ * std_x)
    del seg


# In[ ]:


submission_file.to_csv('submission.csv', index=False)


# In[ ]:




