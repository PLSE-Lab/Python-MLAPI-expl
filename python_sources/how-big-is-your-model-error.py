#!/usr/bin/env python
# coding: utf-8

# # How Big is your Model Error? A comparison between RMSLE and MAPE
# 
# In this competition, the evaluation metric used (Root Mean Square Logarithmic Error) doesn't provide a good intuition about the magnitude of the error of the models. In this kernel, we will make a very simple experiment to put RMSLE in relation to MAPE (Mean Absolute Percentage Error).
# 
# ### The experiment
# For each value of the target variable (meter_reading), we will create a new value picked from a random normal distribution, centered in the original value. We will call this new set of values "y_noise". This will simulate the predictions of an unbiased model for our target variable. We will repeat this procedure many times, increasing the variance of the normal distribution at each iteration, and saving the values of RMSLE and MAPE obtained. This sampling procedure will allow us to create a map of the relationship between RMSLE and MAPE for our dataset.
# 
# #### * Note about MAPE
# The target variable in this dataset has a lot of zeros. The traditional implementation of MAPE can't be used when the ground truth has zeros, because you get an x/0 indetermination. However, there is a known alternative to solve this problem, consisting in replace each actual value of the series in the original formula by the average of all actual values of that series. This is the same as dividing the sum of absolute differences by the sum of actual values, and is sometimes referred to as WAPE (Weighted Absolute Percentage Error).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error


# ## Metrics definition

# In[ ]:


def RMSLE(y_true, y_pred):
    
    if len(y_pred[y_pred<0])>0:
        y_pred = np.clip(y_pred, 0, None)
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))




def MAPE(y_true, y_pred):
    
    if len(y_true[y_true==0])>0: # Use WAPE if there are zeros
        return sum(np.abs(y_true - y_pred)) / sum(y_true) * 100
        
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ## Load data

# In[ ]:


df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', usecols=['meter_reading'])


# ## The experiment

# In[ ]:


rmsle = list()
mape = list()

y = df['meter_reading'].values

for dev in np.arange(0, 1, 0.01):
    
    y_noise = np.random.normal(y, dev*y)
    
    rmsle.append(RMSLE(y, y_noise))
    mape.append(MAPE(y, y_noise))


# In[ ]:


df_results = pd.DataFrame([mape, rmsle], index=['MAPE', 'RMSLE']).T
df_results.sort_values('MAPE', inplace=True)


# ## Plot results

# In[ ]:


plt.figure(figsize=(16,16))
plt.plot(df_results['MAPE'].values, df_results['RMSLE'].values)
plt.xlabel('MAPE %')
plt.ylabel('RMSLE')
plt.grid(True, axis='both')
plt.show()


# ## Conclusions
# At the moment of writing this kernel, the leaderboard top 100 scores are in the range of 1.08 - 1.21 RMSLE. As we can see, this is equivalent to a MAPE in the 40% - 50% range. 
# 
# There is noise in the mapping obtained between the two metrics, as expected. The variance of that noise grows with the MAPE. This can be due to the non linearities we introduced in the metrics functions: for the RMSLE, we are clipping to 0 the negative y_noise values, while for the MAPE these values contribute to the error in their full magnitude (check the metrics code at the beginning of this kernel). At each iteration, the number of values clipped in the calculus of RMSLE grows randomly but not monotonically, giving as result the growing irregular pattern between MAPE and RMSLE observed. 
