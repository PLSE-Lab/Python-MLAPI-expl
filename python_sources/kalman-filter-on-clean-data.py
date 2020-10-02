#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# Clean data from Chris Deotte's dataset [Data Without Drift](https://www.kaggle.com/cdeotte/data-without-drift)

# In[ ]:


#Loading data
train_df = pd.read_csv('../input/data-without-drift/train_clean.csv')
test_df = pd.read_csv('../input/data-without-drift/test_clean.csv')


# Kalman filter from TJ Klein's notebook [A signal processing approach - Kalman Filtering](https://www.kaggle.com/teejmahal20/a-signal-processing-approach-kalman-filtering)

# In[ ]:


from pykalman import KalmanFilter

def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


# Kalman Filter
observation_covariance = .0015
train_df['signal'] = Kalman1D(train_df.signal.values,observation_covariance)
test_df['signal'] = Kalman1D(test_df.signal.values,observation_covariance)


# In[ ]:


train_df.to_csv("train.csv", index=False, float_format="%.4f")
test_df.to_csv("test.csv", index=False, float_format="%.4f")

