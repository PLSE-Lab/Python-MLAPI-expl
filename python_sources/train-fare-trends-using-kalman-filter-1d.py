#!/usr/bin/env python
# coding: utf-8

# # Kalman Filter 1D for time series trend / smoothing

# Often time, the data that we want to observe and analyze have a lot of noise, just like the price flucutation above. 
# 
# There are various smoothing and filtering techniques available. Let me start with some techniques that is not well suited for the purpose above:
# * Seasonal Decomposition: In some cases, the noise doesn't follow seasonal pattern
# * Moving Average: It is late to detect any dip, because it takes average of past N observations
# 
# Instead, Kalman Filter would do a great job for our purpose here. There are some good intro videos [here](http://https://www.youtube.com/watch?v=mwn8xhgNpFY), [here](http://)https://www.youtube.com/watch?v=CaCcOwJPytQ for an overview. 
# 
# In this notebook, I would apply 1D Kalman Filter on prices historical data for each route. Sneak peek of the output is here:
# ![image](https://i.ibb.co/PrNNx7Q/Screen-Hunter-3145.jpg)
# 
# Most famous Kalman Filter usage is for GPS location detection, which is 2D in nature. Python library for kalman filter is in pykalman, and it is generic for >1D. Below is an example of doing Kalman Filter in 1D, which is used to solve time series smoothing purposes.
# 
# *Some Illustrations*
# <img src="https://marumatchbox.com/wp-content/uploads/2017/07/kalman-1024x564.png" alt="Drawing" style="width: 400px;"/>
# > Source: https://marumatchbox.com/blog/taming-tracking-data-using-the-kalman-filter-to-improve-reliability-of-tracking/
# 
# <img src="https://www.researchgate.net/profile/Tarun_Vatwani/publication/311647948/figure/fig1/AS:439395720404992@1481771511852/Multi-Dimensional-Kalman-Filter.png" alt="Drawing" style="width: 400px;"/>
# > Source: https://www.researchgate.net/figure/Multi-Dimensional-Kalman-Filter_fig1_311647948

# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
from plotly.offline import (download_plotlyjs, 
                            init_notebook_mode, 
                            plot, 
                            iplot)
from plotly import io as pio


# Thanks to the following notebooks for code starters and ideas:
# * https://www.kaggle.com/dcanones/exploratory-data-analysis
# * https://www.kaggle.com/mjella45/spanish-high-speed-rail-tickets-pricing

# In[ ]:


renfe = pd.read_csv('../input/renfe.csv', parse_dates=['insert_date', 'start_date', 'end_date'])


# In[ ]:


renfe.head()


# # Quick EDA with Plotly Express

# In[ ]:


import plotly_express as px


# # Preprocessing: Extract product dimensions

# In[ ]:


renfe.loc[:, 'start_date_weekday'] = renfe['start_date'].dt.weekday
renfe.loc[:, 'start_date_time'] = renfe['start_date'].dt.strftime("%H:%M")
renfe.loc[:, 'train_id'] = renfe_index_hash = renfe[['origin', 
                                                     'destination', 
                                                     'start_date_weekday',
                                                     'start_date_time',
                                                     'train_type']] \
                           .apply(lambda x: hash(tuple(x.tolist())), axis=1)
renfe.loc[:,'origin-destination'] = renfe['origin'] + '-' + renfe['destination']


# In[ ]:


renfe.loc[:, 'start_date_day'] = renfe['start_date'].dt.strftime("%D")


# In[ ]:


renfe.head()


# # Datamart: Create groupby

# In[ ]:


renfe_agg1 = renfe.groupby(['origin-destination','start_date_day']).agg({'price': {'mean_price': 'mean'}})


# In[ ]:


renfe_agg1 = renfe_agg1.reset_index()
renfe_agg1.columns = renfe_agg1.columns.droplevel(1)
print(renfe_agg1.shape)
renfe_agg1.head()


# In[ ]:


renfe_agg1.plot(x='start_date_day',y='price')


# In[ ]:


fig = px.line(renfe_agg1,x='start_date_day',y='price',line_group='origin-destination',color='origin-destination')
fig.update(layout=dict(title="All OD routes in a single chart - you can click the legend to select 1 route"))


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go

odpairs = renfe_agg1['origin-destination'].unique()
N = len(odpairs)
fig, ax = plt.subplots((N//2), 2, figsize=(15,9))
for i, od in enumerate(odpairs):
    dfplot = renfe_agg1[renfe_agg1['origin-destination']==od]
#     ax[i//2,i%2] = sns.lineplot(data=dfplot, x='start_date_day', y='price')
    sns.lineplot(data=dfplot, x='start_date_day', y='price', ax=ax[i//2,i%2])
    ax[i//2,i%2].title.set_text(od)
plt.suptitle('Historical price for each of the OD route',y=1.1,size=18)
fig.tight_layout()
plt.show()


# In[ ]:


price_dict = {}
for od in odpairs:
    price_dict[od] = renfe_agg1[renfe_agg1['origin-destination']==od].iloc[:,1:]


# # Kalman Filter

# Let's run the Kalman Filter function itself. First we import pykalman. And then, we can run the functions below.

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

def Kalman1D_plot(observations,damping=1):
    # To return the plot
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
    plt.plot(pred_state)
    plt.plot(observations,linestyle='--',color='grey',linewidth=0.5)


# In[ ]:


fig, ax = plt.subplots((N//2), 2, figsize=(15,12))
for i, od in enumerate(odpairs):
    observations = price_dict[od]['price'].fillna(method='bfill')
    smoothed = Kalman1D(observations.values)
    ax[i//2,i%2].plot(smoothed)
    ax[i//2,i%2].plot(observations.values,linestyle='--',color='grey',linewidth=0.5)
    ax[i//2,i%2].title.set_text(od)
plt.suptitle('Smoothed Trend overlaid on historical prices for each of the OD route',y=1.1,size=18)
plt.tight_layout()
    


# ### Voila! Now you have smoothed trend, and you can clearly see the trend

# The default damping parameter is 1.0. Let's try to use weaker damping. We will see a bit more fluctuation

# In[ ]:


fig, ax = plt.subplots((N//2), 2, figsize=(15,12))
for i, od in enumerate(odpairs):
    observations = price_dict[od]['price'].fillna(method='bfill')
    smoothed = Kalman1D(observations.values,0.5)
    ax[i//2,i%2].plot(smoothed)
    ax[i//2,i%2].plot(observations.values,linestyle='--',color='grey',linewidth=0.2)
    ax[i//2,i%2].title.set_text(od)
plt.suptitle('Smoothed Trend - now with low damping / smoothing factor',y=1.1,size=18)
plt.tight_layout()

