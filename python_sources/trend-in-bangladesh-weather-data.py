#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


# In[ ]:


data = pd.read_csv('../input/Temp_and_rain.csv')


# In[ ]:


data.head()


# ### Data column

# In[ ]:


data.columns


# ### Data Shape**

# In[ ]:


data.shape


# ### year

# In[ ]:


data['Year'].min(), data['Year'].max()


# In[ ]:


sns.lineplot(x="Year", y="tem",markers=True,lw=5, data=data);


# In[ ]:


df = data.groupby('Year').tem.mean()


# # Moving Average for Trend

# In[ ]:


def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])


# In[ ]:


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(20,10))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.values)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


# In[ ]:


plotMovingAverage(df, 2) 


# In[ ]:


plotMovingAverage(df, 4) 


# In[ ]:


plotMovingAverage(df, 8) 


# In[ ]:


plotMovingAverage(df, 12) 


# In[ ]:


plotMovingAverage(df, 15) 


# In[ ]:


plotMovingAverage(df, 8, plot_intervals=True,plot_anomalies=True)

