#!/usr/bin/env python
# coding: utf-8

# # Content
# 1. Introduction
# 2. Dataset description
# 3. Exponential Smoothing algorithm theory
# 4. Exponential Smoothing implementation
# 5. Forecast evaluation
# 6. Conclusion
# ***

# ## 1. Introduction
# A smoothing method reduces the effects of random variations that the deterministic components of a time series can have. With the help of smoothing methods we can also forecast new observations for a given time series. The algorithms that use smooting methods can forecast data for time series that have got or haven't got a trend. If an algorithm using smooting methods is designed to forecast an observation on a time series that has a trend, we should NOT use that algorithm to forecast a time series that does not have a trend and vice versa.
# [Here](https://www.kaggle.com/andreicosma/introduction-in-time-series-moving-average) you cand find a beginner friendly tutorial to introduce you into time series and smoothing methods.
# ***

# ## 2. Dataset description
# Our time series observations that will be used in this example are the number of acres burned in forest fires in Canada over a period of 70 years. We can see below how our time series looks like:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_dataset = pd.read_csv("../input/number_of_acres_burned_in_forest.csv")
csv_dataset.plot()
plt.show()


# As we can see, the data hasn't got a trend and it's composed of 70 observations. In this example we will forecast the next observation.
# ***

# ## 3. Exponential Smoothing algorithm theory
# This algorithm helps us to forecast new observations based on a time series. This algorithm uses smoothing methods. The [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) algorithm is used only on time series that DON'T have a trend. Exponential smoothing is based on the use of [window functions](https://en.wikipedia.org/wiki/Window_function) to smooth time series data. The mathematical model for this algorithm is:
# ![](https://image.ibb.co/bDPazx/exponential_smoothing.png)
# 
# From this mathematical model it results:
# ![](https://image.ibb.co/m85hex/es.png)
# 
# where alpha is the smoothing constant and it can take values between 0 and 1, ,,P_t'' is the forecast at time ,,t'', X_t is the time series observation at time ,,t''.
# In this case we have to solve two problems:
# 1. Choosing the P_1 value;
# 2. Choosing the alpha value.
# 
# For the first problem we have two common options:
# 1. Choosing the P_1 value to be equal to the first observation;
# 2. Choosing the P_1 value to be equal to the arithmetic mean of the first four or five observations.
# 
# To solve the second problem we can use the ,,incremental method'': We set the value of alpha (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), then we calculate the coresponding [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) for each value that alpha takes. After that, we choose the alpha with the minimum MSE.
# ***
# 

# ## 4. Exponential Smoothing implementation
# The following section will propose an algorithm for finding the best alpha. The algorithm will start at alpha = 0.1 and will go up to alpha = 0.9. For each alpha, the algorithm will forecast the already known observations along with the correspondent MSE followed by choosing the alpha with the minimum MSE value.

# In[ ]:


optimal_alpha = None
best_mse = None
db = csv_dataset.iloc[:, :].values.astype('float32')
mean_results_for_all_possible_alpha_values = np.zeros(9)
for alpha in range(0, 9):
    pt = np.mean(db[:, 0][0:5])
    mean_for_alpha = np.zeros(len(db))
    mean_for_alpha[0] = np.power(db[0][0] - pt, 2)
    for i in range(1, len(db)):
        pt = pt + ((alpha + 1) * 0.1) * (db[i - 1][0] - pt)
        mean_for_alpha[i] = np.power(db[i][0] - pt, 2)
    mean_results_for_all_possible_alpha_values[alpha] = np.mean(mean_for_alpha)
optimal_alpha = (np.argmin(mean_results_for_all_possible_alpha_values) + 1) * 0.1
best_mse = np.min(mean_results_for_all_possible_alpha_values)
print("Best MSE = %s" % best_mse)
print("Optimal alpha = %s" % optimal_alpha)


# After the optimal ,,alpha'' has been found, we can forecast the t+1 observation as following:

# In[ ]:


pt = np.mean(db[:, 0][0:5])
for i in range(1, len(db) + 1):
    pt = pt + optimal_alpha * (db[i - 1][0] - pt)
print("Next observation = %s" % pt)


# ***
# ## 5. Forecast evaluation
# In this section we will compare the forecast data with the real data for the optimal ,,alpha''.

# In[ ]:


forecast = np.zeros(len(db) + 1)
pt = np.mean(db[:, 0][0:5])
forecast[0] = pt
for i in range(1, len(db) + 1):
    pt = pt + optimal_alpha * (db[i - 1][0] - pt)
    forecast[i] = pt
plt.plot(db[:, 0],label = 'real data')
plt.plot(forecast, label = 'forecast')
plt.legend()
plt.show()


# Exponential smoothing is a [rule of thumb](https://en.wikipedia.org/wiki/Rule_of_thumb) technique for smoothing time series data using the exponential window function.
# ***

# ## 6. Conclusion
# In this notebook it was shown how the ,,Exponential Smoothing" algorithm forecasts based on the smoothing constant ,,alpha''. Moreover, it was presented an implementation of how you can find the optimal ,,alpha''. When dealing with time series, multiple algorithms should be tested to find out which of them gives the minimum MSE. The algorithm with the minimum MSE should be used for further forecasts on that time series.
# 
# For more tutorials check out my [kernels](https://www.kaggle.com/andreicosma/kernels) page.
# 
# I hope this notebook was helpful and don't forget: Never give up if things turn out to be difficult.
# 
# References: Smaranda Belciug - Time Series course.
