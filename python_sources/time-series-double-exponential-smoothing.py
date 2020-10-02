#!/usr/bin/env python
# coding: utf-8

# # Content
# 1. Introduction
# 2. Dataset description
# 3. Double Exponential Smoothing algorithm theory
# 4. Double Exponential Smoothing implementation
# 5. Forecast evaluation
# 6. Conclusion
# ***
# 

# ## 1. Introduction
# A smoothing method reduces the effects of random variations that the deterministic components of a time series can have. With the help of smoothing methods we can also forecast new observations for a given time series. The algorithms that use smooting methods can forecast data for time series that have got or haven't got a trend. If an algorithm using smooting methods is designed to forecast an observation on a time series that has a trend, we should NOT use that algorithm to forecast a time series that does not have a trend and vice versa. [Here](https://www.kaggle.com/andreicosma/introduction-in-time-series-moving-average) you can find a beginner friendly tutorial to introduce you into time series and smoothing methods.
# ***

# ## 2. Dataset description
# Our time series observations that will be used in this example are the numbers of shampoo sales over a period of 3 years. We can see below how our time series looks like:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_dataset = pd.read_csv("../input/sales_of_shampoo_over_a_three_ye.csv")
csv_dataset.plot()
plt.show()


# As we can see, the data HAS got an ascending trend and it's composed of 36 observations. In this example we will forecast the next observation.
# ***

# ## 3. Double Exponential Smoothing algorithm theory
# This algorithm helps us to forecast new observations based on a time series. This algorithm uses smoothing methods. The ,,Double Exponential Smoothing" algorithm is used only on time series that HAVE a trend. On time series that have a trend  the [,,Exponential Smoothing''](https://www.kaggle.com/andreicosma/time-series-exponential-smoothing) algorithm does not perform very well. [Here](https://www.kaggle.com/andreicosma/time-series-exponential-smoothing) you can learn about the exponential smoothing algorithm. This problem was solved by adding a second smoothing constant: ,,gamma". The mathematical model for this algorithm is:
# ![](https://preview.ibb.co/idX49x/des.png)

# where ,,alpha" and ,,gamma" are the smoothing constants and they can take values between 0 and 1, ,,P_t'' is the forecast at time ,,t'', X_t is the time series observation at time ,,t'', b_t is the trend value at time ,,t''. In this case we have to solve two problems:
# 1. Choosing the values of P_1 and b_1;
# 2. Choosing the values of ,,alpha" and ,,gamma".
# 
# For the first problem we can choose P_1 to be equal to the first observation and for b_1 we have two common options:
# 1. we can choose it to be the second observation minus the first (b_1 = x_2 - x_1);
# 2. we can choose it to be the last observation minus the first, all divided by the total number of observations minus 1 ( b_1 = (x_n - x_1)/(n - 1)).
# 
# To solve the second problem we can use the ,,incremental method'': We set the value of alpha (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) and gamma (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), then we calculate the coresponding [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) for each value that alpha and gamma take. After that, we choose the alpha and gamma with the minimum MSE.
# 
# After we got the optimal alpha and gamma and we have calibrated the trend with the formula presented above we can forecast the next ,,m'' periods of time using the following formula:
# ![](https://image.ibb.co/hvNJKx/dess.png)
# 
# ***

# 
# ## 4. Double Exponential Smoothing implementation
# The following section will propose an algorithm for finding the best alpha and gamma. The algorithm will start at alpha = 0.1 and gamma = 01 and will go up to gamma = 0.9 then incrementing the alpha to alpha = 0.9. For each alpha and beta, the algorithm will forecast the already known observations along with the correspondent MSE followed by choosing the alpha and beta with the minimum MSE value.
# 

# In[ ]:


optimal_alpha = None
optimal_gamma = None
best_mse = None
db = csv_dataset.iloc[:, :].values.astype('float32')
mean_results_for_all_possible_alpha_gamma_values = np.zeros((9, 9))
for gamma in range(0, 9):
    for alpha in range(0, 9):
        pt = db[0][0]
        bt = db[1][0] - db[0][0]
        mean_for_alpha_gamma = np.zeros(len(db))
        mean_for_alpha_gamma[0] = np.power(db[0][0] - pt, 2)
        for i in range(1, len(db)):
            temp_pt = ((alpha + 1) * 0.1) * db[i][0] + (1 - ((alpha + 1) * 0.1)) * (pt + bt)
            bt = ((gamma + 1) * 0.1) * (temp_pt - pt) + (1 - ((gamma + 1) * 0.1)) * bt
            pt = temp_pt
            mean_for_alpha_gamma[i] = np.power(db[i][0] - pt, 2)
        mean_results_for_all_possible_alpha_gamma_values[gamma][alpha] = np.mean(mean_for_alpha_gamma)
        optimal_gamma, optimal_alpha = np.unravel_index(
            np.argmin(mean_results_for_all_possible_alpha_gamma_values),
            np.shape(mean_results_for_all_possible_alpha_gamma_values))
optimal_alpha = (optimal_alpha + 1) * 0.1
optimal_gamma = (optimal_gamma + 1) * 0.1
best_mse = np.min(mean_results_for_all_possible_alpha_gamma_values)
print("Best MSE = %s" % best_mse)
print("Optimal alpha = %s" % optimal_alpha)
print("Optimal gamma = %s" % optimal_gamma)


# After the optimal ,,alpha'' and ,,gamma" have been found, we can calibrate the trend as following:

# In[ ]:


pt = db[0][0]
bt = db[1][0] - db[0][0]
for i in range(1, len(db)):
    temp_pt = optimal_alpha * db[i][0] + (1 - optimal_alpha) * (pt + bt)
    bt = optimal_gamma * (temp_pt - pt) + (1 - optimal_gamma) * bt
    pt = temp_pt
print("P_t = %s" % pt)
print("b_t = %s" % bt )


# Now we can forecast the next ,,m'' periods of time using the formula from section 3 like this:

# In[ ]:


print("Next observation = %s" % (pt + (1 * bt)))


# ***
# ## 5. Forecast evaluation
# In this section we will compare the forecast data with the real data for the optimal ,,alpha'' and ,,gamma''.

# In[ ]:


forecast = np.zeros(len(db) + 1)
pt = db[0][0]
bt = db[1][0] - db[0][0]
forecast[0] = pt
for i in range(1, len(db)):
    temp_pt = optimal_alpha * db[i][0] + (1 - optimal_alpha) * (pt + bt)
    bt = optimal_gamma * (temp_pt - pt) + (1 - optimal_gamma) * bt
    pt = temp_pt
    forecast[i] = pt
forecast[-1] = pt + (1 * bt)
plt.plot(db[:, 0],label = 'real data')
plt.plot(forecast, label = 'forecast')
plt.legend()
plt.show()


# As we can see above, the algorithm gives good results on time series that HAVE a trend.
# ***

# ## 6. Conclusion
# 
# In this notebook it was shown how the ,,Double Exponential Smoothing" algorithm forecasts based on the smoothing constant ,,alpha'' and ,,gamma". Moreover, it was presented an implementation of how you can find the optimal ,,alpha'' and ,,gamma". It was proven that the algorithm gives very good results on time series that have a trend. When dealing with time series, multiple algorithms should be tested to find out which of them gives the minimum MSE. The algorithm with the minimum MSE should be used for further forecasts on that time series.
# 
# For more tutorials check out my [kernels](https://www.kaggle.com/andreicosma/kernels) page.
# 
# I hope this notebook helped you in your studies. Keep doing what you like no matter what others think.
# 
# References: Smaranda Belciug - Time Series course.
# 
