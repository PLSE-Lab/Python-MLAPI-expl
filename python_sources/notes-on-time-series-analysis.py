#!/usr/bin/env python
# coding: utf-8

# There are many people who are looking for "valid" external data to add to their models when forecasting the oil price. I think, but I'd like to raise a few points to keep in mind. Financial instrument data, such as stock prices and oil prices, may be a unit root process, which is a kind of non-stationary process. There are many reasons why this data is very difficult to handle. The first reason for this is that it is a non-stationary process, and needless to say, it is not possible to make a stationary process It is incompatible with statistical modeling/machine learning as a premise. The second reason is that it falsely assumes that variables that are not valid are also valid. I'd like to explain the second point in this article. First, let's plot and check the data for a given crude oil.

# In[ ]:


import pandas as pd
wti = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-01-02_To2020-06-08.csv")
wti.plot()


# I'm going to do a regression analysis on this oil price using a "random walk" created by accumulating the random numbers displayed in the image below.
# The random walk literally moves randomly without any regularity, so it should not be a valid variable for predicting the price of oil.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(3)
randomwalk = np.random.normal(size=len(wti)).cumsum()
plt.show(pd.Series(randomwalk).plot())


# In[ ]:


import statsmodels.api as sm
X = randomwalk
X=sm.add_constant(X)
y = wti["Price"]
model = sm.OLS(y,X)
result = model.fit()
result.summary()


# However, the results of the analysis show that the coefficient and R-squared is quite high and the p-value is low, with the result that "random walks are a useful variable in predicting the price of oil".
# Thus, the result is that variables that are not inherently related to the data and are not valid for prediction have been led to be valid.This is a phenomenon called "spurious regression," which can occur when regression analysis is performed between unit root processes It is.      
# spurious regression example
# :https://www.tylervigen.com/spurious-correlations   
#   main reason for this phenomenon is that there is autocorrelation in the residuals, and when there is autocorrelation in the residuals, the validity of the least-squares estimator is lost.  
# **It is important to note that the modeling is done with variable selection while checking the R-squared and p-values of the external data you have obtained, and even if you get good results in the training period, there is a risk that it will have no predictive power at all in the future.**
# If you can run a regression analysis and predict stock prices without any data processing, anyone can do a billion You can be a chief. The reason this doesn't happen is because time series Analysis(especially finance) are difficult to handle, including unit root processes.
# As explained above, it is dangerous to analyze the raw data as it is, so we first test whether the process is a unit root process or not You need to do this. In this article, I will introduce the commonly used python adfuller.

# In[ ]:


wti=wti["Price"]
res_ctt = sm.tsa.stattools.adfuller(wti,regression="ctt") #There is a trend term (up to 2nd order)
res_ct = sm.tsa.stattools.adfuller(wti,regression="ct") #There is a trend term (up to the first order)
res_c = sm.tsa.stattools.adfuller(wti,regression="c")  #No trend term, constant term
print('"p-value"{0},\n"p-value"{1},\n"p-value"{2}'.format(res_ctt[1],res_ct[1],res_c[1]))


# adfuller sets up the null hypothesis that it is a unit root, and any of the three patterns resulted in high P-values "that do not reject the null hypothesis that it is a unit root".
# This means that the oil price is a unit root process.
# I will not mention it, but a random walk is also a unit root process.
# Thus, it is possible to ascertain whether a process is a unit root process or not through a test.
# 
# One way to counteract the spurious regression is to use a state-space model and other methods, but
# In this article, I'll show you how to "take a difference", which is a simple countermeasure to a simple unit root process.

# In[ ]:


wti_diff = wti.diff().dropna()
wti_diff.plot()


# If you take the difference, the values are scattered around the mean 0, and it looks more like a stationary process than the original data. How about a test?

# In[ ]:


res_ctt = sm.tsa.stattools.adfuller(wti_diff,regression="ctt") 
res_ct = sm.tsa.stattools.adfuller(wti_diff,regression="ct") 
res_c = sm.tsa.stattools.adfuller(wti_diff,regression="c") 
print('"p-value"{0},\n"p-value"{1},\n"p-value"{2}'.format(res_ctt[1],res_ct[1],res_c[1]))


# We were able to reject the null hypothesis, which is a unit root process; if you can't reject it after one difference, reject it The other way is to take a difference and then analyze it until you can, but by taking the difference, you can make the time series The information that This is called the excess difference.Also, there are cases where you shouldn't take a difference in the first place, so I'll explain this point if I have time I think.Note that a random walk can be converted to a stationary process by taking a difference, but it is still a random walk and is not a valid variable.
# 
# In this article, I have conveyed the following two points  
# (1) Time series data may cause  spurious regression, so it is safer to perform a unit root test first.  
# (2) One of the responses to the unit root process is that there is a way to "take the difference".
# 
# There are many more techniques to analyze. I will share them and I hope you will share them too. I hope that the knowledge shared here will help raise the bar for your organization.
# 
