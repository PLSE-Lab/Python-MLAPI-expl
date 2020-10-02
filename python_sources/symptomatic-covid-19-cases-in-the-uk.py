#!/usr/bin/env python
# coding: utf-8

# # Is the prevalence of COVID-19 in the UK rising (as of 21st May)?
#  
# Using data from the [COVID Symptom Study](https://covid.joinzoe.com/data) we calculate whether the recent uptick in symptomatic cases since the easing of the lockdown on 10<sup>th</sup> May is statistically significant.
# 
# The data contains daily estimates of the total number of symptomatic COVID-19 cases in the UK, based on self-reported symptoms.
# 
# Here's the raw data:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats
import math

data = pd.read_csv("/kaggle/input/symptomatics/data.csv",parse_dates=["Date"], dayfirst=True, index_col=0)
#plt.plot(data["Date"],data["Total"])
plt.plot(data)
plt.gcf().autofmt_xdate()


# The initial peak occured shortly after the lockdown began. We assume this is due to the slowness of people's habit change along with the fact that at lockdown, positive cases were spending more time in households consisting of susceptible others. By around 8<sup>th</sup> April, the trend settles down to an exponential decay, as we would expect if R < 1.0.
# 
# We assume the trend follows a function n = Ae<sup>kt</sup> + C and fit that function to data from 8<sup>th</sup> April until the day before the government announced the easing of lockdown on 10<sup>th</sup> May. The offset, C, can be explained, at least partially, by the background prevalence of symptoms not caused by COVID, e.g. hayfever. 

# In[ ]:


def f(x,A,k,C):
    return A*np.exp(k*x)+C

fitdata = data["2020-04-08":"2020-05-09"]
expdata = data["2020-04-08":"2020-05-21"]
(A,k,C),_ = opt.curve_fit(f,list(range(0,32)), fitdata["Total"],bounds=([900000,-1.0,100000],[1500000,-0.01,300000]))
print(A,k,C)
fit = f(np.linspace(0,43,44),A,k,C)
dffit = pd.Series(fit, index=expdata.index, name="Total")
plt.plot(dffit)
plt.plot(expdata)
plt.gcf().autofmt_xdate()


# 
# Note how well the data fits the model up until 10<sup>th</sup> May, this gives us some confidence in the model.
# 
# We now perform a t-test on the error between the data and the fit before and after the announcement to see if there is a significant difference. Under the null hypothesis, and the assumption that the error is Gaussian, the model parameters would be unchanged by the announcement so the error before and after the announcement would be drawn from the same Gaussian. The t-test gives a measure of the likelihood of this.

# In[ ]:


noise = expdata["Total"] - dffit
preEasingNoise = noise["2020-04-08":"2020-05-09"]
postEasingNoise = noise["2020-05-10":"2020-05-21"]
plt.plot(preEasingNoise)
plt.plot(postEasingNoise)
plt.gcf().autofmt_xdate()
print(stats.ttest_ind(preEasingNoise, postEasingNoise, equal_var = True))


# The graph above shows the error before and after the announcement. The t-test gives a p-value of 0.00008, which is highly significant so we can discount the null hypothesis and conclude that something has caused the number of self-reported symptomatic cases to increase since the government's announcement on 10<sup>th</sup> May. The obvious explanation is that R has crept above 1.0 and the disease is now spreading again in the UK. This would be disastrous, especially in view of the government's plan to loosen restrictions further on 1<sup>st</sup> June.
# 
