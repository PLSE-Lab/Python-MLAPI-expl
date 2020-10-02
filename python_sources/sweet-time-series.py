#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will apply all the time series techniques I have learnt. My goal is:
# * Understand the trend of candy production
# * Review some basic time series models
# * Review some basic processes that explain the latency of some of the models
# 
# Table of content:
# - <a href='#0'>0. Time series overview. Stationary. Fitting a time series overview.</a>
# - <a href='#1'>1. Preliminary analysis - Time series decomposition</a>
#     - <a href='#1.1'>1.1 Detrend</a>
#     - <a href='#1.2'>1.2 Seasonal pattern</a>
#     - <a href='#1.3'>1.3 Autocorrelation (ACF) - Measuring correlation in time</a>
#     
# - <a href='#2'>2. Models</a>
#     - <a href='#2.1'>2.1 White noise model </a>
#     - <a href='#2.2'>2.2 Autoregressive (AR) models. Partial Autocorrelation (PACF)</a>
#     - <a href='#2.3'>2.3 Random walk model (with drift) </a>
#     - <a href='#2.4'>2.4 Moving average (MA) models</a>
#     - <a href='#2.5'>2.5 ARMA models</a>
#     - <a href='#2.6'>2.6 ARIMA models</a>

# # <a id='0'>0. Time series overview. Stationary. Fitting a time series overview.</a>
# ## 1. Time Series
# $\{X_t\}_t: X_0, X_1, ..., X_t$  is collection of random variables indexed in time
# Define:
# * Marginal mean per time step: $\mu_X(t) = \mathbb{E}[X_t]$
# * Marginal variance per time step: $var_X(t) = \mathbb{E}[(X_t - \mu_X(t))^2] = \gamma(t,t)$
# * Autocovariance: $\gamma_X(s,t) = cov(X_s, X_t) = \mathbb{E}[(X_s - \mu_X(s))(X_t - \mu_X(t))]$
# 
# ## 2. Stationarity
# ### **2.1 Weak stationarity**
# * Mean and variance are the same for all $X_t$'s: $\mu_X(t) = \mu_X$ & $var_X(t) = var_X$
# * Covariance is only a function of gap: $cov(X_s, X_t) = \gamma_X(s-t)$
# 
# ### **2.2 Strong stationarity**: Distribution of $X_t, ..., X_{t+n}$ is the same as distribution of $X_{t+h},..., X_{t+h+n} \forall t, n, h$
# **Stationarity is important because it allows prediction. Many prediction models on time series rely on stationarity of the series.**
# ## 3. Fitting a time series overview
# ### 3.1 Transformation to make it stationary
# * non-linearly transform (i.e log-transform)
# * remove trends/ seasonality
# * differentiate successively (differencing)
# 
# ### 3.2 Check for white noise (ACF)
# ### 3.3 If stationary, plot autocorrelation (ACF) to find order. If finite lag, fit MA model. 
# ###               Otherwise, fit AR model.
#                 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Data management
import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))

# Visualization
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') #style for time series visualization
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

#Statistics 
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




# Any results you write to the current directory are saved as output.


# # <a id='1'>1. Preliminary analysis</a>

# In[ ]:


candy = pd.read_csv("../input/candy_production.csv")


# In[ ]:


candy.tail()


# In[ ]:


candy.info()


# Visualization of candy production (as % of 2012 production) over time.

# In[ ]:


# Create time series from pandas
rng = pd.date_range(start = '1/1/1972', end = '8/1/2017', freq = 'MS')


# In[ ]:


cdy = pd.Series(list(candy['IPG3113N']), index = rng)


# In[ ]:


rcParams['figure.figsize'] = 20, 5
cdy.plot(linewidth = 1, ls = 'solid')
plt.title('Monthly US Candy Production (1972 - 2017)')
plt.show()


# This suggests that:
# * There might be a trend in candy production (increasing trend)
# * There might be a seasonal pattern 
# 
# Therefore, it is important to search for stationarity over this dataset.

# ## <a id='1.1'>1.1 Detrend</a>
# In general, there are three factors that account for the time series:
# $X_t = T_t + S_t + Y_t$ 
# 
# In which:
# * $T_t$ is (linear) trend, can be found by linear regression
# * $S_t$ is seasonal component
# * $Y_t$ is remainder (hopefully stationary, mean zero)
# 
# There are many methods to detrend. Here, I cover three[](http://) most popular methods:
# * Fit a linear regression model to identify the trend, then subtract it (cons: can only remove linear trend)
# * Non-linear transformation
# * Differentiation

# **Linear regression**
# 
# In order to do linear regression, need to convert date range into fraction of year since the start date (1/1/1972). 
# Regression the candy production on the above time fraction.

# In[ ]:


seconds_of_year = 365*24*3600
frac = [0]*len(rng)
for i in range(1,len(rng)):
    frac[i] = round((rng[i] - rng[0]).total_seconds()/seconds_of_year, 3)
frac = np.array(frac).reshape(-1,1)


# In[ ]:


reg = LinearRegression().fit(frac, cdy)


# In[ ]:


reg.score(frac, cdy)


# In[ ]:


lr_detrended = cdy - reg.predict(frac)


# In[ ]:


lr_detrended.plot(lw = 1)
plt.title('Linear Regression Detrended')
plt.show()


# To detect non-linear trend, we can fit other non-linear model (such as quadratic model) and subtract to get the remainder.

# In[ ]:


frac_sq = frac**2
X = np.concatenate([frac_sq, frac], axis = 1)


# In[ ]:


X.shape


# In[ ]:


reg_2 = LinearRegression().fit(X, cdy)


# In[ ]:


reg_2.score(X, cdy)


# In[ ]:


lr_detrended_2 = cdy - reg_2.predict(X)


# In[ ]:


lr_detrended_2.plot(lw = 1)
plt.title('Linear Regression Detrended (Quadratic Equation)')
plt.show()


# Non linear transformation $Y_t  = \sqrt{X_t - \mu_t}$, where $\mu_t$ is the mean of $X_t$. 
# 
# To get the mean, we need to do sampling. There are two types of sampling:
# * **Upsampling**: Time series is resampled from low frequency to high frequency (monthly to daily frequency). It involves filling or interpolating missing data. 
# * **Downsampling**: Time series is resampled from high frequency to low frequency (weekly to monthly frequency). It involves aggregation of existing data. This is what we can do in this case to estimate $\mu_t$.

# In[ ]:


mean_year = cdy.resample('Y').mean()


# In[ ]:


mean_year.head()


# In[ ]:


dup_mean_year = pd.Series(np.repeat(mean_year.values,12,axis=0)[:548], index = rng)


# In[ ]:


dup_mean_year.head()


# In[ ]:


nonlinear_transformed = cdy - dup_mean_year


# In[ ]:


nonlinear_transformed.plot(lw = 1)
plt.title("Non-linear transformation")
plt.show()


# That was pretty good. Next, we will try the last method - differentiation. 
# 
# Differentiation is taking the differences between the observations by some **lag** value (difference between consecutive observations is lag-1 difference). The lag difference can be adjusted to suit the specific temporal structure. For time series with a seasonal component, the lag may be expected to be the period (width) of the seasonality.
# 
# Temporal structure may still exist after performing a differencing operation, such as in the case of a nonlinear trend. As such, the process of differencing can be repeated more than once until all temporal dependence has been removed. The number of times that differencing is performed is called the **difference order**.
# 
# These following equations summarize the above:
# 
# $Y_t = \nabla X_t = X_t - X_{t-1}$                                                                                     (First order)
# 
# $Y_t = \nabla^2 X_t = \nabla X_t - \nabla X_{t-1} = X_t -  2 X_{t-1} + X_{t-2}$                 (Second order)
# 

# In[ ]:


rcParams['figure.figsize'] = 20, 5
diff_1 = cdy.diff()
diff_2 = diff_1.diff()
ax1 = plt.subplot(211)
diff_1.plot(lw = 1, axes = ax1)
ax2 = plt.subplot(212, sharex=ax1)
diff_2.plot(lw = 1, axes = ax2)
ax1.set_title("Differentiation (1st order)")
ax2.set_title("Differentiation (2nd order)")
plt.show()


# ## <a id='1.2'>1.2 Seasonal pattern</a>
# * Fit periodic regression model & subtract
# * Subtract monthly averages
# * Differentiation
# * Fourier analysis
# * Smoothing by moving average $Y_t = \frac{1}{2k+1} \sum_{h = -k}^k a_h X_{t +h}$

# Subtract monthly averages. It looks like there will be monthly pattern in candy production (in some months there might be more production than the others). To illustrate this method, I will use the linear regression detrended series and remove the seasonal patterns from there.

# In[ ]:


candy['lr_residual'] = lr_detrended.values


# In[ ]:


candy['month'] = candy['observation_date'].apply(lambda x: x[5:7])


# In[ ]:


candy.head()


# In[ ]:


month_average = candy.groupby('month')['lr_residual'].mean().reset_index().rename(columns = {'lr_residual':'month_average'})


# In[ ]:


m = list(month_average['month_average'].values)
dup_mean_month = pd.Series((m*(2017-1971))[:548], index = rng)


# In[ ]:


month_average


# In[ ]:


subtract_month = lr_detrended - dup_mean_month


# In[ ]:


rcParams['figure.figsize'] = 20, 5
dup_mean_month.plot(lw = 1)
plt.title("Seasonal Pattern")


# In summary, the result of the above decomposition is as following:

# In[ ]:


rcParams['figure.figsize'] = 20,9
ax1 = plt.subplot(411)
cdy.plot(lw = 1, axes = ax1)
ax1.set_title("Observed")

ax2 = plt.subplot(412, sharex=ax1)
trend = pd.Series(reg.predict(frac), index = rng)
trend.plot(lw = 1, axes=ax2)
ax2.set_title("Trend")

ax3 = plt.subplot(413, sharex=ax1)
dup_mean_month.plot(lw = 1, axes=ax3)
ax3.set_title("Seasonal Pattern")


ax4 = plt.subplot(414, sharex=ax1)
subtract_month.plot(lw = 1, axes=ax4)
ax4.set_title('Seasonal pattern extracted by monthly average (Residual)')
plt.show()


# Next, we can try removing the seasonal pattern by periodic differentiation (in this case, monthly differentiation) on the 1st-order differentiated time series.

# In[ ]:


rcParams['figure.figsize'] = 20, 5
# diff_monthly = cdy.diff(periods = 12)
diff_monthly=diff_1.diff(periods=12)
ax1 = plt.subplot(211)
diff_1.plot(lw = 1, axes = ax1)
# ax2 = plt.subplot(312, sharex=ax1)
# diff_2.plot(lw = 1, axes = ax2)
ax3 = plt.subplot(212, sharex=ax1)
diff_monthly.plot(lw = 1, axes = ax3)
ax1.set_title("Differentiation (1st order)")
ax2.set_title("Differentiation (2nd order)")
ax3.set_title("Seasonal differentiation (1st order)")
plt.show()


# Using the first order differencing time series, we can applying smoothing technique to reduce the effects of seasonal patterns.

# In[ ]:


moving_average = diff_1.rolling(3, center = True).mean()


# In[ ]:


rcParams['figure.figsize'] = 20, 5
diff_1.plot(lw = 1, alpha = 0.5, label = '1st order diff')
moving_average.plot(lw = 1, color = 'red', label = 'moving average')
plt.title("Removing seasonal patterns by moving average (window size = 3)")
plt.legend()
plt.show()


# Compare this to the built-in method (which also uses moving average to remove seasonal patterns)

# In[ ]:


# Now, for decomposition...
rcParams['figure.figsize'] = 20, 9
decomposed = sm.tsa.seasonal_decompose(cdy,freq=30) # The frequncy is monthly
figure = decomposed.plot()
plt.show()


# ## <a id='1.3'>1.3 Autocorrelation (ACF) - Measuring correlation in time</a>
# <h3><center> $\rho_X(s,t) = \frac{\gamma_X(s,t)}{\sqrt{\gamma_X(s,s)\gamma_X(t,t)}}$ </center></h3>
# 
# Properties of ACF:
# * Symmetric $\gamma_X(s,t) = \gamma_X(t,s)$
# * Measures linear dependence of $X_t$, $X_s$
# * Relates to smoothness
# * For **weak stationarity** $\gamma_X(t, t+h) = \gamma_X(h),  \forall t$
# 
# ### **Sample estimates for stationary series**
# * $\hat{\mu}_X = \bar{X} = \frac{1}{n} \sum_{t=1}^n X_t \qquad var(\bar{X}) = \frac{1}{n} \sum_{h=-n}^n (1-\frac{|h|}{n} \gamma(h) )$ 
# * sample autocovariance:
# <h3><center> $\hat{\gamma}_X(h) = \frac{1}{n} \sum_{t=1}^{n - |h|} (X_t - \mu_X) (X_{t + |h|} - \mu_X)$ </center></h3>
# * sample autocorrelation:
# <h3><center> $\hat{\rho}_X(h) = \frac{\hat{\gamma}_X(h)}{\hat{\gamma}_X(0)}$ </center></h3>

# In[ ]:


rcParams['figure.figsize'] = 20, 15
plt.subplots_adjust(hspace=0.5)

ax1 = plt.subplot(611)
plot_acf(cdy, ax = ax1, marker = '.', lags=200)
ax1.set_title("ACF of original series")

ax2 = plt.subplot(612)
plot_acf(lr_detrended, ax = ax2, marker = '.', lags=200)
ax2.set_title("ACF of linear regression detrended")


ax3 = plt.subplot(613)
plot_acf(diff_1.dropna(), ax = ax3, marker = '.', lags = 200)
ax3.set_title("ACF of 1st order differentiation")

ax4 = plt.subplot(614)
plot_acf(moving_average.dropna(), ax=ax4, marker='.', lags=200)
ax4.set_title("ACF of moving average")

ax5 = plt.subplot(615)
plot_acf(decomposed.resid.dropna(), ax=ax5, marker='.', lags=200)
ax5.set_title("ACF of built-in decomposed residual")

ax6 = plt.subplot(616)
plot_acf(diff_monthly.dropna(), ax=ax6, marker='.', lags=200)
ax6.set_title("ACF of monthly differentiation")

plt.show()


# The above ACF plots show that there are still some level of seasonal patterns after all the possible transformations. Overall, the monthly differentiation (after 1st order differentiation) seems to be very similar to white noise model.

# # <a id='2'>2. Models</a>
# ## <a id='2.1'>2.1 White noise model</a>
# $\{W_t\}_t$ uncorrelated, mean zero, same variance $\sigma_W^2$ (often i.i.d Gaussian)
# <h3><center> $\mu_t = \mathbb{E}[W_t] = 0 \quad \forall t$, $cov(W_s, W_t) = 0 \quad s\neq t$ </center></h3>
# * autocovariance:
# 
# <h3><center> 
#     
# $$ \gamma_W(s,t) =
#   \begin{cases}
#     \sigma_W^2       & \quad \text{if } s=t\\
#     0 & \quad \text{otherwise}
#   \end{cases}
# $$
# </center></h3>
# depends only  on $|s-t|$
# 
# * Stationary, capture no dependencies overtime
# * **Checking for white noise**: $\hat{\rho}(h)$ is approximately $\mathcal{N}(0, \frac{1}{n})$ under mild conditions
# 
# We will simulate a white noise model (Gaussian with mean 0 and std 1) and experiment some of the series' properties.
# 

# In[ ]:


white_noise = pd.Series(normal(size=500))


# In[ ]:


rcParams['figure.figsize'] = 20, 8
plt.subplots_adjust(hspace=0.5)

ax1 = plt.subplot(211)
white_noise.plot(lw=1)
ax1.set_title("White noise series")

ax2 = plt.subplot(212)
plot_acf(white_noise, marker = '.', lags = 200,ax=ax2)
ax2.set_title("ACF of white noise")

plt.show()


# Autocorrelation of white noise vanishs quickly after $h=0$. 

# ## <a id='2.2'>2.2 Autoregressive (AR) models. Partial Autocorrelation (PACF)</a>
# ### $AR(p)$ model
# 
# <center> $X_t = \sum_{h=1}^p \phi_h X_{t-h} + W_t \quad \text{for } W_t \text{ is white noise}$ </center>
# * Autocorrelation decays exponentially over time (but never zero)
# * Only stationary under certain conditions (later)
# 
# ### Partial Autocorrelation
# Autocorrelation of $AR(p)$, $\gamma(h) \neq 0$ even for $h>p$. However, once $X_{t-1}, ..., X_{t-p}$ is known, knowing $X_{t-p-1}$ gives no further information. In other words, *conditional autocorrelation* should be 0. <br>
# Define $\phi_{hh}$ the last coefficient of a *fitted* $AR(h)$ model (*to the dataset*): $\hat{X}_t = \hat{\phi}_1 X_{t-1} + ... + \hat{\phi}_h X_{t-h}$ where $\phi_{hh} = \hat{\phi}_h$. (*Notice that it is a fitted model, so there is no white noise term. $\hat{X}_t$ is an estimate of $X_t$)*<br>
# Formally, 
# <center>$\phi_{11} = corr(X_t, X_{t-1})$ </center>
# <center> $\phi_{hh} = corr(X_t - \hat{X}_t, X_{t-h} - \hat{X}_{t-h})$ </center>
# (both $\hat{X}_t, \hat{X}_{t-h}$ are uncorrelated with $X_{t-1}, ..., X_{t-h}$)<br>
# 
# Then, for an $AR(p)$ model, $\phi_{hh} = 0 \quad \forall h > p$. 
# 
# 
# ### Fitting to a $AR(p)$ model
# 1. Compute PACF (of the residual) to get order. Alternatively, find model with k parameters to minimize [Akaike Information Criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion). In short, AIC is a information-theory-based criterion used in model selection. <br>
# *"When a statistical model is used to represent the process that generated the data, the representation will almost never be exact; so some information will be lost by using the model to represent the process. AIC estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model."* <br>
# The usage of AIC in model selection comes in handy in *statsmodels* package. There are many more similar criteria covered by this package (i.e Bayes Information Criterion, t-stat).
# 2. Estimate *k* coefficients and noise variance $\sigma^2_w$ via [Yule-Walker equations](http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/YWSourceFiles/YW-Eshel.pdf) (also in *statsmodels* package).
# 3. Compute residuals, test for white noise. Alternative, compute mean square error (MSE) by cross-validation.

# Till now, for consistency, we will use the stationary decomposed residual for model fitting. Below, we will try to fit an AR model. Before using a package with information theory to find the best order, we will explore the dataset by plotting the partial autocorrelation.

# In[ ]:


# stationary_cdy = decomposed.resid.dropna()
stationary_cdy = diff_monthly.dropna()


# In[ ]:


# stationary_cdy has data from 1973-04-01 to 2016-05-01 (518 data points)
# split 80-20 for train and test
ind_80 = int(len(stationary_cdy)*0.8)
train, test = stationary_cdy[:ind_80], stationary_cdy[ind_80:]
print(len(train), len(test))


# In[ ]:


plot_pacf(stationary_cdy, title = 'PACF of the stationary candy series', alpha = 0.05, lags = 100)
plt.ylabel('PACF')
plt.xlabel('Lags')
plt.show()


# From the PACF plot, it looks like a model of order 18 would be sufficient. Next, we will use the information criterion to determine the model order.

# In[ ]:


# Order is determined by BIC
ar_aic = AR(stationary_cdy).fit(ic = 'bic')
print('Lag by BIC: %s' % ar_aic.k_ar)


# It looks like the model order at $p=13$ makes the most sense but BIC criterion favors $p=18$. We will then evaluate these models using cross-validation.<br>
# There are three most common ways to conduct cross validation:
# * In limited AR conditions (i.e limited time correlation), using normal cross validation (train on the past and predict on the future)
# * [Rolling forecast](https://www.google.com/search?q=rolling+forecast&oq=rolling+forecast&aqs=chrome..69i57j0l5.2856j0j7&sourceid=chrome&ie=UTF-8). In the "modified" version of rolling forecast, to predict, say '2017-02-01', we can train a $AR(15)$ model using all the data up to '2017-01-01'. Next, to predict '2017-03-01', we train a new model using all the data up to '2017-02-01'.
# 
# <img src="https://i.stack.imgur.com/fXZ6k.png" alt="drawing" width="400"/>
# <center>[Reference](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection)</center>
# 
# * Others (out-of-sample, [h-hv block](https://pdfs.semanticscholar.org/fec8/709c61374672305f5f07d1af884254a01c24.pdf))
# <img src="http://www.jneurosci.org/content/jneuro/36/47/11987/F2.large.jpg?width=800&height=600&carousel=1" alt="drawing" width="200"/>
# <center>[Illustration of hv-block cross-validation (reference: The Journal of Neuroscience)](http://www.jneurosci.org/content/jneuro/36/47/11987/F2.large.jpg?width=800&height=600&carousel=1)</center>
# 
# 
# 
# **In this example, we will evaluate using rolling forecast.**
# 

# In[ ]:


def rolling_forecast_evaluation(order):
    preds = []
    for i in range(ind_80,len(stationary_cdy)):
        cdy_train = stationary_cdy[:i]
        ar_model = AR(cdy_train).fit(maxlag = order)
        one_ahead_predict = ar_model.predict(start = len(cdy_train), end = len(cdy_train), dynamic = False)
        preds.append(one_ahead_predict[0])

    print("MSE of order = %d: %.3f" % (order,mean_squared_error(test, preds)))


# In[ ]:


print(">>> Evaluated by rolling forecast")
rolling_forecast_evaluation(18)


# We can also compare the behavior of these two models based on normal cross validation. Here, we only fit the model up to the train dataset. If we fit the model with all the data, prediction with dynamic=False sample the data from the series (in-sample) to be used as lags. Otherwise, if dynamic=True, use the previously predicted datapoint as lag for the next datapoint.

# In[ ]:


# model_13 = AR(stationary_cdy).fit(maxlag = 13)
model_18 = AR(train).fit(maxlag=18)
# preds_13 = model_13.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
preds_18 = model_18.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
print(">>> Evaluated by normal cross validation")
# print("MSE of order = 13: %.3f" % (mean_squared_error(test, preds_13)))
print("MSE of order = %d: %.3f" % (model_18.k_ar,mean_squared_error(test, preds_18)))


# Choosing the model of order 18, we can test if the residual is white noise.

# In[ ]:


rcParams['figure.figsize'] = 20, 8
all_preds=AR(stationary_cdy).fit(maxlag=18).predict(start = 18, end = len(stationary_cdy)-1, dynamic = False)
# First, plot both series on the same plot
ax1 = plt.subplot(311)
stationary_cdy.plot(lw=1, label = 'stationary')
all_preds.plot(lw=1, label = 'AR(18)')
ax1.legend()

residual = (stationary_cdy - all_preds)
ax2 = plt.subplot(312)
residual.plot(lw=1, label = 'residual')
ax2.legend()

ax3 = plt.subplot(313)
plot_pacf(residual, title = None, alpha = 0.05, lags = 100, ax=ax3, label = 'Residual')
ax3.set_ylabel('PACF')
ax3.legend()

plt.show()


# What if we fit the AR model to our original, non-stationary series?

# In[ ]:


rcParams['figure.figsize'] = 20, 8
cdy_model=AR(cdy).fit()
all_preds=cdy_model.predict(start = cdy_model.k_ar, end = len(cdy)-1, dynamic = False)
# First, plot both series on the same plot
ax1 = plt.subplot(311)
cdy.plot(lw=1, label = 'original')
all_preds.plot(lw=1, label = 'AR')
ax1.legend()

residual = (cdy - all_preds)
ax2 = plt.subplot(312)
residual.plot(lw=1, label = 'residual')
ax2.legend()

ax3 = plt.subplot(313)
plot_pacf(residual, title = None, alpha = 0.05, lags = 100, ax=ax3, label = 'Residual')
ax3.set_ylabel('PACF')
ax3.legend()

plt.show()


# The residual looks like white noise! AR model works very well with both the original candy data and the stationary candy data. 

# ## <a id='2.3'>2.3 Random walk model (with drift)</a>
# Random walk is a special case of AR(p) model ($p=1$):
# <center>$X_t = X_{t-1} + W_t = X_0 + \sum_{h=1}^t W_h \quad \text{(fixed } X_0 \text{)}$</center> 
# Random walk with drift adds a constant term:
# <center>$X_t = X_{t-1} + W_t + \delta = X_0 + \sum_{h=1}^t W_h + t\delta$ </center>
# Some properties:
# * Mean: $\mathbb{E}[X_t] = t\delta + X_0$
# * Variance: $var(X_t) = t\sigma_W^2$ (without drift)
# * Autocovariance: $\gamma_X(s,t) = min\{s,t\} \sigma_W^2$
# * Not stationary. However, first order differentiation $\nabla X$ is stationary (because it is just white noise)
# 
# After fitting AR model, we are confident that our dataset cannot be generated from a random walk model. Below is just a simulation of random walk model using pseudo-random function to generate numbers in [0.0, 1.0) interval.

# In[ ]:


np.random.seed(42)
random_walk = list()
random_walk.append(-1 if np.random.random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if np.random.random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)
    
rcParams['figure.figsize'] = 20, 4
plt.plot(random_walk, lw = 1)
plt.title("Simulation of random walk model")
plt.show()


# We will confirm if differencing the random walk model returns white noise model.

# In[ ]:


rcParams['figure.figsize'] = 20, 6

plt.subplots_adjust(hspace = 0.5)
rw_diff = pd.Series(random_walk).diff(periods=1)
ax1 = plt.subplot(211)
rw_diff.plot(lw=1, ax=ax1)
ax1.set_ylim(1.5,-1.5)
ax1.set_title('1st order differencing of random walk model')

ax2 = plt.subplot(212)
plot_acf(rw_diff.dropna(), ax=ax2, lags=200, marker = '.', title = 'ACF of 1st order differencing of random walk model')


plt.show()


# ## <a id='2.4'>2.4 Moving average (MA) model </a>
# ### $MA(p)$ model
# 
# <center> $X_t = W_t + \theta_1W_{t-1} + ... + \theta_pW_{t-p} \quad \text{for } W_t \text{ is white noise}$ </center>
# * Mean, $\mathbb{E}[X_t] = 0$
# * Autocovarianc $\gamma$ only depends on $|s-t|$ 
# * Always stationary! (Proof comes later)
# * ACF reflects order: $\gamma(s,t) = 0 \iff |s-t| > p$
# * ACF distinguishes MA and AR models
# 
# In practice, this model is not used very often because it requires the series to be strictly stationary and can be decomposed into white noise models. We will consider a more practical model, Autoregressive Moving Average (ARMA).

# ## <a id='2.5'>2.5 Autoregressive Moving average (ARMA) model </a>
# ### $ARMA(p,q)$ model can be thought as AR(p) + MA(q)
# 
# <center> $X_t = \phi_1X_{t-1} + ... + \phi_pX_{t-p} + W_t + \theta_1W_{t-1} + ... + \theta_qW_{t-q} \quad \text{for } W_t \text{ is white noise}$ </center>
# 
# * Both ACF and PACF decay (never converge to zeros)
# 
# We will explore fitting an ARMA to the original candy series.
# 
# <i> **Note: R has nicer packages for these types of models. However, for consistency, let's bear with Python! <i> 

# In[ ]:


plot_acf(stationary_cdy, lags = 50, title = 'Candy production ACF')
plt.show()
plot_pacf(stationary_cdy, lags = 50, title = 'Candy production Partial ACF')
plt.show()


# In[ ]:


p,q = 12,2
#ARMA model doesn't have a nice information criterion built-in hyperparameters tuning
arma_model = ARMA(stationary_cdy, order = (p,q)).fit(disp=0) 
fig, ax = plt.subplots()
ax = stationary_cdy.plot(ax=ax, label='Candy')
arma_model.plot_predict('2015','2019',ax=ax, plot_insample=False)
plt.show()


# ## <a id='2.6'>2.6 ARIMA model </a>
# ARIMA model is similar to ARMA model but with a differencing degree. This differencing degree is used to make the series stationary. The order of ARIMA model should be determined by <a href="https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method#Autocorrelation_and_partial_autocorrelation_plots">Box-Jenkins method.</a>

# In[ ]:


rcParams['figure.figsize'] = 20, 6
plot_acf(stationary_cdy, lags = 50, title = 'Candy production ACF')
plt.show()
plot_pacf(stationary_cdy, lags = 50, title = 'Candy production Partial ACF')
plt.show()


# In[ ]:


model = ARIMA(stationary_cdy, order=[12, 1, 2])
model_fit = model.fit(disp=0)


# In[ ]:


prediction = model_fit.predict()
fig, ax = plt.subplots()
stationary_cdy.plot(ax=ax, label='Candy production')
# prediction.plot(ax=ax, label='ARIMA Predicted')
model_fit.plot_predict('2015','2019',ax=ax, plot_insample=False)
ax.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




