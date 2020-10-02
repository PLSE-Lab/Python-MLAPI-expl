#!/usr/bin/env python
# coding: utf-8

# ---
# # Deconstructing ARIMA and Forecasting Madrid Air Quality
# **[Nicholas Holloway](https://github.com/nholloway)**
# 
# ---
# ### Mission 
# By the end of this tutorial the hope is that we understand all the parts of ARIMA and the foundations of the ARMA family of time series models. We will build a way of thinking about how to decompose a time series, a method to model time series, and introduce a variety of tests we can use on our time series and in many other statistical domains to test how well we modelled our data. We will cover tests for autocorrelation like **Ljung-Box**, and **Breusch-Godfrey**, as well as test for normality like **Shapiro-Wilks** and **Jarque-Bera**. We will fit an ARIMA model and hopefully have what it takes to model our own time series data. 
# 
# ### Table of Contents:
# 1. [ARIMA Breakdown](#arima)
#     1. [Autoregressive](#ar)
#     2. [Moving Average](#ma)
#     3. [Differencing](#i)
# 2. [Box-Jenkins Method](#box-jenkins)
# 3. [Stationarize Data](#stationarize)
# 4. [Estimate Parameters](#parameters)
# 5. [Diagnostic Checking](#diagnostic)
#     1. [Ljung-Box Test](#lju-box)
#     2. [Breusch-Godfrey Test](#bre-god)
#     3. [Shapiro-Wilks Test](#sha-wil)
#     4. [Jarque-Bera Test](#jar-ber)
# 6. [Conclusion](#conclusion)
# 
# ### Time Series Kernels: 2 of 4 
# * [Stationarity, Smoothing, and Seasonality](https://www.kaggle.com/nholloway/stationarity-smoothing-and-seasonality)
# * [Deconstructing ARIMA](https://www.kaggle.com/nholloway/deconstructing-arima)
# * [Seasonality and SARIMAX](https://www.kaggle.com/nholloway/seasonality-and-sarimax)
# * [Volatility Clustering and GARCH](https://www.kaggle.com/nholloway/volatility-clustering-and-garch)

# In[ ]:


import math
import itertools
import os
import pandas as pd
import numpy as np
import operator
from collections import defaultdict
from scipy.stats import boxcox, shapiro, probplot, jarque_bera
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight') 
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '../input/csvs_per_year/csvs_per_year'
files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
df = pd.concat((pd.read_csv(file) for file in files), sort=False)
df = df.groupby(['date']).agg('mean')
df.index = pd.DatetimeIndex(data= df.index)


# In[ ]:


col_list = ['BEN', 'CO', 'EBE', 'NMHC', 'NO_2', 'O_3', 'PM10', 'SO_2', 'TCH', 'TOL']
pm10_df = pd.DataFrame(df['PM10'].resample('M').mean())
pm10_df.rename(columns= {'PM10':'pm10'}, inplace=True)
pm10_df.plot(figsize=(15, 6))
plt.title('PM10 in Madrid Air from 2001-2019', fontsize=20)
plt.legend(loc='upper left')
plt.show()


# <a id='arima'></a>
# # ARIMA
# ---
# <a id='ar'></a>
# ## Autoregressive Process (AR)
# When our time series is regressed against one or more lagged values of itself it can be modelled with an autoregressive model. Here is the general form for an autoregressive process of order p: 
# 
# $$X_t= \phi_1X_{t-1} + \phi_2X_{t-2}+...\phi_pX_{t-p}+Z_t$$
# 
# The parameter p determines how many $\phi_1X_{t-1}$ terms we have in our model, where each term is a prior time step, or lag that is correlated with our current value. This is what is meant by 'is regressed against one of more lagged values of itself'. We can use correlograms, or autocorrelation plots to determine how many lagged values are correlated with our current value and how many terms we need to describe our data- which will become the value of p. If p were 2, then we would have a second order autoregressive model AR(2), where $Z_t$ is white noise. 
# 
# We're introducing autoregressive processes to begin our dive into the ARMA family of models. ARMA, autoregressive (AR) and moving average (MA) are two components to a family of models that are incredibly popular and can model and forecast lots of real word time series problems. 
# 
# Autocorrelation, which we mentioned earlier, is a condition where error terms in a time series are correlated between periods such that an underestimate in one period can result in an underestimate in a subsequent period. Autocorrelation interferes with linear regression models impedes stationarity. If our data shows some serial correlation, we might be able to explain it with an autoregressive model. To decide the value for the parameter p, we can use correlograms like the ACF, autocorrelation function, and PACF, partial autocorrelation function. Lag values on the x-axis where correlation coefficients, y-axis, surpass the confidence intervals indicate there is a significant correlation between the t=0 value and the value at that lag. The difference between the PACF and the ACF is that the PACF removes the effect of prior lags, so often there may be a noticeable trend in the ACF with one value highest but the same high value in the PACF will have a  drop-off after because the high correlation was removed for later lags in the PACF. It can take some practice but getting comfortable interpretting ACf and PACF plots will help us determine the correct order for our AR model quickly. 
# 
# <a id='ma'></a>
# ## Moving Average (MA)
# Where AR models depended on the lagged values of the data, moving average models depend on the residuals (errors) of the previous forecasts, which allows it to learn the shocks and noise of our data.
# 
# Mathematically, the MA(q) is a linear regression model, where $w_t$ is white noise:
# 
# $$x_t=w_t+B_1w_{t-1}+...+B_qw_{t-q}$$
# 
# To find the value for MA we can look at the ACF and PACF. Our ACF looks at correlations between each point and the subsequent points. So if the autocorrelation coefficient is high at value 1 it means there is an autocorrelation between the value x and the value at lag 1 or $x_{t-1}$. The PACF plot summarizes the correlations between an observation and its lagged values but controls for the values of the time series at all shorter lags. This is why in identifying the parameter for the AR we look fist at the ACF and then to the PACF to confirm the lag value. A lag value in the ACF plot with a high coefficient indicates autocorrelation at that value but the following lag values often have high but decaying coefficients as a result. The PACF, because it removes the prior lagged values allows us to make a clear determination about which lag values exhibit the autocorrelation. For finding the parameter, q, for our MA series we use the opposite heuristic. We first look to the PACF at lag values where the coefficients trail off and then to the ACF for a hard cut-off to determine our value. 
# 
# <a id='i'></a>
# ## Differencing (I)
# Differencing is subtracting an observation from an observation at the previous time step. Differencing generates a time series of the changes between raw data points and helps us create a time series that is stationary. Normally, the correct amount of differencing is the lowest order of differencing that yields a time series which fluctuates around a well-defined mean value and whose autocorrelation function (ACF) plot decays fairly rapidly to zero. After each differencing operation, like we perform below, we can conduct an augmented Dickey-Fuller (adf) and Kwiatkowski-Phillips-Schmidt-Shin (kpss) test to check for stationarity. 
# 
# <a id='box-jenkins'></a>
# ## Box-Jenkins Method
# ---
# #### How to fit our ARMA models
# This is the heuristic proposed by the creators of the ARIMA model for how to fit data with it. 
# 
# 1. Model Identification: 
#     1. Difference the series to make it stationary 
#         - Use unit root tests like ADF and KPSS
#         - Avoid overdifferencing, that can cause extra serial correlation
#     2. Find the parameters for AR and MA
#         - ACF and PACF are used in conjunction to determine the values of AR and MA
# 2. Parameter Estimation:
#     1. Use numerical methods to minimize a loss or error term 
# 3. Diagnostic Checking:
#     1. Overfitting 
#         - The model should not be more complex thant it needs to be, measurements like AIC can help
#     2. Residual Errors
#         - The errors from an ideal model would resemble a Gaussian distribution with mean 0, and symmetrical variance. Histograms and Q-Q plots may suggest an opportunity for data pre-processing. A skew or non-zero mean may suggest bias. Additionally the ideal model has no temporal structure as checked by ACF and PACF plots of the residual error time series. 

# <a id='stationarize'></a>
# # Step 1A: Stationarize Data
# ---
# Any stationary data can be approximated with a stationary ARMA model, thanks to the Wold decomposition theorem.
# 
# #### Box Cox Transformation
# The box cox transformation makes non-normal data normally distributed. We can pass it an argument for lambda to automatically perform a log transform, square root transform, or reciprocal transform. If we pass no argument for lambda it will tune automatically and return a lambda value. 
# 
# #### d Parameter
# Below we difference our data up to the third-order difference and plot it so that we can see how the time series changes. After the second difference both the KPSS and ADF return stationary so we don't run any more tests and we will use the second difference for our d value. 

# In[ ]:


def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    print('Null Hypothesis: Unit Root Present')
    print('Test Statistic < Critical Value => Reject Null')
    print('P-Value =< Alpha(.05) => Reject Null\n')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput[f'Critical Value {key}'] = value
    print (dfoutput, '\n')

def kpss_test(timeseries, regression='c'):
    # Whether stationary around constant 'c' or trend 'ct
    print ('Results of KPSS Test:')
    print('Null Hypothesis: Data is Stationary/Trend Stationary')
    print('Test Statistic > Critical Value => Reject Null')
    print('P-Value =< Alpha(.05) => Reject Null\n')
    kpsstest = kpss(timeseries, regression=regression)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output[f'Critical Value {key}'] = value
    print (kpss_output, '\n')


# In[ ]:


pm10_df['bc_pm10'], lamb = boxcox(pm10_df.pm10)
pm10_df['d1_pm10'] = pm10_df['bc_pm10'].diff()
pm10_df['d2_pm10'] = pm10_df['d1_pm10'].diff()
pm10_df['d3_pm10'] = pm10_df['d2_pm10'].diff()
fig = plt.figure(figsize=(20,40))

plt_bc = plt.subplot(411)
plt_bc.plot(pm10_df.bc_pm10)
plt_bc.title.set_text('Box-Cox Transform')
plt_d1 = plt.subplot(412)
plt_d1.plot(pm10_df.d1_pm10)
plt_d1.title.set_text('First-Order Transform')
plt_d2 = plt.subplot(413)
plt_d2.plot(pm10_df.d2_pm10)
plt_d2.title.set_text('Second-Order Transform')
plt_d3 = plt.subplot(414)
plt_d3.plot(pm10_df.d3_pm10)
plt_d3.title.set_text('Third-Order Transform')
plt.show()

pm10_df.bc_pm10.dropna(inplace=True)
pm10_df.d1_pm10.dropna(inplace=True)
pm10_df.d2_pm10.dropna(inplace=True)

print('Unit Root Tests:')
print('BoxCox-No Difference')
adf_test(pm10_df.bc_pm10)
kpss_test(pm10_df.bc_pm10)
print('\nFirst Difference:')
adf_test(pm10_df.d1_pm10)
kpss_test(pm10_df.d1_pm10)


# # Step 1B: Estimate p and q
# ---
# When modeling our time series it is often best to start as simple as possible. We will use the PACF and ACF below to find appropriate parameters but its good to initially fit the data with an AR(1), look at the residuals and see how well it fit. 
# 
# #### p Parameter (AutoRegressive)
# Where the trend in the ACF tends to dampen and there is a corresponding dropoff in the PACF indicates our significant lag. The significant lag indicates the order of our autoregression. An AR(2) may fit this data. 
# 
# #### q Parameter (Moving Average)
# Our ACF and PACF have an obvious seasonality that we also see in the original time series. We can model seasonality with extensions of ARIMA like SARIMA, SARIMAX, or with seasonality decomposition. 
# We'll use the reverse process that we used for p, to find q. The significant lag in the PACF that has a corresponding drop off in the ACF indicates the order for our moving average. I'd start with MA(1). 
# 
# Putting it all together I'd try to fit a model (2, 2, 1). When fitting our ARIMA model it is possible that combied AR and MA effects can overlap and that we can get by with one less AR or MA terms- particularly if the parameter estimates require 10+ iterations to converge. 

# In[ ]:


f_acf = plot_acf(pm10_df['d2_pm10'], lags=50)
f_pacf = plot_pacf(pm10_df['d2_pm10'], lags=50)
f_acf.set_figheight(10)
f_acf.set_figwidth(15)
f_pacf.set_figheight(10)
f_pacf.set_figwidth(15)
plt.show()


# <a id='parameters'></a>
# # Step 2: Parameter Estimation
# ---
# To find the best estimation for our ARIMA model we will try the parameters we looked for ourselves as well as employ a gridseach that will exhaustively look for the best model parameters. This is a review of different measurements we can use to evaluate our model fit.
# 
# ## Measuring Models
# #### Scale-dependent Errors
# Scale dependent errors are errors on the same scale as the data. 
#     - Mean Squared Error (MAE): Popular and easy to understand
#     - Root Mean Squared Error (RMSE)
#     
# #### Percentage Errors
# Percentage errors are unit free and are frequently used to compare forecast performace between data sets. The percentage erros are typically some form of the estimated_value/ true_value. The downside to percentage errors is that they can lead to infinite or undefined values when the true value is zero. Also, when data is without a meaningful zero, like temperature, the combination of division and then absolute value, like in MAPE, can lead to errors that don't capture the true difference. 
#     - Mean absolute percentage error (MAPE)
#     - Symmetric MAPE: Deals with MAPE's tendency to put a heavier penalty on negative errors
# 
# #### Scaled Errors
# Scaled errors are an attempt to get around some of the problems with percentage errors. 
#     - Mean absolute scaled error (MASE)
#     
# #### Akaike Information Criterion (AIC)
# AIC provides us an estimate of model quality by attempting to balance the complexity of the model with how well it fits the data. AIC uses a combination of the log likelihood and a penalty for models with many parameters to find a model that neither overfits nor is bias. AIC is used as a measure to compare a set of models  because it does not give us an absolute score but a relative one, a lower AIC score signals a better model. 
# 
# ## Gridsearch
# A gridsearch is an ordered walk across our parameter space where we can use compute power to compensate for experience. We will use a gridsearch, where fit our model to every combination of parameters and evaluate our models to see which set of parameters produce the best fit. We'll use AIC and RMSE to evaluate. 

# In[ ]:


def evaluate_model(data, pdq):
    split_date = '2012-01-01'
    train, test = data[:split_date], data[split_date:]
    test_len = len(test)
    model = ARIMA(train, order=pdq)
    model_fit = model.fit(disp=-1)
    predictions = model_fit.forecast(test_len)
    aic = model_fit.aic
    mse = mean_squared_error(test, predictions[0])
    rmse = math.sqrt(mse)
    return {'rmse': rmse, 'aic': aic}


def gridsearch(data, p_range, d_range, q_range):
    models = defaultdict()
    best_score, best_params = float('inf'), None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                params = (p,d,q)
                try:
                    score = evaluate_model(data, params)
                    models[str(params)] = score
                    if score['aic'] < best_score:
                        best_score, best_params = score['aic'], params
                except:
                    continue
    return best_params, models


# In[ ]:


p_rng = range(0, 10)
d_rng = [2]
q_rng = range(0, 10)

parameter, models = gridsearch(pm10_df['bc_pm10'], p_rng, d_rng, q_rng)


# In[ ]:


sorted_scores = sorted(models.items(), key = lambda x:x[1]['aic'])

print('Best ARIMA Parameters and Scores Ranked By AIC')
for i in range(6):
    print(f'{sorted_scores[i][0]}, {sorted_scores[i][1]}')


# <a id='diagnostic'></a>
# # Step 3: Diagnostic Checking
# ---
# ## Model Validation
# To validate our model we will look at our model's residuals. The residuals are the difference between the observed data and the predicted value from our model. Ideally they will be a stationary series, looking like white noise if plotted, and normally distributed with a mean of 0. If our residuals show a trend or autocorrelation it means there is information we have failed to capture in our model. Here are tests we can use to check that our residuals are staitonary. 
# 
# <a id='lju-box'></a>
# #### Ljung-Box Test
# Ljung-Box is a test for autocorrelation that we can use in tandem with our ACF and PACF plots. The Ljung-Box test takes our data, optionally either lag values to test, or the largest lag value to consider, and whether to compute the Box-Pierce statistic. Ljung-Box and Box-Pierce are two similar test statisitcs, $Q$, that are compared against a chi-squared distribution to determine if the series is white noise. We might use the Ljung-Box test on the residuals of our model to look for autocorrelation, ideally our residuals would be white noise. 
# 
# * $H_o$: The data are independently distributed, no autocorrelation.
# * $H_a$: The data are not independently distributed; they exhibit serial correlation.
# 
# The Ljung-Box with the Box-Pierce option will return, for each lag, the Ljung-Box test statistic, Ljung-Box p-values, Box-Pierce test statistic, and Box-Pierce p-values. 
# 
# If $p<\alpha$ (0.05) we reject the null hypothesis. 
# 
# <a id='bre-god'></a>
# #### Breusch-Godfrey Test
# Breusch-Godfrey tests for serial correlation in a model's residuals that which, if present, would mean that incorrect conclusions would be drawn from other tests, or that sub-optimal estimates of model parameters are obtained if it is not taken into account. The Breusch-Godfrey test is especially good to use where lagged values of the dependent variables are used as independent variables in the model's representation, as is the case in our ARIMA model and AR models generally. 
# * $H_o$: There is no serial correlation 
# * $H_a$: There is serial correlation present
# 
# The Breusch-Godfrey test for autocorrelation is performed very differently than the Ljung-Box and the popular Durbin-Watson test, and for that reason its important to use both when evaluating whether autocorrelation is present in our model's residuals.
# 
# <a id='sha-wil'></a>
# #### Shapiro-Wilks Test
# The Shapiro-wilks test evaluates a data sample and quantifies how likely it is that the data was drawn from a Gaussian distribution. 
# * $H_o$: The data is normally distributed
# * $H_a$: The data is not normally distributed
# 
# <a id='jar-ber'></a>
# #### Jarque-Bera Test
# The Jarque-Bera test is a type of lagrange multiplier test for normality. It is usually used for large data sets because other normality tests are not reliable when n is large, Shapiro-Wilk isn't reliable with n more than 2,000. Jarque-Bera specifically mathces skewness and kurtosis to a normal distribution.  
# * $H_o$: The data is normally distributed
# * $H_a$: The data does not come from a normal distribution

# In[ ]:


def model_diagnostics(residuals, model_obj):
    # For Breusch-Godfrey we have to pass the results object
    godfrey = acorr_breusch_godfrey(model_obj, nlags= 40)
    ljung = acorr_ljungbox(residuals, lags= 40)
    shap = shapiro(residuals)
    j_bera = jarque_bera(residuals)
    print('Results of Ljung-Box:')
    print('Null Hypothesis: No auotcorrelation')
    print('P-Value =< Alpha(.05) => Reject Null')
    print(f'p-values: {ljung[1]}\n')
    print('Results of Breusch-Godfrey:')
    print('Null Hypothesis: No auotcorrelation')
    print('P-Value =< Alpha(.05) => Reject Null')   
    print(f'p-values: {godfrey[1]}\n')
    print('Results of Shapiro-Wilks:')
    print('Null Hypothesis: Data is normally distributed')
    print('P-Value =< Alpha(.05) => Reject Null')   
    print(f'p-value: {shap[1]}\n')
    print('Results of Jarque-Bera:')
    print('Null Hypothesis: Data is normally distributed')
    print('P-Value =< Alpha(.05) => Reject Null')   
    print(f'p-value: {j_bera[1]}')

def plot_diagnostics(residuals):
    residuals.plot(title='ARIMA Residuals', figsize=(15, 10))
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].set_title('ARIMA Residuals KDE')
    ax[1].set_title('ARIMA Resduals Probability Plot')    
    residuals.plot(kind='kde', ax=ax[0])
    probplot(residuals, dist='norm', plot=ax[1])
    plt.show()  


# In[ ]:


best_parameters = (5, 2, 1)
model = ARIMA(pm10_df['bc_pm10'], order=best_parameters)
model_fit = model.fit(disp=-1)
resid = model_fit.resid

model_diagnostics(resid, model_fit)
plot_diagnostics(resid)


# <a id='conclusion'></a>
# # Conclusion
# ---
# In the end we have a (5, 2, 1) parameter ARIMA model that seems to fit our data- it has a root mean squared error of 3.2 and an AIC of 516, but from our Breusch-Godfrey and Ljung-Box test we still have autocorrelation in our residuals not captured by the model. This is an exercise in fitting ARIMA models and statistics in general- look below and you'll see the plot for our model prediction. In the data there looks to be seasonality and heteroskedasticity- and while relying on the `statsmodels` package may have led us to believe we had a viable model, we clearly do not. In the next two kernels of this mini time series series we will see SARIMAX, a model for dealing with seasonality, and GARCH, a model for dealing with *volatility clustering*. Real world datasets often have a variety of features that make them difficult to model. For now, we have a method, the Box-Jenkins, for fitting a model, we know about stationarity and how to test for it, and once we fit a model we know how to test the fit with model measurements and tests with the residuals. 

# In[ ]:


best_parameters = (5, 2, 1)
split_date = '2012-01-01'
data = pm10_df['bc_pm10']
train, test = data[:split_date], data[split_date:]
test_len = len(test)
model = ARIMA(train, order=best_parameters)
model_fit = model.fit(disp=-1)
prediction = model_fit.forecast(test_len)

pred_df = pd.DataFrame(prediction[0], index= test.index)
mae = mean_absolute_error(test, pred_df)
mse = mean_squared_error(test, pred_df)
rmse = math.sqrt(mse)

plt.figure(figsize=(20, 10))
plt.title('How ARIMA Fits Volatile Data', fontsize=30)
plt.plot(train, label='Train')
plt.plot(pred_df, label='Prediction')
plt.plot(test, label='Test')

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
plt.legend(fontsize= 25)
plt.show()

