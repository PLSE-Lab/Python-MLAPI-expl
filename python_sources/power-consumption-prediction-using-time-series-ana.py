#!/usr/bin/env python
# coding: utf-8

# # Power Consumption Prediction using Time Series Analysis

# We start off by importing functions and classes we'd need. Sice the data is in the form of a time series, statsmodels comes in handy here as it provides functions such as plot_acf, seasonal_decompose etc, which are extremely handy during initial analysis. It also has ARIMA models like SARIMAX and ARIMA which help us while contructing the regression model itself.

# In[ ]:


get_ipython().system('pip install pmdarima')

import pandas as pd
import numpy as np
from matplotlib import pyplot
from pylab import rcParams

from statsmodels.tsa.seasonal import seasonal_decompose as SDecompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima

pd.options.mode.chained_assignment = None

df = pd.read_csv('/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt', delimiter=';', 
                na_values=['nan','?'], dtype={'Date':str,'Time':str,'Global_active_power':np.float64,
                'Global_reactive_power':np.float64, 'Voltage':np.float64 ,'Global_intensity':np.float64,
                'Sub_metering_1':np.float64, 'Sub_metering_2':np.float64,'Sub_metering_3': np.float64})


# We start off by setting the index value for the entire dataframe by using DateTime which gives us one less column to analyse and a more accurate index. Next we get a look at plots of the first two columns.

# In[ ]:


df['DTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.drop(['Date', 'Time'], axis=1)
df = df.set_index('DTime')

df['Global_active_power'].plot(figsize=(20, 5))
df['Global_reactive_power'].plot()
pyplot.show()


# Since this plot is very dense, we create another dataframe which resamples datapoints for each week and takes the mean for them; we now have a time series sampled_df which can be used for our analysis. We can also then plot these resampled versions.

# In[ ]:


print(df.isna().sum())
sampled_df = df.bfill().resample('W').mean()
print(sampled_df.corr())


# Since we see a great number of null values, we use backfilling to fill each empty value with the next value on the column. We call it before we resample and then get the correlation of the new dataframe. 
# 
# Looking at the correlation for all the columns, we see that the global_active power and global_intensity are heavily correlated while the others are not. So if we're looking to predict Global_reactive_power, we can just include only one of these two columns.

# In[ ]:


sampled_df.Global_reactive_power.plot(figsize=(20, 10), color='y', legend=True)
sampled_df.Global_active_power.plot(color='r', legend=True)
sampled_df.Sub_metering_1.plot(color='b', legend=True)
sampled_df.Global_intensity.plot(color='g', legend=True)
pyplot.show()


# ### Seasonality and Stationarity

# The plot shows us a bit of seasonality in the dataframe and it can be analysed more by looking into the decomposition of the series. Using statsmodel's seasonal_decompose, we can try to find trend, seasonal and residual(noise) components and analyze for more insight. 
# 
# Having seasonality will tell us that our model isn't stationary and if we assume that then the residual must resemble white noise.

# In[ ]:


rcParams['figure.figsize'] = 15, 6

fig, axes = pyplot.subplots(4, 2)

mul_decomposition = SDecompose(sampled_df.Global_reactive_power, model='multiplicative')
add_decomposition = SDecompose(sampled_df.Global_reactive_power, model='additive')

axes[0][0].plot(mul_decomposition.observed)
axes[0][0].set_title("Multiplicative Decomposition")
axes[1][0].plot(mul_decomposition.trend)
axes[2][0].plot(mul_decomposition.seasonal)
axes[3][0].plot(mul_decomposition.resid)

axes[0][1].plot(add_decomposition.observed)
axes[0][1].set_title("Additive Decomposition")
axes[1][1].plot(add_decomposition.trend)
axes[2][1].plot(add_decomposition.seasonal)
axes[3][1].plot(add_decomposition.resid)
pyplot.show()


# Seasonal_decompose also allows us to choose between an additive or a multiplicative model. 
# We see the trend being the same across both the decompositions. However, the seasonality is significant and stable, also it doesn't seem to be increasing over time. This implies that the seasonal variations are simply added to the trend and not multiplied.
# 
# If we look at the Y-Axis range, we see how the residuals are greater in the multiplicative model. This means there's more information in the series that it may not be able to model well enough and instead just classifies as noise. 
# This hints at the additive model being better for our series.
# 
# Since we know our series has seasonality, we know it will require a model which analyses the seasonal component differently, like a SARIMA or SARIMAX model. We know the frequency of our series is weekly so we can use the seasonality to be 7. For the other parameters, p, d, q and P, D, Q for the seasonal component, we need to look at their ACF and PACF plots. 
# 
# Now before we go further, it makes sense to split sampled_df into test and training sets so that we do not accidentally use any insight obtained by plotting the test set while designing our model. We go with a 75%-25% split.

# In[ ]:


split = int(0.75 * len(sampled_df))
sampled_train, sampled_test = sampled_df[:split], sampled_df[split:]

plot_acf(sampled_train.Global_reactive_power, lags=30, zero=False)
plot_pacf(sampled_train.Global_reactive_power, lags=30, zero=False)

pyplot.show()


# The ACF function for the data shows high significance (ACF values outside the blue area) for lags up till 10 and the PACF shows significance till 4. This tells us that if we were to construct an ARMA model, p would be in the range 0 to 4 while q would be in the range 0 to 10. 
# 
# Now the next thing to do is to check stationarity and how many times the time series would need to be differenced for us to remove the stationarity in our time series. The Dickey Fuller checks our data against the null hypothesis that it is not stationary.

# In[ ]:


print(adfuller(sampled_train.Global_reactive_power))

shifted_power = sampled_train.Global_reactive_power.diff(1)[1:]

print(adfuller(shifted_power))


# The high test-statistic (-2.19) falls outside the 10% check and the p-value of 0.2 means the null hypothesis is not rejected and the series is not stationary. But if we shift the datapoints by just 1, we can see that it becomes stationary with an extremely small p value and a very low test statistic. This means that the order of integration (or d) for the model can be chosen as 1, since we only had to difference our model once to get a stationary model.

# ### Creating SARIMA models
# 
# Now we can start modeling by using a basic SARIMA model that uses (1, 1, 1), (1, 1, 1, 7) as (p, d, q)(P, D, Q, s) parameters respectively.

# In[ ]:


params, seasonal_params = (1, 1, 1), (1, 0, 1, 7)

mod = SARIMAX(sampled_train.Global_reactive_power, order=params, seasonal_order=seasonal_params, 
              enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
results.summary()


# The log likelihood for any model needs to be high while the AIC needs to be low. Since we haven't tried out other models, we don't yet know if the current score is great or not. However, we can also udge the fit by the coefficients and whether they have significant predictive powers or not.
# 
# So the coefficients for ma.L1, ma.S.L7 and sigma2 are significant as implied by the P>|z| value being 0, but that for ar.L1 does not seem to pass this test. Changing the parameters to different values will help in moving towards a model with a better fit. Moreover, including other columns from the dataset, like voltage and global_intensity, might boost it's performance.

# An easy of doing so is using the pmdarima module for it's auto_arima class, which keeps iterating over multiple parameter combinations to give us one with the best fit. First, we start off with the same range of p, q and P, Q as we had decided before but no exogenous (external) variables. We will also decide between it having a constant+linear trend (trend='ct') or having just linear trend (trend='t'), but for now we use only trend.  

# In[ ]:


model_auto = auto_arima(sampled_train.Global_reactive_power, max_order=None, max_p=4, max_q=10, max_P=4, max_Q=10, 
                max_D=1, m=7, alpha=0.05, trend='t', information_criteria='oob', out_of_sample=int(0.02*len(sampled_train)),
                maxiter=200, suppress_warnings=True)

model_auto.summary()


# We see that the winning model is a SARIMA with order (0, 1, 2) and seasonal_order (0, 0, 1, 7). Also this model gives better log likeihood for the model and lower AIC, along with significant coefficients for some of the layers, but not all. For example, the drift (which represents the slop of the linear function) has a value very close to 0 and pvalue 0.859, implying it doesn't contribute much to the forecasting.
# 
# Let's try again with exogenous variables, ocne with trend='t' and once with trend='ct'.

# In[ ]:


model_auto = auto_arima(sampled_train.Global_reactive_power, exogenous=sampled_train[['Global_intensity', 'Sub_metering_1',  
                'Sub_metering_2', 'Sub_metering_3', 'Voltage']], max_order=None, max_p=4, max_q=10, max_P=4, max_Q=10, 
                max_D=1, m=7, alpha=0.05, trend='ct', information_criteria='oob', out_of_sample=int(0.02*len(sampled_train)),
                maxiter=200, suppress_warnings=True)

model_auto.summary()


# In[ ]:


model_auto = auto_arima(sampled_train.Global_reactive_power, exogenous=sampled_train[['Global_intensity', 'Sub_metering_1',  
                'Sub_metering_2', 'Sub_metering_3', 'Voltage']], max_order=None, max_p=4, max_q=10, max_P=4, max_Q=10, 
                max_D=1, m=7, alpha=0.05, trend=None, information_criteria='oob', out_of_sample=int(0.02*len(sampled_train)),
                maxiter=200, suppress_warnings=True)

model_auto.summary()


# We see that the models it picks out as best performing are both simple moving average models of the order 2 with no differencing and lower AIC than before. The one with constant and linear trend has higher AIC than the other as well as insignificant values for both intercept and drift. After comparing both, we pick the latter as the more favorable chocie, due to marginally lower AIC.
# 
# However, wee see that out model has no seasonal component despite it being very evident in the plots we had created. We may attribute this to using the other columns, like Global_intensity and Submetering_1 and Submetering_3, both of which have significant coefficients. Since they themselves are time series with seasonalities of their own, it makes sense that the model obtained the information about seasonal variations from those columns itself.  

# In[ ]:


sampled_train['Predicted_Global_reactive_power'] = pd.DataFrame(model_auto.predict_in_sample(exogenous=
    sampled_train[['Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Voltage']]), 
    index = sampled_train.Global_reactive_power.index, columns=['Global_reactive_power'])
sampled_test['Predicted_Global_reactive_power_test'] = pd.DataFrame(model_auto.predict(n_periods=53, exogenous=
    sampled_test[['Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Voltage']]), 
     index = sampled_test.Global_reactive_power.index, columns=['Global_reactive_power'])

ax = sampled_train.Global_reactive_power.plot(figsize=(20, 5), color='red', legend=True)
sampled_test.Global_reactive_power.plot(ax=ax, color='red')

sampled_train['Predicted_Global_reactive_power'].plot(color='blue', ax=ax, legend=True)
sampled_test['Predicted_Global_reactive_power_test'].plot(color='green', ax=ax, legend=True)

pyplot.show()


# Plotting the actual data from the training set against the predicted values shows that while the model isn't very good at adjusting to shocks in the series, it performs well overall. The same can be said of the predicted values.
# 
# To see just how much data is getting left out of our predictions, let's look at the residuals of the model.

# In[ ]:


resid = model_auto.resid()
resid_test = sampled_test['Predicted_Global_reactive_power_test']-sampled_test['Global_reactive_power']

ax=sampled_train['Predicted_Global_reactive_power'].plot(color='green')
sampled_test['Predicted_Global_reactive_power_test'].plot(color='green', ax=ax)

resid.plot(ax=ax, color='blue')
resid_test.plot(ax=ax, color='black')

ax.set_title('Residuals vs Predicted Values')
ax.legend(['Predicted Values', 'Residual in training set', 'Residual in testing set'])

pyplot.show()


# Plotting the residuals tells us how much information our model was simply not able to learn. For a good model, it must not have any seasonality and must look like white noise, i.e. mean=0 and constant covariance. Looking at the residuals, we see clearly that it resembles white noise and it's magnitude isn't far off from the residuals we obtained by using seasonal_decompose.
# 
# So fianlly, we successfully obtain a model that is able to understand and predict our time series. 
