#!/usr/bin/env python
# coding: utf-8

# # ARIMA forecast for Coronavirus confirmed cases volume

# ## Dataset

# I will analyze and forecast coronavirus confirmed cases volume all over the world. Epidemy started in Wuhan in December 2019. On 2/11/2020, the virus is officially named COVID-19 by the World Health Organization.
# Data comes from: https://github.com/CSSEGISandData/COVID-19.
# 
# First we need to transform our dataset into series object containing date and comfirmed cases volumes.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df = df.fillna('unknow')
df.head()


# In[ ]:


df = df.append(df.sum(numeric_only=True), ignore_index=True)
df


# In[ ]:


df = df.iloc[df.shape[0] - 1][4:df.shape[1]]
df


# As our series object is ready, we will split it into two, one for model development (dataset.csv) and the other for validation - last week(validation.csv)
# 

# In[ ]:


split_point = len(df) - 4
dataset, validation = df[0:split_point], df[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header = False)
validation.to_csv('validation.csv', header = False)


# ## Persistence

# The first step before getting bogged in modeling is to establish a baseline of performance. This will provide a performance measure by which all more elaborate predictive models can be compared. Persistence is where the observation from the previous step is used as the prediction for the observation at the next time step

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
series = pd.read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# The example ends by printing the RMSE for the model. We can see that the persistence model achieved 4762. This means that on average, th model was wrong by about 4762 confirmed coronavirus cases for each prediction made.

# ## Summary Statistics

# In[ ]:


series.describe()


# Running the example provides a number of statistics to review:
# 
# *   The mean is about 26213, which we might consider our level in this series
# *   The standard deviation (average spread from the mean) is relatively large at 23240 cases.
# *   The percentiles along ith the standard deviation suggest a large spread to the data
# 
# 
# 
# 

# ## Line and density plots

# In[ ]:


series.plot()


# *   There is increasing trend over time which means that the dataset is non-stationary
# 
# 

# In[ ]:


plt.figure(1)
plt.subplot(211)
series.hist()
plt.subplot(212)
series.plot(kind='kde')
plt.show()


# 
# 
# *   The distribution is not Gaussian
# *   The distribution is left shifted and may be exponential or double Gaussian
# 
# 

# ## Manually cofigured ARIMA

# ARIMA(p, d, q) requires 3 parameters and is traditionally configured manually. We will try to *guess* probable values, starting from *d*.
# 
# The time series is non-stationary. We can make it stationary by differencing the series.

# In[ ]:


from statsmodels.tsa.stattools import adfuller

# create a differenced time series
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return pd.Series(diff)

# difference data
stationary = difference(X)
stationary.index = series.index[1:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv', header = False)


# Running an example outputs the result of statistical significance test of whether the 1-lag differenced series is stationary. Specifically, the augmented Dickey-Fuller test. The results show that te test static value -3.897 is smaller than the critical value at 5% of -2.992. This suggests we can reject null hypothesis and conclude that 1-lag differenced series is stationary. Then at least one level of differencing is required.  *d* >= 1
# 
# The next step is to select the lag values for the Autoregression (AR) and Moving Average (MA) parameters, *p* and *q* respectively. We can do this by reviewing Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plt.figure()
plt.subplot(211)
plot_acf(series, lags=24, ax=plt.gca())
plt.subplot(212)
plot_pacf(series, lags=24, ax=plt.gca())
plt.show()


# 
# 
# *   The ACF shows significant lag for 1-2 months
# *   The PACF does not show a significant lag
# 
# Good starting point will be p = 1 and q = 0
# 
# 
# 

# # Grid search for ARIMA

# We will use a grid search to explore all combinations of the ARIMA parameters and find the best one.

# In[ ]:


import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = range(0,13)
d_values = range(0, 4)
q_values = range(0, 13)
warnings.filterwarnings("ignore")
evaluate_models(X, p_values, d_values, q_values)


# In[ ]:


best_cfg = (0, 1, 0)


# # Review Residual Errors

# A good final check is to review residual forecast errors. Ideally, the distribution should be Gaussian with a zero mean.

# In[ ]:


train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=best_cfg)
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = pd.DataFrame(residuals)
plt.figure()
plt.subplot(211)
residuals.hist(ax=plt.gca())
plt.subplot(212)
residuals.plot(kind='kde', ax=plt.gca())
plt.show()


# ## Validate Model

# Now we can load the model and use it in a rolling - forecast manner, updating the transform and model for each time step. 

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import exp
from math import log
import numpy

# load and prepare datasets
dataset = pd.read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = dataset.values.astype('float32')
history = [x for x in X]
validation = pd.read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
y = validation.values.astype('float32')
# run model
model = ARIMA(history, order=best_cfg)
model_fit = model.fit(disp=0)
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0][0]
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
  # predict
  model = ARIMA(history, order=best_cfg)
  model_fit = model.fit(disp=0)
  yhat = model_fit.forecast()[0][0]
  predictions.append(yhat)
  # observation
  obs = y[i]
  history.append(obs)
  print('>Predicted=%i, Expected=%i' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %i' % rmse)
#predict next day
# predict
model = ARIMA(history, order=(0,1,0))
model_fit = model.fit(disp=0)
yhat = model_fit.forecast()[0][0]
predictions.append(yhat)
print('>Predicted next day volume=%i' % (yhat))
plt.plot(y)
plt.plot(predictions, color='red')
plt.show()


# Predicted number of confirmed coronavirus cases for the next day not available in dataset (currently 21.02) is 78807. A plot of the predictions compared to validation dataset is also provided. The forecast has the characteristics of a presistence forecast. This suggests that although this time serie has obvious tred , it is still reasonably difficult problem. 
