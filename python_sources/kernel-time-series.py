#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Now, we will load the data set and look at some initial rows and data types of the columns:
#data = pd.read_csv('AirPassengers.csv')
df = pd.read_csv("../input/timeseries/time_series1.csv", names=['value'], header=0)
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()


# In[ ]:


import pandas as pd
series = pd.read_csv("../input/timeseries/time_series1.csv", names=['value'], header=0)


# In[ ]:


from pandas.plotting import lag_plot
lag_plot(series)
pyplot.show()


# In[ ]:


from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[ ]:


from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
series = read_csv("../input/timeseries/time_series1.csv", names=['value'], header=0)
autocorrelation_plot(series)
pyplot.show()


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/timeseries/time_series1.csv", names=['value'], header=0)
print(df)


# In[ ]:


from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=31)
pyplot.show()


# **#persistance model for time series model**
# 

# In[ ]:


from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
series = read_csv('../input/timeseries/time_series1.csv', header=0, index_col=0)
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return x

# walk-forward validation


predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()



# In[ ]:


print(train)
print(test)
print(predictions)
print(x)


# # autoregression with prediction
# 

# In[ ]:


import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt


# In[ ]:


def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')
series = pd.read_csv('../input/timeseries2/time_series2.csv',index_col=0,parse_dates=[0],date_parser=parser)
series.head()


# In[ ]:


series.plot()


# #auto regression model using 7 records
# 

# In[ ]:


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')
series = pd.read_csv('../input/timeseries2/time_series2.csv',index_col=0,parse_dates=[0],date_parser=parser)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# above red colr as predicted and blue color as a real values
# 

# 

# In[ ]:


print(train)
print('all')
print(X)


# #continuesly new observation
# 

# In[ ]:


import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from pandas import datetime

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')
series = pd.read_csv('../input/timeseries2/time_series2.csv',index_col=0,parse_dates=[0],date_parser=parser)

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

sns.lineplot(data=data, palette="tab10", linewidth=2.5)


# In[ ]:


import seaborn as sns
sns.set(style="ticks")

dots = sns.load_dataset("dots")

# Define a palette to ensure that colors will be
# shared across the facets
palette = dict(zip(dots.coherence.unique(),
                   sns.color_palette("rocket_r", 6)))

# Plot the lines on two facets
sns.relplot(x="time", y="firing_rate",
            hue="coherence", size="choice", col="align",
            size_order=["T1", "T2"], palette=palette,
            height=5, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=dots)


# In[ ]:


get_ipython().system('pip install sunpy')


# In[ ]:


get_ipython().system('pip install sunpy[all]')


# In[ ]:


import sunpy.data.sample
import sunpy.map
aia = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
aia.peek()


# In[ ]:


autoplot(uschange[,"Consumption"]) +
  xlab("Year") + ylab("Quarterly percentage change")


# #ARIMA model

# In[ ]:


import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from pandas import datetime
import warnings
import itertools
from pandas import read_csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


import statsmodels.api as sm


# In[ ]:


def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')
data = pd.read_csv('../input/timeseries2/time_series2.csv',index_col=0,parse_dates=[0],date_parser=parser)


# In[ ]:


y = data
y.plot(figsize=(15, 6))
plt.show()


# #The ARIMA Time Series Model
# Parameter Selection for the ARIMA Time Series Model

# In[ ]:


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# # Fitting an ARIMA Time Series Model

# In[ ]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# In[ ]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# 

# #Validating Forecasts

# In[ ]:


y


# In[ ]:


pred = results.get_prediction(start=pd.to_datetime('2008-06-01'), dynamic=False)
pred_ci = pred.conf_int()


# In[ ]:


ax = y['2000':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

plt.show()


# In[ ]:


y_forecasted = pred.predicted_mean
y_truth = y['2008-06-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[ ]:


pred_dynamic = results.get_prediction(start=pd.to_datetime('2008-06-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


# In[ ]:


ax = y['2000':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()


# In[ ]:


# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['2008-06-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[ ]:


# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()


# In[ ]:


ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('USERS')

plt.legend()
plt.show()


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
sns.lineplot(data=data, palette="tab10", linewidth=2.5)


# In[ ]:





# # matploatlib 538

# In[ ]:


import pandas as pd
direct_link = 'http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv'
women_majors = pd.read_csv(direct_link)
print(women_majors.info())
women_majors.head(20)


# In[ ]:


under_20 = women_majors.loc[0, women_majors.loc[0] < 20]
under_20


# In[ ]:


under_20_graph = women_majors.plot(x = 'Year', y = under_20.index, figsize = (12,8))
print('Type:', type(under_20_graph))

