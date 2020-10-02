#!/usr/bin/env python
# coding: utf-8

# # Notebook, Time series analysis

# Create a summary of Time series analysis using Avocado price data as the subject.

# Previous notebooks
# 
# Notebooks<br>
# Classification method<br>
# https://www.kaggle.com/urayukitaka/notebook-classification-method<br>
# Regression method<br>
# https://www.kaggle.com/urayukitaka/notebook-regression-method<br>
# Dimension reduction method<br>
# https://www.kaggle.com/urayukitaka/notebook-dimension-reduction<br>
# Image preprocessing OpenCV library<br>
# https://www.kaggle.com/urayukitaka/notebook-image-preprocessing-opencv-library<br>
# Time series analysis<br>
# https://www.kaggle.com/urayukitaka/notebook-statistical-test-with-avocado-price<br>

# Time series analysis
# - Time series change
# - Rolling
# - Trend and seasonaly
# - Autocovariance and Autocorrelation
# - AR model
# - ARMA model
# - ARIMA model
# - SARIMA model
# - GARCH model

# In[ ]:


# Basic library
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Visualization
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns


# In[ ]:


# Time series analysis library
import statsmodels.api as sm
import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa import stattools as st
from statsmodels.tsa.arima_model import ARMA,ARIMA
get_ipython().system('pip install arch')
from arch import arch_model


# ## Data loading and Data check

# In[ ]:


# data loading
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv", header=0)


# In[ ]:


# data head
df.head()


# In[ ]:


# null value
df.isnull().sum()


# ### Preparing dataframe, average by date.

# In[ ]:


# Change date to datetime type and set index
df["Date"] = pd.to_datetime(df["Date"])


# In[ ]:


# pivot by type and Date
pivot = pd.pivot_table(df, index="Date", columns="type", values="AveragePrice", aggfunc="mean")


# In[ ]:


pivot.head() # Date is per week


# # Average price time series

# In[ ]:


# data set, separate conventional and organic
conv = pivot["conventional"]
orga = pivot["organic"]

# Visualization
plt.figure(figsize=(10,6))
plt.plot(conv.index, conv, label="conventional")
plt.plot(orga.index, orga, label="organic")
plt.xlabel("date")
plt.xticks(rotation=45)
plt.ylabel("Average price")
plt.legend()


# # Time series change

# In[ ]:


data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data["diff"] = data["price"].diff() # difference
data["ch_rate"] = data["price"].pct_change() # rate of change
data["log_ch_rate"] = np.log(data["price"]/data["price"].shift(1)) # Logarithmic change rate

data.set_index("date", inplace=True)


# In[ ]:


# difference Visualization
plt.figure(figsize=(20,6))
plt.plot(data.index, data["diff"], label="difference", color="blue", linewidth=1)


# In[ ]:


# rate of change Visualization
plt.figure(figsize=(20,6))
plt.plot(data.index, data["ch_rate"], label="rate of change", color="blue", linewidth=1)


# In[ ]:


# Logarithmic change rate Visualization
plt.figure(figsize=(20,6))
plt.plot(data.index, data["log_ch_rate"], label="Logarithmic change rate", color="blue", linewidth=1)


# # Rolling Statistics
# 
# ### object is average price of conventional

# In[ ]:


# calculation
window=4
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data["min"] = data["price"].rolling(window=window).min()
data["max"] = data["price"].rolling(window=window).max()
data["mean_short"] = data["price"].rolling(window=window).mean()
data["mean_long"] = data["price"].rolling(window=12).mean()

data.set_index("date", inplace=True)


# In[ ]:


# Visualization
plt.figure(figsize=(20,6))
plt.plot(data.index, data["price"], label="price", color="blue", linewidth=1)
plt.plot(data.index, data["min"], label="min", color="blue", linestyle='--', linewidth=0.5)
plt.plot(data.index, data["max"], label="max", color="blue", linestyle='--', linewidth=0.5)
plt.plot(data.index, data["mean_short"], label="mean_4_short", color="red", linestyle='-', linewidth=1)
plt.plot(data.index, data["mean_long"], label="mean_12_long", color="green", linestyle='-', linewidth=1)
plt.xlabel("date")
plt.xticks(rotation=45)
plt.ylabel("Price")
plt.legend()


# # Trend and seasonaly

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# Stats model
res = sm.tsa.seasonal_decompose(data["price"], freq=52)

# Decomposition
data["trend"] = res.trend
data["seaso"] = res.seasonal
data["resid"] = res.resid


# In[ ]:


# Visualization
fig, ax = plt.subplots(4,1, figsize=(20,15))
ax[0].plot(data.index, data["price"], label="price", color="blue", linewidth=1)
ax[0].set_xlabel("date")
ax[0].set_ylabel("price")

ax[1].plot(data.index, data["trend"], label="trend", color="blue", linewidth=1)
ax[1].set_xlabel("date")
ax[1].set_ylabel("trend")

ax[2].plot(data.index, data["seaso"], label="seasonaly", color="blue", linewidth=1)
ax[2].set_xlabel("date")
ax[2].set_ylabel("seasonaly")

ax[3].plot(data.index, data["resid"], label="residual error", color="blue", linewidth=1)
ax[3].set_xlabel("date")
ax[3].set_ylabel("residual error")


# # Autocovariance and Autocorrelation

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# plot with Stats model
plt.figure(figsize=(10,6))
plot_acf(data["price"], lags=52)
plt.show()


# In[ ]:


# plot with pandas
plt.figure(figsize=(10,6))
autocorrelation_plot(data["price"])
plt.show() # dash line is 99% confidence interval, solid line is 95% confidence interval


# In[ ]:


# plot with pandas 2
plt.figure(figsize=(10,6))
plt.acorr(data["price"]-data["price"].mean(), maxlags=52);


# # Partial autocorrelation coefficient

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# plot with Stats model
plt.figure(figsize=(10,6))
plot_pacf(data["price"], lags=52)
plt.show()


# # Unit root test

# In[ ]:


# ADF test
# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# Null hypothesis : The process is a unit root AR (p)
adf = sm.tsa.stattools.adfuller(data["price"])

print("p-value:{}".format(adf[1]))


# p-value > 0.05, this is not the process of unit roof AR(p)

# ## Time series prediction

# Confirm the time series prediction method such as ARIMA model. Therefore, we used the plan shown in the following figure, which can confirm a certain periodicity for the sample time series data.

# In[ ]:


# data set, separate conventional and organic
conv = pivot["conventional"][-102:]

# Visualization
plt.figure(figsize=(10,6))
plt.plot(conv.index, conv, label="conventional")
plt.xlabel("date")
plt.xticks(rotation=45)
plt.ylabel("Average price")
plt.legend()


# # AR model

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# train data is 2year data forward date.
train_data = data["price"][:-13]
date = data.index[:-13]

# with statsmodel 
ar = AR(train_data, dates=date).fit(maxlag=52, ic='aic')

# prediction is 
predict = ar.predict('2017-03-26','2018-03-25')


# In[ ]:


# Visualization
plt.figure(figsize=(10,6))
plt.plot(data.index, data["price"], label="raw data")
plt.plot(predict.index, predict, label="AR model future prediction", color="red")
plt.legend()


# # ARMA model

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# train data is 2year data forward date.
train_data = data["price"][:-13]
date = data.index[:-13]

# with statsmodel, aic check of params
st.arma_order_select_ic(train_data, ic='aic')


# In[ ]:


# predict with statsmodel
arma = ARMA(train_data, order=[4,2]).fit(maxlag=4, ic='aic', dates=date)

predict = arma.predict('2017-03-26','2018-03-25')


# In[ ]:


# Visualization
plt.figure(figsize=(10,6))
plt.plot(data.index, data["price"], label="raw data")
plt.plot(predict.index, predict, label="AR model future prediction", color="red")
plt.legend()


# # ARIMA model

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# train data is 2year data forward date.
train_data = data["price"][:-13]
date = data.index[:-13]


# In[ ]:


# predict with statsmodel, p,q are same as ARMA.
arima = ARIMA(train_data, order=[3,0,2],).fit(ic='aic', dates=date)

predict = arima.predict('2017-03-26','2018-03-25')


# In[ ]:


# Visualization
plt.figure(figsize=(10,6))
plt.plot(data.index, data["price"], label="raw data")
plt.plot(predict.index, predict, label="AR model future prediction", color="red")
plt.legend()


# # SARIMA model

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# train data is 2year data forward date.
train_data = data["price"][:-13]
date = data.index[:-13]


# ### optimizing sarima params, round-robin algorithm

# In[ ]:


# SARIMA params optimization, round-robin algorithm
# paramiter range
# order(p, d, q)
min_p = 1; max_p = 2 # min_p must be >1
min_d = 0; max_d = 2
min_q = 0; max_q = 2 

# seasonal_order(sp, sd, sq)
min_sp = 0; max_sp = 1
min_sd = 0; max_sd = 1
min_sq = 0; max_sq = 1

test_pattern = (max_p - min_p +1)*(max_q - min_q + 1)*(max_d - min_d + 1)*(max_sp - min_sp + 1)*(max_sq - min_sq + 1)*(max_sd - min_sd + 1)
print("pattern:", test_pattern)

sfq = 12 # seasonal_order
ts = train_data # training data

test_results = pd.DataFrame(index=range(test_pattern), columns=["model_parameters", "aic"])
num = 0
for p in range(min_p, max_p + 1):
    for d in range(min_d, max_d + 1):
        for q in range(min_q, max_q + 1):
            for sp in range(min_sp, max_sp + 1):
                for sd in range(min_sd, max_sd + 1):
                    for sq in range(min_sq, max_sq + 1):
                        sarima = sm.tsa.SARIMAX(
                            ts, order=(p, d, q), 
                            seasonal_order=(sp, sd, sq, sfq), 
                            enforce_stationarity = False, 
                            enforce_invertibility = False
                        ).fit()
                        test_results.iloc[num]["model_parameters"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), seasonal_order=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
                        test_results.iloc[num]["aic"] = sarima.aic
                        print(num,'/', test_pattern-1, test_results.iloc[num]["model_parameters"],  test_results.iloc[num]["aic"] )
                        num = num + 1

# result and AIC
print("best[aic] parameter ********")
print(test_results[test_results.aic == min(test_results.aic)])


# In[ ]:


test_results.sort_values(by='aic').head(10) 


# In[ ]:


# statsmodel with dsarima
sarimax = sm.tsa.SARIMAX(train_data, 
                        order=(1, 0, 0),
                        seasonal_order=(0, 0, 0, 4),
                        enforce_stationarity = False,
                        enforce_invertibility = False
                        ).fit(ic='aic', dates=date)

sarimax_optimization_resid = sarimax.resid # resid check

fig = plt.figure(figsize=(8, 8))

# ACF of resid
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(sarimax_optimization_resid, lags=40, ax=ax1) 

# PACF of resd
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(sarimax_optimization_resid, lags=40, ax=ax2) 
plt.show()


# In[ ]:


# prediction
predict = sarimax.predict('2017-03-26','2018-03-25')

# Visualization
plt.figure(figsize=(10,6))
plt.plot(data.index, data["price"], label="raw data")
plt.plot(predict.index, predict, label="AR model future prediction", color="red")
plt.legend()


# Neither model can predict the time series correctly. For models with such periodicity but strong non-stationarity, it may be inappropriate to make predictions with these models.

# # Reference) GARCH model

# In[ ]:


# Create dataframe
data = pd.DataFrame({})
data["date"] = conv.index
data["price"] = conv.values
data.set_index("date", inplace=True)

# train data is 2year data forward date.
train_data = data["price"][:-13]
date = data.index[:-13]


# In[ ]:


garch = arch_model(train_data, mean='AR', lags=4, vol='GARCH',
                  p=1, o=0, q=1, dist='studentst')
garch.fit()


# In[ ]:




