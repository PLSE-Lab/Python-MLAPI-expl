#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 15, 6

import statsmodels.api as sm


# In[ ]:


# reference: https://logics-of-blue.com/wp-content/uploads/2017/05/python-state-space-models.html


# In[ ]:


df = pd.read_csv("../input/rossmann-store-sales/train.csv", parse_dates = True, index_col = 'Date')
df.head()


# In[ ]:


data = df['Sales'].groupby('Date').sum()
data.head()


# In[ ]:


thres = '2014-12-31'
train = data[data.index <= thres]
test = data[data.index > thres]


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


test.tail()


# In[ ]:


plt.plot(train)


# In[ ]:


plt.plot(test)


# # Mean absolute percentage error (MAPE)

# $$MAPE=\frac{1}{n}\sum_{t=1}^{n}\left |\frac{y_t - \hat{y}_t}{y_t}\right|$$

# # Root Mean Square Percentage Error (RMSPE)

# $$\textrm{RMSPE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} \left(\frac{y_t - \hat{y}_t}{y_t}\right)^2}$$
# where $y_t$ is the actual value and $\hat{y}_t$ is the forecast value.

# In[ ]:


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[ ]:


def rmspe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true)**2))


# # State space models

# ## Local level model

# In[ ]:


mod_local_level = sm.tsa.UnobservedComponents(train, 'local level', freq='D')

res_local_level = mod_local_level.fit()

print(res_local_level.summary())
plt.rcParams['figure.figsize'] = 24, 20
res_local_level.plot_components();


# In[ ]:


pred_local_level = res_local_level.predict('2015-01-01', '2015-07-31')
plt.plot(test)
plt.plot(pred_local_level, 'r');
print("MAPE = ", mape(test, pred_local_level))
print("RMSPE = ", rmspe(test, pred_local_level))


# ## Local linear trend model

# In[ ]:


mod_trend = sm.tsa.UnobservedComponents(train, 'local linear trend', freq='D')

res_trend = mod_trend.fit()

print(res_trend.summary())
plt.rcParams['figure.figsize'] = 24, 20
res_trend.plot_components();


# In[ ]:


pred_trend = res_trend.predict('2015-01-01', '2015-07-31')
plt.rcParams['figure.figsize'] = 24, 6
plt.plot(test)
plt.plot(pred_trend, 'r');
print("MAPE = ", mape(test, pred_trend))
print("RMSPE = ", rmspe(test, pred_trend))


# ## Seasonal local level model

# In[ ]:


mod_season_local_level = sm.tsa.UnobservedComponents(train, 'local level', freq='D', seasonal=12)

res_season_local_level = mod_season_local_level.fit()

print(res_season_local_level.summary())
plt.rcParams['figure.figsize'] = 24, 20
res_season_local_level.plot_components();


# In[ ]:


pred_season_local_level = res_season_local_level.predict('2015-01-01', '2015-07-31')
plt.rcParams['figure.figsize'] = 24, 6
plt.plot(test)
plt.plot(pred_season_local_level, 'r');
print("MAPE = ", mape(test, pred_season_local_level))
print("RMSPE = ", rmspe(test, pred_season_local_level))


# ## Seasonal local linear trend model

# In[ ]:


mod_season_trend = sm.tsa.UnobservedComponents(train, 'local linear trend', freq='D', seasonal=12)

res_season_trend = mod_season_trend.fit()

print(res_season_trend.summary())
plt.rcParams['figure.figsize'] = 24, 20
res_season_trend.plot_components();


# In[ ]:


pred_season_trend = res_season_trend.predict('2015-01-01', '2015-07-31')
plt.rcParams['figure.figsize'] = 24, 6
plt.plot(test)
plt.plot(pred_season_trend, 'r');
print("MAPE = ", mape(test, pred_season_trend))
print("RMSPE = ", rmspe(test, pred_season_trend))


# In[ ]:


mod_season_trend = sm.tsa.UnobservedComponents(train, 'local linear trend', freq='D', seasonal=12)

#res_season_trend = mod_season_trend.fit()
res_season_trend = mod_season_trend.fit(
    method='bfgs',
    maxiter=500,
    start_params=mod_season_trend.fit(method='nm', maxiter=500).params
)

print(res_season_trend.summary())
plt.rcParams['figure.figsize'] = 24, 20
res_season_trend.plot_components();


# In[ ]:


pred_season_trend = res_season_trend.predict('2015-01-01', '2015-07-31')
plt.rcParams['figure.figsize'] = 24, 6
plt.plot(test)
plt.plot(pred_season_trend, 'r');
print("MAPE = ", mape(test, pred_season_trend))
print("RMSPE = ", rmspe(test, pred_season_trend))

