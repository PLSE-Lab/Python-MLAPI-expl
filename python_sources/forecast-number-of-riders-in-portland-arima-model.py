#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/portland-oregon-average-monthly-.csv', index_col=0)
df.index.name=None
df.reset_index(inplace=True)
df.drop(df.index[114], inplace=True)


# In[ ]:


start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114)]
df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None


# In[ ]:


df.columns= ['riders']
df['riders'] = df.riders.apply(lambda x: int(x)*100)
df.shape


# In[ ]:


df.head()


# In[ ]:


data = [
    go.Scatter(
        y=df.riders,
        x=df.index
        
    )
]
layout = go.Layout(title='Riders Count',
                  yaxis=dict(title='Rider Count'),
                  xaxis=dict(title='Years'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='order count')


# ## Time Series Analysis

# In[ ]:


ts = df["riders"] 
ts.head(10)


# In[ ]:


ts['1978']


# ## Stationarity Check

# ### Dickey Fuller Test

# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(8,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


test_stationarity(ts)


# ### Using Log to get stationarity

# In[ ]:


ts_log = np.log(ts)
plt.plot(ts_log)


# In[ ]:


ts_smooth = ts_log.rolling(window = 12).mean()
plt.plot(ts_smooth, color = 'red')
plt.plot(ts_log)
plt.show()


# #### Case 1 : No subtraction

# In[ ]:


no_sub_ts = ts_smooth
no_sub_ts.dropna(inplace = True)
test_stationarity(no_sub_ts)


# #### Case 2: Subtraction

# In[ ]:


sub_ts = ts_log - ts_smooth
sub_ts.dropna(inplace = True)
test_stationarity(sub_ts)


# ### Exponential weighted average

# In[ ]:


expwighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(expwighted_avg, color='red')
plt.plot(ts_log)


# #### Case 1: No subtraction

# In[ ]:


no_sub_ts = expwighted_avg
no_sub_ts.dropna(inplace = True)
test_stationarity(no_sub_ts)


# #### Case 2: Subtraction

# In[ ]:


exp_ts_diff = ts_log-expwighted_avg
exp_ts_diff.dropna(inplace = True)
test_stationarity(exp_ts_diff)


# ### Differencing

# In[ ]:


ts_diff = ts_log - ts_log.shift(12)
ts_diff.dropna(inplace = True)
test_stationarity(ts_diff)


# #### Differencing + Exponential weighting

# In[ ]:


ts_diff_exp = ts_diff  - ts_diff.ewm(halflife = 12).mean()
ts_diff_exp.dropna(inplace = True)
test_stationarity(ts_diff_exp)


# ## Decomposition

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_diff_exp)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_diff_exp, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# ## Forecasting

# ### ARIMA model

# In[ ]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts_diff_exp.iloc[13:], lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts_diff_exp.iloc[13:], lags=20, ax=ax2)


# In[ ]:


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df.riders,
                                            order=param,
                                            seasonal_order=param_seasonal
                                            )

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[ ]:


mod = sm.tsa.statespace.SARIMAX(df.riders, order=(0, 1,0 ), seasonal_order=(1,1,1,12))
results = mod.fit()
print (results.summary())


# In[ ]:


df['forecast'] = results.predict(start = 97, end= 114, dynamic= True)  
df[['riders', 'forecast']].plot(figsize=(8, 5)) 
plt.savefig('ts_df_predict.png', bbox_inches='tight')


# In[ ]:


npredict =df.riders['1982'].shape[0]
fig, ax = plt.subplots(figsize=(8,4))
npre = 12
ax.set(title='Ridership', xlabel='Date', ylabel='Riders')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'riders'], 'g', label='Observed')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'forecast'], 'r', label='Dynamic forecast')
ax.grid(True)
legend = ax.legend(loc='best')
legend.get_frame().set_facecolor('w')
plt.savefig('ts_predict_compare.png', bbox_inches='tight')


# ### Future Prediction

# In[ ]:


start = datetime.datetime.strptime("1982-07-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= df.columns)
df = pd.concat([df, future])


# In[ ]:


df['forecast'] = results.predict(start = 113, end = 140, dynamic= True)  
df[['riders', 'forecast']].ix[-26:].plot(figsize=(10, 6)) 
plt.grid(True)
plt.savefig('ts_predict_future.png', bbox_inches='tight')


# We see that the number of riders increases when the new year starts due to everyone making new year resolutions and it gradually declines as the year passes.
