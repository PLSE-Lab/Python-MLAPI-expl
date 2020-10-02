#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This notebook will focus on time series EDA on the variables in our dataset. While the original data table is per transaaction basis, we can derive hourly / daily summary of is_fraud and other predictive variables. This has two purposes: 
# * (1) Better understanding of underlying variables 
# * (2) Use time series metrics as additional features for our prediction model

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
pd.options.display.precision = 15

import time
import datetime
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import json
# import altair as alt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# alt.renderers.enable('notebook')

import plotly.graph_objs as go
# import plotly.plotly as py
import plotly.offline as pyo
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly_express as px
init_notebook_mode(connected=True)
from matplotlib import cm
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly_express as px


# # Loading data

# In[ ]:


folder_path = '../input/ieee-fraud-detection//'
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')
# test_identity = pd.read_csv(f'{folder_path}test_identity.csv')
# test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')
sub = pd.read_csv(f'{folder_path}sample_submission.csv')
# let's combine the data and work with the whole dataset
# I will save this for later
# train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
# test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[ ]:


print(f'Train Transaction dataset has {train_transaction.shape[0]} rows and {train_transaction.shape[1]} columns.')


# I also sample down the dataset since this is an EDA and it will accelerate data processing and visualization

# In[ ]:


train_transaction = train_transaction.sample(n=10000)
train_transaction.head()


# In[ ]:


print(f'Train Transaction dataset has {train_transaction.shape[0]} rows and {train_transaction.shape[1]} columns.')


# # Create datetime feature
# 
# I use some codes from the following notebooks which provides a handy datetime feature creation. Based on their EDA, the notebook also found that the datetime is likely starting at 1-Dec-2017
# * https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
# * https://www.kaggle.com/kevinbonnes/transactiondt-starting-at-2017-12-01

# In[ ]:


def make_day_feature(df, offset=0, tname='TransactionDT'):
    """
    Creates a day of the week feature, encoded as 0-6. 
    
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    offset : float (default=0)
        offset (in days) to shift the start/end of a day.
    tname : str
        Name of the time column in df.
    """
    # found a good offset is 0.58
    days = df[tname] / (3600*24)        
    encoded_days = np.floor(days-1+offset) % 7
    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23. 
    
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    tname : str
        Name of the time column in df.
    """
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')


# In[ ]:


train_transaction['TransactionDateTime'] = train_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
train_transaction['TransactionDate'] = [x.date() for x in train_transaction['TransactionDateTime']]
train_transaction['TransactionHour'] = train_transaction.TransactionDT // 3600
train_transaction['TransactionHourOfDay'] = train_transaction['TransactionHour'] % 24
train_transaction['TransactionDay'] = train_transaction.TransactionDT // (3600 * 24)


# In[ ]:


train_transaction.head(10)


# # Define groups of metrics

# In[ ]:


trx_colnames = train_transaction.columns
trx_colnames_core_num = ['isFraud', 'TransactionAmt', 'card1','card2', 'card3','card5']
trx_colnames_core_cat = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
trx_colnames_C = [c for c in trx_colnames if c.startswith("C") ]
trx_colnames_V = [c for c in trx_colnames if c.startswith("V") ]
trx_colnames_M = [c for c in trx_colnames if c.startswith("M") ]


# # Time Series EDA based on: 

# # Core Metrics - Hourly Time Series

# In[ ]:


agg_dict = {}
for col in trx_colnames_core_num:
    agg_dict[col] = ['mean','sum']
train_trx_hour = train_transaction.groupby(['TransactionHour']).agg(agg_dict).reset_index()
train_trx_hour.columns = ['_'.join(col).strip() for col in train_trx_hour.columns.values]
train_trx_hour.head()


# In[ ]:


import math
df = train_trx_hour
plotted_columns = train_trx_hour.columns[1:]
lencols = len(plotted_columns)
fig, axes = plt.subplots(math.ceil(lencols//3),3,figsize=(12,lencols))
for i, metrics in enumerate(plotted_columns):
    df.plot(x='TransactionHour_',y=metrics,title=metrics + " by Hour",ax=axes[i//3,i%3])
plt.tight_layout()
plt.suptitle("Core Metrics on Hourly Basis",y="1.05")
fig.show()


# ### Some interesting observations:
# * Card5 mean have a common cap at $225, and the spiky variation tend to be on the negative direction. So card5 is probably a "Maximum Balance" / "Remaining Balance" type of variables
# * Card3 mean has a common band between 150 and 185
# * Card1 and Card2 mean hugely varied, and they're at different scales

# # Core Metrics - Daily time series 

# In[ ]:


agg_dict = {}
for col in trx_colnames_core_num:
    agg_dict[col] = ['mean','sum']
train_trx_date = train_transaction.groupby(['TransactionDate']).agg(agg_dict).reset_index()
train_trx_date.columns = ['_'.join(col).strip() for col in train_trx_date.columns.values]
train_trx_date.head()


# In[ ]:


pd.plotting.register_matplotlib_converters()
import math
df = train_trx_date
plotted_columns = df.columns[1:]
lencols = len(plotted_columns)
fig, axes = plt.subplots(math.ceil(lencols//3),3,figsize=(16,lencols))
for i, metrics in enumerate(plotted_columns):
    df.plot(x='TransactionDate_',y=metrics,title=metrics + " by Date",ax=axes[i//3,i%3])
plt.tight_layout()
plt.suptitle("Core Metrics on Date Basis",y="1.05")
fig.show()


# # Core Metrics - Seasonal Decomposition using Prophet
# 
# Next, I will do seasonal decomposition using Facebook's prophet library. Because the data is not yet 2 years, we couldn't extract yearly seasonality. Nonetheless, we are still able to extract the DayOfWeek seasonality, which is quite significant in some metrics. 

# In[ ]:


from fbprophet import Prophet
prophet = Prophet()

df = train_trx_date[['TransactionDate_','isFraud_sum']]
df.columns = ['ds','y']
prophet = Prophet()
prophet.fit(df)
forecast = prophet.predict(df)


# In[ ]:


from fbprophet import Prophet

def plot_forecast(df_input,metrics):
    prophet = Prophet()
    df = df_input[['TransactionDate_',metrics]]
    df.columns = ['ds','y']
    prophet.fit(df)
    forecast = prophet.predict(df)
    fig,ax = plt.subplots(1,3,figsize=(20,5),sharey=True)
    forecast.weekly[:7].plot(ax=ax[0])
    ax[0].set_title("weekly component")
    forecast.trend.plot(ax=ax[1])
    ax[1].set_title("trend component")
#     ax[1].xticks(forecast.ds)
    ax[2].plot(forecast.yhat)
    ax[2].plot(df.y)
    ax[2].set_title("comparing fitted vs. actual")
    plt.suptitle(metrics + ' based on: Day Of Week, Long-term Trend, Seasonality vs. Actual')


# In[ ]:


for metrics in plotted_columns:
    plot_forecast(train_trx_date,metrics)


# # C Metrics - Daily time series 

# In[ ]:


agg_dict = {}
for col in trx_colnames_C:
    agg_dict[col] = ['mean','sum']
train_trx_date = train_transaction.groupby(['TransactionDate']).agg(agg_dict).reset_index()
train_trx_date.columns = ['_'.join(col).strip() for col in train_trx_date.columns.values]
train_trx_date.head()


# In[ ]:


import math
pd.plotting.register_matplotlib_converters()
df = train_trx_date
plotted_columns = df.columns[1:]
lencols = len(plotted_columns)
fig, axes = plt.subplots(math.ceil(lencols//3)+1,3,figsize=(16,lencols))
for i, metrics in enumerate(plotted_columns):
    df.plot(x='TransactionDate_',y=metrics,title=metrics + " by Date",ax=axes[i//3,i%3])
plt.tight_layout()
plt.suptitle("Core Metrics on Date Basis",y="1.05")
fig.show()


# #### The spike is clearly the Christmas holiday (25-Dec). Probably removing those holiday dates would help to generalize the training data. I do wonder what these C variables could be, since some of them have very low mean (<10) in all of the days, except during the holiday
# 
# #### Some of these variables also demonstrate spikes on several dates. C3 is the most prominent one, because it is usually 0, except for a couple of days. It is probably some kind of indicator of rare event.

# # C Metrics - Seasonal decomposition

# I will only be taking the time series for data after 1-Jan, since the holiday will really skew the decomposition. Alternatively, I can add in the holiday date into Prophet specification, but I'm not doing that for now

# In[ ]:


import datetime
t = datetime.date(2018,1,1)
train_trx_date2 = train_trx_date[train_trx_date['TransactionDate_']>t].reset_index(drop=True)
train_trx_date2.head()


# In[ ]:


for metrics in plotted_columns:
    plot_forecast(train_trx_date2,metrics)


# # C Metrics - Hour of Day Analysis

# In[ ]:


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

agg_dict = {}
for col in trx_colnames_core_num:
    agg_dict[col] = ['mean','sum']
train_trx_HOD = train_transaction.groupby(['TransactionHourOfDay']).agg(agg_dict).reset_index()
train_trx_HOD.columns = ['_'.join(col).strip() for col in train_trx_HOD.columns.values]
train_trx_HOD.head()

agg_dict = {}
for col in trx_colnames_core_num:
    agg_dict[col] = [percentile(25),'median',percentile(75)]
train_trx_HOD2 = train_transaction.groupby(['TransactionHourOfDay']).agg(agg_dict).reset_index()
train_trx_HOD2.columns = ['_'.join(col).strip() for col in train_trx_HOD2.columns.values]
train_trx_HOD2.head()


# In[ ]:


pd.plotting.register_matplotlib_converters()
import math
df = train_trx_HOD
plotted_columns = df.columns[1:]
lencols = len(plotted_columns)
fig, axes = plt.subplots(math.ceil(lencols//2),2,figsize=(16,lencols))
for i, metrics in enumerate(plotted_columns):
    df.plot(x='TransactionHourOfDay_',y=metrics,title=metrics + " by HourOfDay",ax=axes[i//2,i%2])
plt.tight_layout()
plt.suptitle("Core Metrics by HourOfDay",y="1.05")
fig.show()

