#!/usr/bin/env python
# coding: utf-8

# # **Bakers dozen - predicting transactions** ==================================================

# <img src="https://www.rd.com/wp-content/uploads/2017/10/this-is-why-there-are-13-in-a-baker-s-dozen_179437478_stevemart-1024x683.jpg" width="400px"/>

# ### --- ** Let's get started!**

# In[ ]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

sns.set(style="ticks")


# ### --- Import and examine the data

# In[ ]:


# import the data
df = pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


# examine example data
df.head()


# In[ ]:


# examine the entire set
df.info()


# ### --- Wrangle the data

# In[ ]:


# parse the dates
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['date'] = pd.to_datetime(df['Date'])

# deconstruct the dates
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# drop the date, time and transaction
df = df.drop(['Date', 'Time', 'Transaction'], axis=1)

# examine the data
df.head()


# ### --- Exploring transactions

# At this point, rather than concern ourselves with the items being sold in each transation we will instead first examine the number of items sold in each transaction and the number of transactions each day.

# In[ ]:


df2 = df.groupby(['date', 'year', 'month', 'day', 'weekday'], as_index = False).agg({'datetime': [pd.Series.nunique, 'count']})
df2.columns = ['date','year','month','day','weekday','transactions','items']
df2.head()


# Lets examine the relationship between transactions made each day and the number of items sold each day

# In[ ]:


x=df2['transactions']
y=df2['items']

# define a straight line fit with forced zero intercept
def lin_fit(x, y):
    fitfunc = lambda params, x: params[0] * x
    errfunc = lambda p, x, y: fitfunc(p, x) - y
    init_p = np.array((1.0))
    p1, success = optimize.leastsq(errfunc, init_p.copy(), args = (x, y))
    f = fitfunc(p1, x)
    return p1, success,  f  

# fit the line
p, s, f = lin_fit(x,y)
df2['fit'] = f

# print the fit parameter
print("Gradient = ", p[0])


# This shows that each transaction made equates to approximately 2.24 items sold.

# In[ ]:


data = []
weekdayDict = {6:'Sunday', 0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}

# construct plotly plot showing the relationship by day of sale,
trace = go.Scatter(x=df2['transactions'], y=df2['fit'], name = 'best fit')
data.append(trace)
for day in df2['weekday'].unique():
    tmp = df2[df2['weekday'] == day]
    trace = go.Scatter(x=tmp['transactions'], y=tmp['items'], mode='markers', name = weekdayDict[day])
    data.append(trace)
layout = go.Layout(
    xaxis=dict(title='Transactions made',titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')),
    yaxis=dict(title='Items sold',titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')))

# plot and embed in notebook
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# The figure above clearly shows that the linear fit captures the relationship between transactions and items sold reasonably well. Additionaly the figure shows that the number of transactions made is clearly a function of weekday with the most transactions clearly being made on Saturdays.

# Now, lets examine the timeseries of transactions to see if this pattern is evident...

# In[ ]:


# construct plotly plot showing the relationship by day of sale,
data = []
trace = go.Scatter(x=df2['date'], y=df2['transactions'])
data.append(trace)
layout = go.Layout(
    xaxis=dict(title='Date',titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')),
    yaxis=dict(title='Transactions made',titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')))

# plot and embed in notebook
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Clearly there is a strong seasonality in the data. In the next section we try to unpick it

# ### --- Modelling the seasonality of transactions

# In[ ]:


# Create a new data frame with a row per date
df3 = pd.DataFrame()
df3['date'] = pd.date_range(start=df2['date'].min(), end=df2['date'].max())

# Merge the two series to highlight any missing dates
df3 = pd.merge(df3,df2,how='left',on='date')

# Set the index and interpolate across missing data to create a uniform time series
df3.set_index(['date'], inplace=True)
df3['transactions'].interpolate(inplace=True)


# In[ ]:


resA = sm.tsa.seasonal_decompose(df3['transactions'], freq=7, model = "additive")
resM = sm.tsa.seasonal_decompose(df3['transactions'], freq=7, model = "multiplicative")

def plotseasonal(res, axes, col ):
    res.observed.plot(ax=axes[0], legend=False, color=col)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False, color=col)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False, color=col)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False, color=col)
    axes[3].set_ylabel('Residual')

fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(14,8))
plotseasonal(resA, axes[:,0], 'r')
plotseasonal(resM, axes[:,1], 'g')

plt.tight_layout()
plt.show()


# In[ ]:


df3['add_seasonality'] = resA.seasonal
df3['mul_seasonality'] = resM.seasonal

df3['av_trans'] = df3['transactions'].mean()
df3['add_forecast'] = df3['transactions'].mean() + df3['add_seasonality']
df3['mul_forecast'] = df3['transactions'].mean()*df3['mul_seasonality']


trace1 = go.Scatter(x=df3.index, y=df3['transactions'], name = 'Observed')
trace2 = go.Scatter(x=df3.index, y=df3['add_forecast'], line=dict(color=('rgb(105, 12, 24)'),width=2,dash='dash'), name='Additive model forecast')
trace3 = go.Scatter(x=df3.index, y=df3['mul_forecast'], line=dict(color=('rgb(60, 179, 13)'),width=2,dash='dot'), name = 'Multiplicative model forecast')
trace4 = go.Scatter(x=df3.index, y=df3['av_trans'], name = 'Average')
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(title='Date',titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')),
    yaxis=dict(title='Transactions made',titlefont=dict(family='Courier New, monospace',size=18,color='#7f7f7f')))

# plot and embed in notebook
fig = go.Figure(data=data, layout=layout)
iplot(fig)

print("Additive median % accuracy", 100.*(1-(abs(df3['transactions'] - df3['add_forecast'])/df3['transactions']).median()))
print("Multiplicative median % accuracy", 100.*(1-(abs(df3['transactions'] - df3['mul_forecast'])/df3['transactions']).median()))


# Hurrah, simply by measuring the seasonality we have been able to turn an average transaction day to a transaction forecast based on the given weekday, and it applying it across the data set, gives a median percentage accuracy of greater than 87%.
