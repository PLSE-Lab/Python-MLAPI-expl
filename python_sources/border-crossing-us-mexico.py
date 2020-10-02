#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/marharyta98/border-crossing-eda - Thx for your kernel!:)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
import datetime

import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


df_non_filtered = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")


# In[ ]:


df = df_non_filtered.query('Border == "US-Mexico Border"')
print(df)


# In[ ]:


df.shape


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


print('Attribute '+ 'Values')
for i in df.columns:
    print( i,len(df.loc[:,i].unique()) )


# ### Also, there are almost duble the locations than the port codes. This can mean that a port useually has 2 locations asociated with it. Let's see this.

# In[ ]:


indexes = df['Location'].drop_duplicates().index
temp = df.iloc[indexes].groupby(by='Port Code')['Location'].count()
temp.value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, explode=None,startangle=15)
del temp


# #### We have a date column that is in string format we can get better results if we change it to datetime format. Also than we can extract Year and Month from date and see the distribution according to them.

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].apply(lambda x : x.year)

month_mapper = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun'
               ,7:'Jul', 8:'Aug', 9:'Sep' ,10:'Oct', 11:'Nov', 12:'Dec'}
df['Month'] = df['Date'].apply(lambda x : x.month).map(month_mapper)

del month_mapper


# In[ ]:


df.head()


# #### Let's see number of crossings by Mesures.

# In[ ]:


temp = pd.DataFrame(df.groupby(by='Measure')['Value'].sum().sort_values(ascending=False)).reset_index()
fig = px.bar(temp, x='Measure', y='Value', height=400)
fig.show()
del temp


# #### Most crossings are done by personal vehicle pessengers and personal vehicles. Let's see if it is likely distributed in both the borders of not.

# In[ ]:


temp = df.groupby(by=['Border','Measure'])['Value'].sum().reset_index()
temp.fillna(0,inplace=True)
temp.sort_values(by='Value', inplace=True)
fig = px.bar(temp, x='Measure', y='Value', color='Border', barmode='group')
fig.show()
del temp


# #### measures are likely distributed in both borders. But, One has higher number of crossings than the other. Let's see that.

# #### Below graph represents total number of crossings from borders.

# In[ ]:


temp = df.groupby(by='Border')['Value'].sum()
fig = go.Figure(data=[go.Pie(labels = temp.index, values=temp.values)])
fig.update_traces(textfont_size=15,  marker=dict(line=dict(color='#000000', width=2)))
fig.show()
del temp


# In[ ]:


plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Year', y='Value', hue='Measure',legend='full')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Measure Values Through Years')


# #### Above chart shows number of crossings throughout years.Crossings have been decreasing since year 2000. There is a slight increment in pedestrians crossing over past few yers.

# In[ ]:


plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Month', y='Value',legend='full', hue='Measure')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Value by month')


# ### Above graph shows number of crossings by month. July and Aug have highest crossings where Feb has the least number of crossings.

# In[ ]:


temp = pd.DataFrame(df.groupby(by='Port Name')['Value'].sum().sort_values(ascending=False)).reset_index()
px.bar(temp, x='Port Name', y='Value')
del temp


# #### Above graph represents ports and their number of crossings

# ### We can group measures by their size.

# #### Below we have two plots, both showing bar chart showing sum of values of different size of measures by different states.

# In[ ]:


measure_size = {'Trucks' : 'Mid_Size', 'Rail Containers Full' : 'Mid_Size', 'Trains' : 'Big_Size',
       'Personal Vehicle Passengers':'Small_Size', 'Bus Passengers':'Small_Size',
       'Truck Containers Empty':'Mid_Size', 'Rail Containers Empty':'Mid_Size',
       'Personal Vehicles' : 'Small_Size', 'Buses' : 'Mid_Size', 'Truck Containers Full' : 'Mid_Size',
       'Pedestrians':'Small_Size', 'Train Passengers':'Small_Size'}

df['Size'] = df['Measure'].map(measure_size)


# In[ ]:


temp = df.groupby(by=['Size','State'])['Value'].sum()
temp.fillna(0,inplace=True)
temp = temp.reset_index()
px.bar(temp, x='State', y='Value', facet_col='Size')


# In[ ]:


temp = df.groupby(by=['Size','State'])['Value'].sum().unstack()
temp.fillna(0,inplace=True)

plt.figure(figsize=(15,4))

plt.subplot(131)
temp.iloc[0].sort_values().plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Big_Size')

plt.subplot(132)
temp.iloc[1].sort_values().plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Mid_Size')

plt.subplot(133)
temp.iloc[2].sort_values().plot(kind='bar')
plt.xticks(rotation=90)
plt.title('Small_Size')

del temp


# ## Insights :
# - Minnesota has most number of big_size crossings but has averege on the other two categories.
# - Arizona has good number of small size crossings but average on the other two categories.
# - Ohio, Alaska and Montana has least amount of crossings in all the categories.
# - Michigan has 2nd heighest BIg and Mid size crossings but comparitively less small size crossings.
# - Texas has most mid_size and small size crossings and also, 3rd largest big_size crossings.
# - New York also has good number of crossings in all the three states.

# ### Let's see if crossings of different sizes are seasonal or not.

# In[ ]:


plt.figure(figsize=(15,6))
g = sns.FacetGrid(data=df, col='Size', sharey=False, height=5, aspect=1)
g.map(sns.lineplot, 'Month', 'Value')


# ## Insights :
# - Mid_Size crossings are least in dec, jan and jul and most in oct, mar and aug.
# - Big_Size crossings are least in feb and most in oct, mar and aug.
# - Small_Size crossings are least in jan and feb and most in aug and july.
# - Crossing rate per month is negetively correlated with size of crossing.

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

people = df[df['Measure'].isin(['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers'])]

people_crossing_series = people[['Date','Value']].groupby('Date').sum()

pcsm = people_crossing_series.loc['2011':]

# Multiplicative Decomposition 
res_mul = seasonal_decompose(pcsm, model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
res_add = seasonal_decompose(pcsm, model='additive', extrapolate_trend='freq')

# extrapolate_trend='freq' gets rid of NaN values
# Plot
fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(15,8))

res_mul.observed.plot(ax=axes[0,0], legend=False)
axes[0,0].set_ylabel('Observed')

res_mul.trend.plot(ax=axes[1,0], legend=False)
axes[1,0].set_ylabel('Trend')

res_mul.seasonal.plot(ax=axes[2,0], legend=False)
axes[2,0].set_ylabel('Seasonal')

res_mul.resid.plot(ax=axes[3,0], legend=False)
axes[3,0].set_ylabel('Residual')

res_add.observed.plot(ax=axes[0,1], legend=False)
axes[0,1].set_ylabel('Observed')

res_add.trend.plot(ax=axes[1,1], legend=False)
axes[1,1].set_ylabel('Trend')

res_add.seasonal.plot(ax=axes[2,1], legend=False)
axes[2,1].set_ylabel('Seasonal')

res_add.resid.plot(ax=axes[3,1], legend=False)
axes[3,1].set_ylabel('Residual')

axes[0,0].set_title('Multiplicative')
axes[0,1].set_title('Additive')
    
plt.tight_layout()
plt.show()


# In[ ]:


des = res_mul.trend * res_mul.resid
des.plot(figsize = (15,10))

plt.show()


# In[ ]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(des.Value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[ ]:


from statsmodels.tsa.stattools import adfuller


result_diff = adfuller(des.diff().Value.dropna())
print('ADF Statistic: %f' % result_diff[0])
print('p-value: %f' % result_diff[1])


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(3, 2, figsize=(16,10))

axes[0, 0].plot(des.Value)
axes[0, 0].set_title('Original Series')
plot_pacf(des, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(des.Value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_pacf(des.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(des.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_pacf(des.diff().diff().dropna(), ax=axes[2, 1])

plt.tight_layout()
plt.show()


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA

# ARIMA(p,d,q) Model 
model_1 = ARIMA(des, order=(1,0,1))
model_1_fit = model_1.fit(disp=0)

model_2 = ARIMA(des, order=(1,1,2))
model_2_fit = model_2.fit(disp=0)

model_3 = ARIMA(des, order=(1,0,2))
model_3_fit = model_3.fit(disp=0)


# In[ ]:


des.head(3)
data = des.iloc[1:,]
data.head(98)


# In[ ]:


model_1_fit.plot_predict()
plt.title((sum((model_1_fit.fittedvalues -  des['Value']) ** 2)))
plt.show()

model_2_fit.plot_predict()
plt.title((sum((model_2_fit.fittedvalues -  data['Value']) ** 2)))
plt.show()

model_3_fit.plot_predict()
plt.title((sum((model_3_fit.fittedvalues -  des['Value']) ** 2)))
plt.show()


# In[ ]:


date_start = people_crossing_series.tail(1).index[0]
date_end = '2020-12-01'
date_rng = pd.date_range(start=date_start, end=date_end, freq='MS', closed = 'right') # range for forecasting
n_forecast = len(date_rng) # number of steps to forecast

seasonal = res_mul.seasonal.loc['2018-01-01':'2018-12-01'].values # seasonal component, we take the 2018 ones, but they are all the same.
tms = pd.Series(np.tile(seasonal.flatten(),11), index = pd.date_range(start='2019-01-01', end = '2029-12-01', freq='MS'))  # This is just a very long series with the seasonality.

def make_seasonal(ser) :
    seasonal_series = ser * tms # Include the seasonality
    seasonal_series = seasonal_series[~seasonal_series.isnull()] # trim extra values
    return seasonal_series
    
# Forecast

model = ARIMA(des, order=(1,1,2))
model_fit = model.fit(disp=0)

fc1, se1, conf1 = model_fit.forecast(n_forecast, alpha = 0.0455)  # 2 sigma Confidence Level (95,55% conf)
fc2, se2, conf2 = model_fit.forecast(n_forecast, alpha = 0.3173)  # 1 sigma Confidence Level (68,27% conf)

# Make as pandas series 
fc1_series = pd.Series(fc1, index = date_rng)
lower_series1 = pd.Series(conf1[:, 0], index = date_rng)
upper_series1 = pd.Series(conf1[:, 1], index = date_rng)

# Include seasonality
fc1_series, lower_series1, upper_series1 = [make_seasonal(fc1_series), make_seasonal(lower_series1), make_seasonal(upper_series1)]

plt.figure(figsize=(12,5), dpi=100)

#plt.plot(des, label='actual')
#plt.plot(people_crossing_series, label='actual')
plt.plot(des * res_mul.seasonal, label='data')
plt.plot(fc1_series , label='forecast')

# Confidence level intervals
plt.fill_between(lower_series1.index,lower_series1, upper_series1, 
                 color='k', alpha=.15, label='2$\sigma$ Confidence level (95%)')
plt.title('Forecast 2019/20')
plt.legend(loc='upper left', fontsize=8)
#plt.ylim(10000000, 30000000)
plt.xlim('2016', '2021')
plt.show()

