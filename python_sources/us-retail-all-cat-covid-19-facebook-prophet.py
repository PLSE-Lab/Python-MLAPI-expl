#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_excel("../input/usa-monthly-retail-trade/mrtssales92-present.xls",sheet_name = None, header = 0, encoding = 'utf-8',errors='strict')


# ## Data Cleaning

# removing Unnamed columns created due to encoding error by pandas in the dataframe

# In[ ]:


for key in df:
    df[key] = df[key].loc[:, ~df[key].columns.str.contains('^Unnamed')]


# ### Extract main categories

# Slicing Database into two dataframes of Not Adjusted and Adjusted Main Categories (NAICS code : 4xx, 7xx)

# In[ ]:


df_cat_ad = {}
df_cat_nad = {}
for key in df:
    df_cat = df.copy()
    df_cat[key]['NAICS  Code'] = df_cat[key]['NAICS  Code'].astype('str')
    df_cat[key]['NAICS Code Cat'] = df_cat[key]['NAICS  Code'].str.extract('^(\d{3})$')
    index = pd.Index(df_cat[key]['Kind of Business'])
    x = index.get_loc('ADJUSTED(2)')
    df_cat_ad[key] = df_cat[key].iloc[x:]
    df_cat_ad[key] = df_cat_ad[key].dropna(subset = ['NAICS Code Cat'])
    df_cat_ad[key] = df_cat_ad[key].reset_index(drop = True)
    df_cat_ad[key] = df_cat_ad[key].drop(['NAICS Code Cat'], axis=1)
    df_cat_nad[key] = df_cat[key].iloc[0:x]
    df_cat_nad[key] = df_cat_nad[key].dropna(subset = ['NAICS Code Cat'])
    df_cat_nad[key] = df_cat_nad[key].reset_index(drop = True)
    df_cat_nad[key] = df_cat_nad[key].drop(['NAICS Code Cat'], axis=1)


# checking weather data contains (NA):Not Available or (S):Suppressed values

# In[ ]:


def string_finder(row, words):
    if any(word in field for field in row for word in words):
        return 1
    return 0


# In[ ]:


match = ['(NA)', '(S)']
for key in df:
    df_cat_nad[key]['isContained'] = df_cat_nad[key].astype(str).apply(string_finder, words=match, axis=1)
    df_cat_ad[key]['isContained'] = df_cat_ad[key].astype(str).apply(string_finder, words=match, axis=1)
    print(key)
    print('Not Adjusted data contains', df_cat_nad[key]['isContained'].sum(), '(NA):Not Available or (S):Suppressed values')
    print('Adjusted data contains', df_cat_ad[key]['isContained'].sum(), '(NA):Not Available or (S):Suppressed values')
#     df_cat_nad[key] = df_cat_nad[key].drop(['isContained'], axis=1)
#     df_cat_ad[key] = df_cat_ad[key].drop(['isContained'], axis=1)


# Since data contains many different kinds of dtypes, cleaning data of different dtypes

# In[ ]:


for key in df:
    df_cat_nad[key].iloc[ : ,2: ] = df_cat_nad[key].iloc[ : ,2: ].astype(float)
    df_cat_ad[key].iloc[ : ,2: ] = df_cat_ad[key].iloc[ : ,2: ].astype(float)


# In[ ]:


for key in df:
    df_cat_nad[key] = df_cat_nad[key].drop(['isContained'], axis=1)
    df_cat_ad[key] = df_cat_ad[key].drop(['isContained'], axis=1)


# Since our data is Categorical data we need to modify it
# * seperating Total from data 
# * Encoding rows into columns
# * converting Rows to column and making a single dataframe for all years

# In[ ]:


for key in df:
    df_cat_nad[key]['Kind of Business'] = df_cat_nad[key]['NAICS  Code'] + ': ' + df_cat_nad[key]['Kind of Business']
    df_cat_ad[key]['Kind of Business'] = df_cat_ad[key]['NAICS  Code'] + ': ' + df_cat_ad[key]['Kind of Business']


# In[ ]:


df_cat_nad_m = {}
df_cat_ad_m = {}
for key in df:
    df_cat_nad_m[key] = df_cat_nad[key].iloc[ : ,1:-1]
    df_cat_nad_m[key] = df_cat_nad_m[key].transpose(copy = True)
    header = df_cat_nad_m[key].iloc[0]
    df_cat_nad_m[key] = df_cat_nad_m[key][1:]
    df_cat_nad_m[key] = df_cat_nad_m[key].rename(columns = header)
    df_cat_nad_m[key] = df_cat_nad_m[key].astype(float)
    df_cat_nad_m[key] = df_cat_nad_m[key].reset_index().rename(columns={'index': 'Months'})
    df_cat_ad_m[key] = df_cat_ad[key].iloc[ : ,1:-1]
    df_cat_ad_m[key] = df_cat_ad_m[key].transpose(copy = True)
    header = df_cat_ad_m[key].iloc[0]
    df_cat_ad_m[key] = df_cat_ad_m[key][1:]
    df_cat_ad_m[key] = df_cat_ad_m[key].rename(columns = header)
    df_cat_ad_m[key] = df_cat_ad_m[key].astype(float)
    df_cat_ad_m[key] = df_cat_ad_m[key].reset_index().rename(columns={'index': 'Months'})


# In[ ]:


df_NA_months = pd.concat(df_cat_nad_m,ignore_index=True)
df_A_months = pd.concat(df_cat_ad_m,ignore_index=True)


# ### Time Series Indexing
# Converting Months to Datetime64[ns] for time series analysis

# In[ ]:


df_A_months


# In[ ]:


df_NA_months['Months'] = pd.to_datetime(df_NA_months['Months'], infer_datetime_format=True)
df_A_months['Months'] = pd.to_datetime(df_A_months['Months'], infer_datetime_format=True)


# In[ ]:


df_NA_months = df_NA_months.set_index('Months')
df_A_months = df_A_months.set_index('Months')


# ## Forcasting using Facebook Prophet

# In[ ]:


from fbprophet import Prophet


# In[ ]:


df_A_months


# In[ ]:


def change_name(timeseries):
    ts = {}
    ts['ds'] = timeseries.index
    ts['y'] = timeseries.values
    ts_df = pd.DataFrame(data = ts)
    return ts_df


# In[ ]:


for i in range(0,12):
    ts = change_name(df_A_months.iloc[:,i])
    model = Prophet(yearly_seasonality=False,daily_seasonality=False)
    model.fit(ts)
    forecast = model.make_future_dataframe(periods=300, include_history=False)
    forecast = model.predict(forecast)
    figure = model.plot(forecast, xlabel = 'Date', ylabel = df_A_months.columns.values[i])
    figure2 = model.plot_components(forecast)


# In[ ]:





# In[ ]:




