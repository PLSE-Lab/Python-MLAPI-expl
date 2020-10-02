#!/usr/bin/env python
# coding: utf-8

# # City Pairs: Domestic Traffic
# 
# AndrewJ, 2020-04-05

# ## Description
# 
# Visualisation sandbox on some random data sets using Python3 and Vega-Lite (via Altair). In this case, it's Australian [monthly airport domestic traffic data](https://data.gov.au/dataset/domestic-airlines-top-routes-and-totals) via data.gov.au.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import datetime as dt
import altair as alt


# ## Read and process the data

# In[ ]:


def read_traffic():
    return pd.read_csv("../input/au-dom-traffic/audomcitypairs-201912.csv")


# In[ ]:


def transform_traffic(df):
    df1 = df.assign(
        Journey = df.City1 + "-" + df.City2, 
        Month = dt.datetime(1899, 12, 30) + df['Month'].map(dt.timedelta))
    return df1


# ## Run

# In[ ]:


dom = transform_traffic(read_traffic())


# In[ ]:


dom.head()


# In[ ]:


dom.dtypes


# ## Visualise

# Top sectors by total passengers

# In[ ]:


trips = dom['Passenger_Trips']     .groupby(dom['Journey'])     .mean()     .sort_values(ascending = False)     .reset_index()

trips.head(5)


# In[ ]:


bars = alt.Chart(trips.head(15))     .mark_bar(size = 15)     .encode(
        x = alt.X(
            'Passenger_Trips:Q',
            scale = alt.Scale(domain = [0, 550000])),
        y = alt.Y(
            'Journey:O', 
            sort = '-x'))

labels = bars     .mark_text(dx = 25)     .encode(
        text = alt.Text(
            'Passenger_Trips:Q', 
            format = ".0d"))

(bars+labels).properties(
        width = 500, 
        height = 350)


# Time series of monthly passenger numbers.

# In[ ]:


trips_months = dom     .groupby(['Month'])[['Passenger_Trips']]     .sum()     .reset_index()


# In[ ]:


alt.Chart(trips_months)     .mark_line(color = 'green')     .encode(
        x = 'Month:T',
        y = 'Passenger_Trips:Q') \
    .properties(
        width = 400,
        height = 250)


# Plot the moving average centered over a 12-month window.

# In[ ]:


df = trips_months     .set_index('Month').rolling(window = 12, center = True)     .mean()     .reset_index()


# In[ ]:


alt.Chart(df)     .mark_line(color = 'green')     .encode(
        x = 'Month:T',
        y = 'Passenger_Trips:Q') \
    .properties(
        width = 400,
        height = 250)


# In[ ]:




