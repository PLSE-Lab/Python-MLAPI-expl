#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as exp
import plotly.graph_objects as go

import pandas as pd
import numpy as np


# ##  Goal

# Given the unit sales of various products from Walmart, forecast daily sales for 28 days into the future(from the last day for which the data is available) .
# 
# The data is from three states (California, Texas, and Wisconsin) and spans over a period of 5 years from 2011 - 2016. For each item being sold, we have information on the department it's from, its category, and the store details selling the item. There are 3049 items belonging to 3 categories and 7 departments, being sold in 10 stores in 3 states.
# Apart from the above, we also have some other variables such as prices of the product for the week, and special events.
# 

# ## Data Organization
# 
# - **Calendar.csv**: Contains the dates on which the products are sold.
#     - date, wm_yr_wk, weekday, wday, month, year, d
#     - event_name_1, event_type_1, event_name_2, event_type_2: Certain events which may affect the sales
#     - snap_CA, snap_TX, snap_WI: 
#     
#     
# 
# - **sales_train_validation.csv**: Contains the unit sales data for each product for 1913 days.
#     - ***id, item_id***
#     - ***cat_id:***  3, hobbies, household, food
#     - ***dept_id***:  7 departments,  2 for hobbies, 2 for household, 3 for foods
#     - ***state_id***:  3 states, CA, WI, TX
#     - ***store_id***:  10 stores, CA: 4, TX: 3, WI: 3
#     - ***d_1, d_2, ... d_1912, d_1913***:  Daily unit sales data for each item belonging to a particular category
#       (cat_id), from a department (dept_id), sold at some store with id as store_id, in a particular state identified by state_id.
# 
# 
# - **sell_prices.csv**: Contanins the price of each product that was sold in different stores and the date it was sold.
# 

# In[ ]:


calendar = "/kaggle/input/m5-forecasting-accuracy/calendar.csv"
sales_train_validation = "/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv"
sell_prices = "/kaggle/input/m5-forecasting-accuracy/sell_prices.csv"


calendar_ = pd.read_csv(calendar, delimiter=",")
sales_train_validation_ = pd.read_csv(sales_train_validation, delimiter=",")
sell_prices_ = pd.read_csv(sell_prices, delimiter=",")


# In[ ]:


print(calendar_.shape)
calendar_.head()


# In[ ]:


print(sales_train_validation_.shape)
sales_train_validation_.head()


# In[ ]:


print(sell_prices_.shape)
sell_prices_.head()


# ## Time Series Forecasting:
# 
# - https://otexts.com/fpp2/intro.html
# - https://robjhyndman.com/papers/forecastingoverview.pdf

# ## Category wise sales

# In[ ]:


groups = sales_train_validation_.groupby(['cat_id'])
counts_dict = {}
for name, group in groups:
    counts_dict[name] = len(group)
df = pd.DataFrame(counts_dict.items(), columns=['category', 'value'])
fig = exp.pie(df, values='value', names='category', title='Category wise sales of items')
fig.show()


# In[ ]:





# ## State wise sales

# In[ ]:


groups = sales_train_validation_.groupby(['state_id'])
counts_dict = {}
for name, group in groups:
    counts_dict[name] = len(group)
df = pd.DataFrame(counts_dict.items(), columns=['category', 'value'])
fig = exp.pie(df, values='value', names='category', title='State wise sales of items')
fig.show()


#  

# ## Plotting the time series data

# In[ ]:


date_dict = pd.Series(calendar_.date.values,index=calendar_.d).to_dict()
dates = list(date_dict.values())[0:1913] # we have sales data for 1913 days 

def plot_time_series(row):
    
    daily_sales = row.iloc[6:].values
    
    
    df = pd.DataFrame(
    {'daily sales': daily_sales,
     'date': dates,
    })

    fig = exp.line(df,x="date",y="daily sales")
    
    fig.update_layout(
    title={
        'text': "Daily Unit Sales of a particular item from 2011 - 2016",
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    
    fig.show()

plot_time_series(sales_train_validation_.iloc[1,:])


# In[ ]:


# plot a random sample of 10 points

df_sample = sales_train_validation_.sample(n=10, replace=False)
df_sample_ = pd.DataFrame(columns = ['item id','daily sales','date'])

colors = (exp.colors.sequential.Plasma)

def plot_sample(df_sample):
    
    fig = go.Figure()
    count = 0
    
    for index,row in df_sample.iterrows():
        
        
        item_id = row[1]
        daily_sales = row.iloc[6:].values
           
        fig.add_trace(go.Scatter(x=dates, y=daily_sales,
                    mode='lines+markers',
                    name=item_id, marker = dict(color=colors[count])))
        
        """ uncomment to plot a sample of more than 10 points
        fig.add_trace(go.Scatter(x=dates, y=daily_sales,
               mode='lines+markers',
               name=item_id))"""
        
        count+=1
    
    fig.update_layout(
    autosize=False,
    width=1200,
    height=500, title={
        'text': "Daily Unit Sales of items from 2011 - 2016",
        'y':1,
        'x':0.3,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    fig.show()

plot_sample(df_sample)        
        
        


# ### Key Observations
# 
# - Looking at the plots, we observe that most of the values are zeroes. It's noisy. There are too many fluctuations in this data.
#   
#   Can we say that the demand for these products is **intermittent**? How can we determine this statistically?
#   Intermittent time series have a large number of values that are zero. When an item has several periods of zero demand, the demand is said to be intermittent. In cases other than zero, the demand is erratic. Read more at:
#       - https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf
#       - https://www.researchgate.net/publication/237019869_Intermittent_demand_forecasts_with_neural_networks
#       - https://robjhyndman.com/papers/foresight.pdf

# #### Data Smoothing using moving averages.
# 
# Since our time series data is noisy, we can smooth it using moving averages.
# Moving average removes the random fluctuations/ variation in data by taking the average of a fixed window of observations, over the entire time series. Here we consider a window size of 30.
# 

# In[ ]:


### Moving Average

def plot_moving_avg_series(row):
    
    item_id = row.iloc[1]
    
    daily_sales_ = row.iloc[6:].values
    daily_sales = list(map(int, daily_sales_))
    
    df = pd.DataFrame(
    {'daily sales': daily_sales,
     'date': dates,
    })
    
    rolling = df.rolling(window=30)
    moving_avg = rolling.mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=dates, y=daily_sales,mode='lines+markers',name=item_id))
    
    fig.add_trace(go.Scatter(x=dates, y=moving_avg['daily sales'],mode='lines+markers',name=item_id))

    
    fig.update_layout(
    autosize=False,
    width=1500,
    height=500,
    title={
        'text': "Daily Unit Sales of a particular item from 2011 - 2016",
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    
    fig.show()
    
plot_moving_avg_series(sales_train_validation_.iloc[1,:])


# In[ ]:


def plot_moving_avg_sample(df_sample):
    
    fig = go.Figure()
    
    for index,row in df_sample.iterrows():
        
        
        item_id = row[1]
        
        daily_sales_ = row.iloc[6:].values
        daily_sales = list(map(int, daily_sales_))
    
        df = pd.DataFrame(
            {'daily sales': daily_sales,
             'date': dates,
            })
    
        rolling = df.rolling(window=30)
        moving_avg = rolling.mean()

        
        fig.add_trace(go.Scatter(x=dates, y=moving_avg['daily sales'],
               mode='lines+markers',
               name=item_id))
        
    fig.update_layout(
    autosize=False,
    width=1200,
    height=500,
    title={
        'text': "Daily Unit Sales of a sample from 2011 - 2016",
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    
    fig.show()

plot_moving_avg_sample(df_sample)


# 

# ### Decomposition of a time series
# 
# A time series can be defined by decomposing it into four main elements:
# 
# - Trend: Long-term movements- increasing/decreing trend.
# - Seasonal Effect: Calendar-related cyclical fluctuations.
# - Cycles: Business cycle-related fluctuations.
# - Residuals: Random or systematic fluctuations
# 
# ### Questions
# 
# - Seasonality: Any repititive pattern observed in the sales of various items over the years?
# - Fluctuaton in the sales during an event/ promotions/ specific time period?

# ### References
# - https://mk0mcompetitiont8ake.kinstacdn.com/wp-content/uploads/2020/02/M5-Competitors-Guide_Final-1.pdf
# - https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda
# - https://www.kaggle.com/robikscube/m5-forecasting-starter-data-exploration 

# In[ ]:




