#!/usr/bin/env python
# coding: utf-8

# # How to change plot data using Dropdowns!
# 
# ### #Plotly #Python #Drowdowns #Stocks #Tutorial
# 
# **Problem:** For a given dataset with many attributes, I was plotting the same style of plot but with different data for the y-axis. I was trying to find a way to save space in the notebook and somehow have the ability to swap the data in the same plot quickly without having to re-running the cell.
# 
# **Solution:** I found that in [Plotly](https://plot.ly/python/), you can configure a [dropdown of buttons](https://plot.ly/python/dropdowns/) that can force the plot to update its style or data. The dropdown menu code is straightforward and the user interface is simple enough to include in professional notebooks.
# 
# I have been using [Plotly](https://plot.ly/python/) for some time now. The previous versions of their library weren't as user friendly, but the 4.x version of the library is really good and easy to use. However, there are still a lot of things about their documentation that lack clarity and demonstration on how the features work. Hence this notebook is designed as a tutorial or a referrence cheat sheet for whenever in the future I want to add a plot with a dropdown menu in my notebooks.

# ### Step 1: Get some data to plot
# For this tutorial, I will use stocks data from the Python library called [yfinance](https://pypi.org/project/yfinance/) because it was simple to use, and it contains real world data to work with. This library depends on the stock markers from [Yahoo Finance](https://finance.yahoo.com/).
# 
# **Note: To install the yfinance library, you will have to enable Internet in the Kaggle notebook settings**

# In[ ]:


get_ipython().system('pip install yfinance')


# In[ ]:


import yfinance as yf

import pandas as pd
import plotly.graph_objects as go


# In[ ]:


# Example of how yfinance works

# Request stocks data for Microsoft (MSFT)
MSFT = yf.Ticker("MSFT")
df_MSFT = MSFT.history(period="max")

# Display the dataset
df_MSFT


# In[ ]:


# Request stocks data for Apple (AAPL)
AAPL = yf.Ticker("AAPL")
df_AAPL = AAPL.history(period="max")

# Request stocks data for Amazon (AMZN)
AMZN = yf.Ticker("AMZN")
df_AMZN = AMZN.history(period="max")

# Request stocks data for Google (GOOGL)
GOOGL = yf.Ticker("GOOGL")
df_GOOGL = GOOGL.history(period="max")


# In[ ]:


df_stocks = pd.DataFrame({
    'MSFT': df_MSFT['High'],
    'AAPL': df_AAPL['High'],
    'AMZN': df_AMZN['High'],
    'GOOGL': df_GOOGL['High'],
})

# manually create a dataset of stocks at their daily High
df_stocks


# # Step 2: Plot the data

# In[ ]:


# How to change plot data using dropdowns
#
# This example shows how to manually add traces
# to the plot and configure the dropdown to only
# show the specific traces you allow.

fig = go.Figure()

for column in df_stocks.columns.to_list():
    fig.add_trace(
        go.Scatter(
            x = df_stocks.index,
            y = df_stocks[column],
            name = column
        )
    )
    
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=list(
            [dict(label = 'All',
                  method = 'update',
                  args = [{'visible': [True, True, True, True]},
                          {'title': 'All',
                           'showlegend':True}]),
             dict(label = 'MSFT',
                  method = 'update',
                  args = [{'visible': [True, False, False, False]}, # the index of True aligns with the indices of plot traces
                          {'title': 'MSFT',
                           'showlegend':True}]),
             dict(label = 'AAPL',
                  method = 'update',
                  args = [{'visible': [False, True, False, False]},
                          {'title': 'AAPL',
                           'showlegend':True}]),
             dict(label = 'AMZN',
                  method = 'update',
                  args = [{'visible': [False, False, True, False]},
                          {'title': 'AMZN',
                           'showlegend':True}]),
             dict(label = 'GOOGL',
                  method = 'update',
                  args = [{'visible': [False, False, False, True]},
                          {'title': 'GOOGL',
                           'showlegend':True}]),
            ])
        )
    ])

fig.show()


# In[ ]:


# Configure functions to automate the data collection process

# getStocks requires three variables:
# - stocks is a list of strings which are the code for the stock
# - history is timeframe of how much of the stock data is desired
# - attribute is the attribute of the stock 
def getStocks(stocks, history, attribute):
    return pd.DataFrame({stock:yf.Ticker(stock).history(period=history)[attribute] for stock in stocks})

# multi_plot requires two variables:
# - df is a dataframe with stocks as columns and rows as date of the stock price
# - addAll is to have a dropdown button to display all stocks at once
def multi_plot(df, addAll = True):
    fig = go.Figure()

    for column in df.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = df.index,
                y = df[column],
                name = column
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ])
    
    fig.show()


# In[ ]:


# How to change plot data using dropdowns
#
# This example shows how to automatically add traces
# to the plot and automatically configure the dropdown
# to include all columns of the dataframe

# The beauty of this example is that all you need to do
# is to change the list of stock codes below and the plot
# will adjust accordingly

stocks = ['MSFT', # Microsoft
          'AAPL', # Apple
          'AMZN', # Amazon
          'GOOGL' # Google Alphabet
         ]

df_stocks = getStocks(stocks, 'max', 'High')

multi_plot(df_stocks)


# In[ ]:


# How to change plot data using dropdowns
#
# This example shows how to automatically add traces
# to the plot and automatically configure the dropdown
# to include all columns of the dataframe

# The beauty of this example is that all you need to do
# is to change the list of stock codes below and the plot
# will adjust accordingly

stocks = ['HMC', # Honda
          'TM', # Toyota Motor Corporation
          'GM', # General Motors
          'F', # FORD
          'TSLA' # Tesla
         ]

df_stocks = getStocks(stocks, 'max', 'High')

multi_plot(df_stocks)


# If you liked my work, please remember to upvote the notebook!
# 
# If you have any questions, feel free to comment down below or [message me on LinkedIn](https://www.linkedin.com/in/jaydeep-mistry/).
