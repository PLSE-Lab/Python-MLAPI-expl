#!/usr/bin/env python
# coding: utf-8

# ![](https://cdn.pixabay.com/photo/2020/01/22/10/59/stock-exchange-4785086_960_720.jpg)
# # Introduction 
# 
# ## Context
# Exchange rates are a common indicator for both travelers to investors. While these exchange quotes can be readily obtained anywhere, it is much easier for people to gain key insights through simple chart visualisation. A time series will prove useful to help understand past behaviors and also predict future values. 
# 
# ## Objective
# - Create a time series to compare different exchange rates using [Plotly Express](https://plot.ly/python/plotly-express/) 

# ## Importing required packages 

# In[ ]:


import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go


# ## Importing csv dataset into Pandas dataframe and have an overview
# - Some rows contain ND (No Data) and we will need to get rid of them before our analysis.
# - I will also replace spaces from column names into underscores "_"

# In[ ]:


data = pd.read_csv('../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')
data.head()
data= data.replace('ND', np.nan)
data = data.dropna()
data


# ## Creating a list of countries name that we need 

# In[ ]:


country_lst = list(data.columns[2:])
type(country_lst[1])


# In[ ]:


updatemenu_lst = []
for a,b in enumerate(country_lst):
    visible = [False] * len(country_lst)
    visible[0] = True
visible


# In[ ]:


# Initialise figure 
fig = go.Figure()
fig.update_yaxes(automargin=True)
# Add Traces

for k,v in enumerate(country_lst):
    colour_lst = ['#91930b', '#6cdc93', '#935049', '#acbc09', '#0b92d3', '#dc8845', '#a60c7c', '#4a31f7', '#d8191c', '#e86f71','#efd4f3','#2e0e88','#7d4c26','#0bc039','#fa378c','#54f1e5','#7a0a8b','#43142d','#beaef4','#04b919','#91dde5','#2a850d']
    fig.add_trace(
    go.Scatter(x= data['Time Serie'],
                    y= data[country_lst[k]],
                    name= country_lst[k],
                    line=dict(color=colour_lst[k])))
    
    
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label= 'AUS/US',
                     method="update",
                     args=[{"visible": [True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[0]}]),
                dict(label='EUR/US',
                     method="update",
                     args=[{"visible": [False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[1]}]),
                dict(label='NZ/US',
                     method="update",
                     args=[{"visible": [False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[2]}]),
                dict(label='UK/US',
                     method="update",
                     args=[{"visible": [False, False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[3]}]),
                dict(label='BRA/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[4]}]),
                dict(label='CAN/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[5]}]),
                dict(label='CHINA/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[6]}]),
                dict(label='HK/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[7]}]),
                dict(label='INDIA/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[8]}]),
                dict(label='KOR/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[9]}]),
                dict(label='MEX/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[10]}]),
                dict(label='SOUTHAFRICA/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[11]}]),
                dict(label='SG/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[12]}]),
                dict(label='DEN/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False]},
                           {"title": country_lst[13]}]),
                dict(label='JAP/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False]},
                           {"title": country_lst[14]}]),
                dict(label='MY/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False]},
                           {"title": country_lst[15]}]),
                dict(label='NOR/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False]},
                           {"title": country_lst[16]}]),
                dict(label='SWE/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False]},
                           {"title": country_lst[17]}]),
                dict(label='SRI/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False]},
                           {"title": country_lst[18]}]),
                dict(label='CHE/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False]},
                           {"title": country_lst[19]}]),
                dict(label='TAI/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False]},
                           {"title": country_lst[20]}]),
                dict(label='THAI/US',
                     method="update",
                     args=[{"visible": [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True]},
                           {"title": country_lst[21]}]),
                dict(label='Asia & Oceania',
                     method="update",
                     args=[{"visible": [True, False,True,False,False,False,True,True,True,True,False,False,True,False,True,True,False,False,True,False,True,True]},
                           {"title": 'Asia & Oceania'}]),
                dict(label='Europe',
                     method="update",
                     args=[{"visible": [False, True,False,True,False,False,False,False,False,False,False,False,False,True,False,False,True,True,False,True,False,False]},
                           {"title": 'Europe'}]),
            
            ]),
        )
    ])


# #### Looking at the figure we realise that it gets abit sqeezy when all the variables are being plot together. This is why I created the dropdown menu for each currency. You can zoom in around to look at the dates closely. I have also sort them out according to continent (you can see from menu). 
# - I have tried using functions and loops to interate the variables for the layout but the package created 22 dropdown menu, all overlapping one another. Hence, I have decided to hardcode the layout using copy and paste. 
