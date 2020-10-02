#!/usr/bin/env python
# coding: utf-8

# Libraries Used:

# In[ ]:



import os
print(os.listdir("../input"))

# Exploratory Data Analysis Tools
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from string import ascii_letters

# Modeling Tools
import fbprophet
import warnings
import itertools
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt


# Data Visualization Tools
import seaborn as sns
from matplotlib import pyplot
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls


# **Read and Print Raw Data**

# In[ ]:


# Opening source file
org_data = pd.read_csv('../input/Monthly_data_cmo.csv')
apmc = org_data # creating a separate dataset to operate on!!
apmc.tail()


# ** About the Raw Data:**

# In[ ]:


apmc.describe()


# From the description above can see that the dataset contains 3 years of data from 2014, 2015 and 2016, with 62,429 data points. However, lets check if there is any missing values.

# In[ ]:


# Checking Missing Data points, if any
apmc = apmc.dropna(subset=['Year', 'arrivals_in_qtl', 'min_price', 'max_price', 'modal_price'])
apmc.describe() # cleaning, dropping those rows with no values in any one of the above columns


# Wooah! The dataset seems to be clean in terms of No-Missing values!!

# ## Exploratory Data Analysis (EDA)
# 
# 
# ### <font color='green'>1. Total Consumption (in 3 years) v/s Commodity</font> : 
# #### In order to look which commodity has sold the maximum over the years this plot is generated.

# In[ ]:


df3 = pd.DataFrame(apmc.groupby(['Commodity', 'Year']).agg('sum')).reset_index()
df3.tail()


# In[ ]:


trace1 = go.Bar(
    x= df3.loc[df3['Year'] == 2016].Commodity,
    y= df3.loc[df3['Year'] == 2016].arrivals_in_qtl,
    name='2016',
    marker=dict(
        color='yellow', 
        line=dict(
            color='rgb(8,48,107)',
            width=0.2),
        ),
    opacity=0.6
)

trace2 = go.Bar(
    x= df3.loc[df3['Year'] == 2015].Commodity,
    y= df3.loc[df3['Year'] == 2015].arrivals_in_qtl,
    name='2015',
    marker=dict(
        color='brown', 
        line=dict(
            color='rgb(8,48,107)',
            width=0.2),
        ),
    opacity=0.6
)

trace3 = go.Bar(
    x= df3.loc[df3['Year'] == 2014].Commodity,
    y= df3.loc[df3['Year'] == 2014].arrivals_in_qtl,
    name='2014',
    marker=dict(
        color='red', 
        line=dict(
            color='rgb(8,48,107)',
            width=0.2),
        ),
    opacity=0.6
)

layout = go.Layout(
    title='Commodities Purchased (in Volumes) per year'
)

data = [trace1, trace2, trace3]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_commodity")


# ### <font color='green'>2. Most Popular and Least Popular Commodity</font> : 
# #### In order to look which commodity has sold the maximum and minimum over the years this plot is generated.

# In[ ]:


df1 = pd.DataFrame(apmc.groupby(['Commodity']).sum()).reset_index()
df1 = df1[['Commodity', 'arrivals_in_qtl']]
# df1.tail()

df1_a = df1[df1['arrivals_in_qtl'] > 1000000]
df1_a_sort = df1_a.sort_values('arrivals_in_qtl', ascending=True) # for latest python df.sort has been deprecated and updated to df.sort_values


trace = go.Bar(
    x= df1_a_sort.Commodity,
    y= df1_a_sort.arrivals_in_qtl,
    marker=dict(
        color='orange',
    ),
)

layout = go.Layout(
    title='Most Popular Commodity'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_Commodity")


# From the plot above we can see that the <font color='red'>Onion</font> is by far the most produced or popular Commodity in Maharashtra followed by <font color='red'>Soybean, Potato, Coriander, Cotton, Rice and Tomato</font>
# 
# However, the onion is purchased more than 100 Million Quintals in 2 years span(Oct 2014 - Oct 2016)!!
# 
# Lets now check which commodities fare bad in terms of Quintals purchased. For simplification all the commodities less than 100 Quintals of purchase between (Oct 2014 - Oct 2016) is plotted

# **Selecting the bottom Selling Commodity in term of < 100  Quintals purchased over 2 years span**

# In[ ]:


# selecting the bottom Selling Commodity in term of < 100  Quintals purchased over 2 years span
df1_b = df1[df1['arrivals_in_qtl']<100]
df1_b_sort = df1_b.sort_values('arrivals_in_qtl', ascending=True) # for latest python df.sort has been deprecated and updated to df.sort_values

trace = go.Bar(
    x= df1_b_sort.Commodity,
    y= df1_b_sort.arrivals_in_qtl,
    marker=dict(
        color='green',
    ),
)

layout = go.Layout(
    title='Least Popular Commodity'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_Commodity")


# One interesting thing to be noted here is that all the meat products like <font color='green'> Buffalo, Sheep, Goat, skin & Bones </font> are all under 100 quintals in 2 years. This infers that there is some strict regulation In the state , in terms of purchasing meat products.

# ### <font color='green'>3. Market Share by Disctrict</font> : 
# #### In order to look which district has generated the maximum market share in these two years

# *Feature 1:*

# In[ ]:


#creating a new column Calculating the total market price
apmc['Total_Market_Price'] = apmc['modal_price'] * apmc['arrivals_in_qtl']


# In[ ]:


df2 = pd.DataFrame(apmc.groupby(['district_name', 'Year']).agg('mean')).reset_index()
df2.tail(n=6)

trace14 = go.Bar(
    x= df2.loc[df2['Year'] == 2014].district_name,
    y = df2.loc[df2['Year'].isin([2014])].Total_Market_Price,
    name='2014',
    
    marker=dict(
        color='orange', 
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=1.0
)

trace15 = go.Bar(
    x= df2.loc[df2['Year'] == 2015].district_name,
    y= df2.Total_Market_Price.loc[df2['Year'] == 2015],
    name='2015',
    marker=dict(
        color='purple', 
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.8
)

trace16 = go.Bar(
    x= df2.loc[df2['Year'] == 2016].district_name,
    y= df2.loc[df2['Year'] == 2016].Total_Market_Price,
    name='2016',
    marker=dict(
        color='pink', 
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

layout = go.Layout(
    title='Market share per district per year'
)

data = [trace14, trace15, trace16]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="district_price")


# <font color='red'>Mumbai </font> has the most market share for all the three years, followed by Sangli and Wardha.

# ### <font color='green'>4. Price versus Demand Seasonality</font> : 
# #### To check the effect of seasonality in Modal Price as well as Volume purchased

#  Lets sort the top 10 selling Commodities and check their seasonal variability

# In[ ]:


# selecting the top 10 Selling Commodity in term of NUmber of Quintals purchased over 3 years
print(df1.sort_values('arrivals_in_qtl', ascending=False).head(n=10)) # for latest python df.sort has been deprecated and updated to df.sort_values


# #### Lets plot the seasonal variability of the price and quantity for these commodities

# *Feature 2*

# In[ ]:


df4 = apmc
df4 = df4.loc[df4['Commodity'].isin(['Onion','Soybean', 'Potato', 'Cotton', 'Rice(Paddy-Hus)', 'Tomato', ' Coriander  ', 'Methi (Bhaji)','Pigeon Pea (Tur)', 'Maize'])]
# Getting the Sum and the mean of the values
df4_a = pd.DataFrame(df4.groupby(['Commodity', 'date']).agg('sum')).reset_index()
df4_b = pd.DataFrame(df4.groupby(['Commodity', 'date']).agg('mean')).reset_index()


# In[ ]:


trace21 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Onion'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Onion'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Onion'
)

trace22 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Soybean'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Soybean'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Soybean'
)

trace23 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Cotton'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Cotton'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Cotton'
)

trace24 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Potato'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Potato'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Potato'
)

trace25 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Rice(Paddy-Hus)'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Rice(Paddy-Hus)'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Rice(Paddy-Hus)'
)

trace26 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Tomato'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Tomato'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Tomato'
)

trace27 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Coriander'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Coriander'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Coriander'
)

trace28 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Methi (Bhaji)'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Methi (Bhaji)'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Methi (Bhaji)'
)

trace29 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Pigeon Pea (Tur)'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Pigeon Pea (Tur)'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Pigeon Pea (Tur)'
)

trace20 = go.Scatter(
    x= df4_a.loc[df4_a['Commodity'] == 'Maize'].date,
    y= df4_a.loc[df4_a['Commodity'] == 'Maize'].arrivals_in_qtl,
    mode = 'lines+markers',
    name = 'Maize'
)

trace31 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Onion'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Onion'].modal_price,
    name = 'Onion_Price',
    yaxis='y2',
    opacity=0.5
)

trace32 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Soybean'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Soybean'].modal_price,
    name = 'Soybean_Price',
    yaxis='y2',
    opacity=0.5
)

trace33 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Cotton'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Cotton'].modal_price,
    name = 'Cotton_Price',
    yaxis='y2',
    opacity=0.5
)

trace34 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Potato'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Potato'].modal_price,
    name = 'Potato_Price',
    yaxis='y2',
    opacity=0.5
)

trace35 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Rice(Paddy-Hus)'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Rice(Paddy-Hus)'].modal_price,
    name = 'Rice(Paddy-Hus)_Price',
    yaxis='y2',
    opacity=0.5
)
    
trace36 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Tomato'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Tomato'].modal_price,
    name = 'Tomato_Price',
    yaxis='y2',
    opacity=0.5
)
    
trace37 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Coriander'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Coriander'].modal_price,
    name = 'Coriander_Price',
    yaxis='y2',
    opacity=0.5
)
    
trace38 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Methi (Bhaji)'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Methi (Bhaji)'].modal_price,
    name = 'Methi (Bhaji)_Price',
    yaxis='y2',
    opacity=0.5
)
    
trace39 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Pigeon Pea (Tur)'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Pigeon Pea (Tur)'].modal_price,
    name = 'Pigeon Pea (Tur)_Price',
    opacity=0.5,
    yaxis='y2'
)
    
trace30 = go.Bar(
    x= df4_b.loc[df4_b['Commodity'] == 'Maize'].date,
    y= df4_b.loc[df4_b['Commodity'] == 'Maize'].modal_price,
    name = 'Maize_Price', 
    opacity=0.5,
    yaxis='y2'
)


data = [trace21, trace22, trace23, trace24, trace25, trace26, trace27, trace28, trace29, trace20,
        trace31, trace32, trace33, trace34, trace35, trace36, trace37, trace38, trace39, trace30]
    

layout = go.Layout(
    legend=dict(orientation="h"),
    
    title='Monthly Chart : Top 10 Commodity, Economics v/s Quantity',
    yaxis=dict(
        title='Quintals_Purchased_in_Maharashtra'
    ),
    yaxis2=dict(
        title='Average_Modal_Price_Per_Quintal(INR)',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_commodity")


# 
# 
# **Instructions to access Graphs:**
#     1. Dis-select all the legend boxes by single left clicking on each one of them.
#     2. Now select Individual 'Crop' and The 'Crop_Price' by single left clicking them and compare 
# 
# 
# 
# 
# <font color='blue'>Lets look into the graph:</font>
#     
#     1. Maize, Cotton, Soybean, Pigeon Pea are winter crops so their demands are higher in the winter months (NOV to FEB) -
#            There is no particular co-relation between the price and the amount purchased for both crops
#         
#     2. Methi (Bhaji) has interestingly dropped its demands in the year 2016 -
#            There is a negative co-relation between Price and Demand ( As price Increases,  Demand Decreases)
#         
#     3. Onion has seen a steep drop in demand in the month of Aug - Nov 2015 , owing to the high price increase - 
#            There is a negative co-relation between Price and Demand ( As price Increases,  Demand Decreases)
#         
#     4. Potato does not show any particular seasonal trend in its demand. However, over the time the demand has increased.
#            There is a negative co-relation between Price and Demand ( As price Increases,  Demand Decreases)
#         
#     5. Rice has no particular seasonal trend. However, over the time, the demand has significantly grown.
#            There is no co-relation between the price and demand for this crop.
#         
#     6. Tomato has no seasonal Trend.
#            There is a negative co-relation between Price and Demand ( As price Increases,  Demand Decreases) 

# ### <Font color = 'Green'> 5. District Versus Least Popular Commodity </font> : 
# #### This will help us identify the district which purchased least popular commodities over this period.

# In[ ]:


print(df1.sort_values('arrivals_in_qtl', ascending=True).head()) # for latest python df.sort has been deprecated and updated to df.sort_values


# *Feature 3:*

# In[ ]:


df0 = apmc
df0 = df0.loc[df0['Commodity'].isin(['CASTOR SEED','LEAFY VEGETABLE', 'Baru Seed', 'Jui', 'Papnas', 'MUSTARD', 
                                           'SARSAV', 'Terda','GOATS', 'Kalvad', 'Peer', 'NOLKOL', 'Plum',
                                          'GROUNDNUT PODS (WET)', 'Karvand', 'He Buffalo'])]
# Getting the Sum and the mean of the values
df0_a = pd.DataFrame(df0.groupby(['Commodity', 'date']).agg('sum')).reset_index()
df0_b = pd.DataFrame(df0.groupby(['Commodity', 'date']).agg('mean')).reset_index()


# In[ ]:


data = [
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'CASTOR SEED'].district_name, y=df0.loc[df0['Commodity'] == 'CASTOR SEED'].arrivals_in_qtl,
        name = 'Castor seed', opacity=0.5),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'GOATS'].district_name, y=df0.loc[df0['Commodity'] == 'GOATS'].arrivals_in_qtl,
        name = 'Goats', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Karvand'].district_name, y=df0.loc[df0['Commodity'] == 'Karvand'].arrivals_in_qtl,
        name = 'Karvand', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'He Buffalo'].district_name, y=df0.loc[df0['Commodity'] == 'He Buffalo'].arrivals_in_qtl,
        name = 'Buffalo', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'LEAFY VEGETABLE'].district_name, y=df0.loc[df0['Commodity'] == 'LEAFY VEGETABLE'].arrivals_in_qtl,
        name = 'Leafy Veggie', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Baru Seed'].district_name, y=df0.loc[df0['Commodity'] == 'Baru Seed'].arrivals_in_qtl,
        name = 'Baru Seed', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Jui'].district_name, y=df0.loc[df0['Commodity'] == 'Jui'].arrivals_in_qtl,
        name = 'Jui', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Papnas'].district_name, y=df0.loc[df0['Commodity'] == 'Papnas'].arrivals_in_qtl,
        name = 'Papnas', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'MUSTARD'].district_name, y=df0.loc[df0['Commodity'] == 'MUSTARD'].arrivals_in_qtl,
        name = 'Mustard', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'SARSAV'].district_name, y=df0.loc[df0['Commodity'] == 'SARSAV'].arrivals_in_qtl,
        name = 'Sarsav', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Terda'].district_name, y=df0.loc[df0['Commodity'] == 'Terda'].arrivals_in_qtl,
        name = 'Terda', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Kalvad'].district_name, y=df0.loc[df0['Commodity'] == 'Kalvad'].arrivals_in_qtl,
        name = 'Kalvad', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Peer'].district_name, y=df0.loc[df0['Commodity'] == 'Peer'].arrivals_in_qtl,
        name = 'Peer', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'Plum'].district_name, y=df0.loc[df0['Commodity'] == 'Plum'].arrivals_in_qtl,
        name = 'Plum', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'NOLKOL'].district_name, y=df0.loc[df0['Commodity'] == 'NOLKOL'].arrivals_in_qtl,
        name = 'Nolkol', opacity=0.5
    ),
    go.Bar(
        x=df0.loc[df0['Commodity'] == 'GROUNDNUT PODS (WET)'].district_name, y=df0.loc[df0['Commodity'] == 'GROUNDNUT PODS (WET)'].arrivals_in_qtl,
        name = 'Groundnut Pods', opacity=0.5
    )

]


layout = go.Layout(
    barmode='stack',
    title='Least Popular Commodities and their Purchase District',
    yaxis=dict(
        title='Quintals_Purchased'
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_commodity")


# Here, we can see that 'Pune' has the maximum number of 'Least Popular' commodities <Font Color = 'Blue'> (Less than 10 Quintals) </Font> being purchased. Interestingly, Mumbai,Wardha, Sangli which has the top 3 market share does not surface in this list at all!

# ### <Font Color ='Green'> 6. Price range of a Commodity V/S Districts </Font> : 
# #### To Identify the districts with highest and lowest price ranges for a particular Commodity

# *Feature 4:*

# In[ ]:


df6 = apmc
df6_a = pd.DataFrame(df6.groupby(['Commodity', 'district_name']).agg('mean')).reset_index()


# In[ ]:


Commodity = 'Maize'

trace00 = go.Scatter(
    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,
    y= df6_a.loc[df6_a['Commodity'] == Commodity].max_price,
    mode = 'lines+markers',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 2,
        dash = 'dot'),
    name = 'Price_MAX', 
    opacity=0.5
)

trace01 = go.Scatter(
    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,
    y= df6_a.loc[df6_a['Commodity'] == Commodity].modal_price,
    mode = 'lines+markers',
    line = dict(
        color = ('Red'),
        width = 2),
    name = 'Price_MODE',
    opacity=1.0
)

trace02 = go.Scatter(
    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,
    y= df6_a.loc[df6_a['Commodity'] == Commodity].min_price,
    mode = 'lines+markers',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 2,
        dash = 'dot'),
    name = 'Price_MIN', 
    opacity=0.5
)

trace03 = go.Bar(
    x= df6_a.loc[df6_a['Commodity'] == Commodity].district_name,
    y= df6_a.loc[df6_a['Commodity'] == Commodity].arrivals_in_qtl,
    name = 'Quantity', 
    opacity=0.2,
    yaxis='y2'
)


data = [trace00, trace01, trace02, trace03]

    

layout = go.Layout(
    legend=dict(orientation="v"),
    
    title='Price Range Chart district-wise',
    yaxis=dict(
        title='Price per Quintal'
    ),
    yaxis2=dict(
        title='Average_Quintal',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_commodity")


# <font color='blue'>
# **Instructions to access Graphs:**</font>
# <br>
# <font color='blue'>    
#     1. In the commodity Section type any Commodity name as per your choice (by default it is Onion)</font>
# <font color='blue'>     
#     2. Run the code snippet </font>
# 
# <br>
# 
# *Lets look into the graph:*
#     
#    1. For Onion (The most widely sold product, there is inconsistency in average prices across the districts. Onions are heavily costly in <Font Color = 'Red'> Jalna </font>.
#    
#    2. For Maize the cheapest price available is in <Font Color = 'Red'> Wasim </font>.
#    
#            
#            
# However, for all the commodities , there is no particular co-relation between the price each district and the demand. (It might be more clear if the economics is broken down, but the data is not available for that)
#     

# ### <font color='green'>7. Co-relation Check </font> : 
# #### To check if any of the attributes are heavily corelated with other : 
# 
# ##### a. for Most popular Commodities (top 5) 
# ##### b. for Least popular Commodities (bottom 10)

# In[ ]:


df7 = apmc
df7_a = df7.loc[df7['Commodity'].isin(['CASTOR SEED','LEAFY VEGETABLE', 'Baru Seed', 'Jui', 'Papnas', 'MUSTARD', 
                                           'SARSAV', 'Terda','GOATS', 'Kalvad', 'Peer', 'NOLKOL', 'Plum',
                                          'GROUNDNUT PODS (WET)', 'Karvand', 'He Buffalo'])]

df7_a= df7_a[[ 'Year', 'date' , 'arrivals_in_qtl', 'modal_price', 'min_price', 'max_price', 'Total_Market_Price' ]]

sns.set(style="white")

# Generate a large random dataset
d = df7_a

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


df7_b = df7.loc[df7['Commodity'].isin(['Onion', 'Soybean','Potato', 'Cotton', 'Rice(Paddy-Hus)'])]

df7_b= df7_b[[ 'Year', 'date' , 'arrivals_in_qtl', 'modal_price', 'min_price', 'max_price', 'Total_Market_Price' ]]

sns.set(style="white")

# Generate a large random dataset
d = df7_b

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# #### Note:
# 
# Comparing the second Image from the first one we can see that there is a striking difference in co-relation between 
#  Quantity and Modal Prices. 
#  
# For Top Performimng Commodities these fators are weakly corelated. Which is not the same case for least purchased commodities.
#  
# Is it because there are very less datapoints or scattered datapoints for the former?? (May be!!)

# ### <font color='green'>8. District versus total market price(Monthly Pattern) </font> : 
# #### To identify if there is any monthly pattren of total market price distribution for the top performing districts .

# In[ ]:


df8 = apmc
df8 = df8.loc[df8['district_name'].isin(['Mumbai','Sangli', 'Wardha', 'Pune', 'Thane', 'Wasim', 'Nasik','Yewatmal', 'Gondiya'])]
df8_a = pd.DataFrame(df8.groupby(['district_name', 'Month']).agg('mean')).reset_index()


# In[ ]:


a4_dims = (15, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)

# sns.set(style="whitegrid")
p = sns.violinplot(ax = ax,
                   data=df8_a,
                   x = 'district_name',
                   y = 'Total_Market_Price', bw=0.5, saturation = 1.25, width = 1.1
                   )
plt.show()


# #### Note:
# From the above graph we can say that the average market price over the months from top performing districts :
# 
# 1. Nasik, Pune, Thane have high seasonal market cap.
# 2. Wasim has the mean price of the commodities on higher band in most of the months.
# 3. Yewatmal, Gondiya  has the mean price of the commodities on lower band in most of the months.
# 4. Mumbai, Sangli and Wardha has their mean market price evenly distributed across the months.

# # Price Forecasting Model
# 
# 
# ---------------------------
# 
# 
# <font color = 'Blue'> Two Forecast models have been created:</font>
# 
# 1. AR-I-MA    : Based on regression and moving average 
# 
# 2. FbProphet  : Based on Bayesian fourier series 
# 
# ---------------------------
# 
# <font color = 'Blue'>In order to forecast the price of an item 3 month ahead, following steps has been performed:</font>
# 
# <font color = 'red'>Step 1:</font> Curate the dataset 
# <font color = 'red'>Step 2:</font> Up-Sample the dataset
# <font color = 'red'>Step 3:</font> Check for Stationarity
# <font color = 'red'>Step 4:</font> Check for trend
# <font color = 'red'>Step 5:</font> Use modeling equations
# 

# ###  <Font color = 'red' >ARIMA Forecasting Model </font>

# Auto Regression- Integrated- Moving Average is a model that incorporates future, past and seasonal variability of a data in modeling the forecasted value of a dataset,
# 
# ARIMA is a commomnly used timeseries forecast model in the industry today and is approached due to its robustness in terms of detecting seasonality.

# ** Curating Dataset according to the model checks and needs:**
# 
# > --> For our testing purpose Pigeon Pea has been selected as our test case.

# In[ ]:


df5 = apmc
df5 = df5.loc[df5['Commodity'].isin(['Pigeon Pea (Tur)'])]
df5 = pd.DataFrame(df5.groupby(['date']).agg('mean')).reset_index()
df5 = df5 [['date', 'modal_price']]
df5.tail()


# **Plotting the basic price variability with monthly data**

# In[ ]:



trace00 = go.Scatter(
    x= df5.date,
    y= df5.modal_price,
    mode = 'lines+markers',
)

layout = go.Layout(
    title='Pigeon_Pea Prices across the month'
)

data = [trace00]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="popular_commodity")


# <font color = 'blue'> Here we can see that the commodity price has increased over the period and then decreased recently. However, overall there has been some growth. </Font>
# 
# Before we move into forecasting lets check the seasonality of this data. 

# In[ ]:


#Preparing the dataset:   

df5.to_csv('forecast_APMC.csv')
# df5.dtypes # our datatypes for date column is in objects, we shall convert it to datetime 
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

df5 = pd.read_csv('forecast_APMC.csv', parse_dates=['date'], index_col='date',date_parser=dateparse) # indexed the date column
df5 = df5[['modal_price']]
df5.head()


# #### UP Sampling
# 
# 
# 
# <font color = 'Blue'> Need to upsample to weekly data from monthly since we have only 26 datapoints and we need atleast 100 points to perform ARIMA forecast </font>

# In[ ]:


ts = df5
ticks = ts.loc[:, ['modal_price']]
upsampled = ticks.modal_price.resample('7D', how = 'last') # upsampling it to weekly data points
interpolated = upsampled.interpolate(method='spline', order=2) # for more smoothened curve values. 
# print(interpolated )
plt.plot(interpolated, color='red', label = 'Interpolated weekly data')
plt.plot(ts, color='green', label = 'Original Monthly data')
plt.show()


# From the graph above, the <font color = 'green'> green </font> line plot is the original monthly data and the <font color = 'red'> red </font> line is the upsampled weekly data plot. 
# 
# The weekly data (red line) is more smoother as there are more number of interpolated data points. Now we have the number of data points = 113

# ### 1. ARIMA Forecast Model

# *Test-Train Split*

# In[ ]:


# split into train and test sets
 
start_train = 0          # add variable integer for location of data points for start of training
end_train = 100          # add variable integer for location of data points for end of training
start_test = 100        # add variable integer for location of data points for start of testing
end_test = 112          # add variable integer for location of data points for end of testing

X = interpolated

train, test = X[start_train:end_train], X[start_test:end_test]
# train, test = train_test_split(interpolated, test_size=0.2)       # splitting into train and test
history = [interpolated for interpolated in train]                  # creating a historical memory bin
predictions_forecast = list()                                    # creating an empty list for prediction forecasts
predictions_CI = list()                                          # creating an empty list for 95% Confidence Intervals
predictions_STD = list()  

test = test.tolist()    # converting test to list...
train_list = train.tolist()  # converting train to list...


# *Forecast*

# In[ ]:


# walk-forward validation
for t in range(len(test)):
	# fit model
	model = ARIMA(history, order=(2,1,2)) # ideal order is taken from the AIC test above
	model_fit = model.fit()

# one step forecast
# 	yhat = model_fit.forecast()[0]
	forecast, stderr, conf = model_fit.forecast(steps = 1, alpha = 0.05) 

# store forecast and ob
	predictions_forecast.append(forecast)
	predictions_STD.append(stderr)
	predictions_CI.append(conf)
	history.append(test[t])


# **Plot Residual Error**

# In[ ]:


residual_error = [predictions_forecast-test for predictions_forecast,test in zip(predictions_forecast,test)]
plt.plot(residual_error, color='orange', label='Residual Errors')


# ##### Note:
# 
# We can see that the residual error is mostly positive indicating that the model has over predicted most of the times, although by a small margin.

# **Forecasted versus Original Data Plot**

# In[ ]:


# plot forecasts against actual outcomes
plt.figure(figsize=(8,4))
plt.plot(test, color = 'blue', label = 'Test Data')
plt.plot(predictions_forecast, color='red' , label='Predicted Data')

plt.grid(True)
plt.xticks(rotation=90)
plt.xlabel("Units")
plt.ylabel("Power Demand (kW)")
# plt.ylim(ymin=0)
plt.legend()
plt.show()


# ------------------------------

# ### 2. FbProphet Model

# In[ ]:


# data curating
prophet_data = pd.read_csv('forecast_APMC.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)
prophet_data =  prophet_data[[ 'modal_price']]
# prophet_data.tail()


# **Curated and upsampled Data**

# In[ ]:


#UPSAMPLING
ticks = prophet_data.loc[:, ['modal_price']]
upsampled = ticks.modal_price.resample('D', how = 'last') # upsampling it to daily data points
data = upsampled.interpolate(method='spline', order=2) # for more smoothened curve values. 
# print(interpolated )
data.plot()
plt.show()


# In[ ]:


data=data.reset_index()
data = data[['date', 'modal_price']]
data.tail()


# In[ ]:


# Prophet requires columns ds (Date) and y (value)
prc = data.rename(columns={'date': 'ds', 'modal_price': 'y'})


# *Model Fitting*

# In[ ]:


# Make the prophet model and fit on the data
prc_prophet = fbprophet.Prophet(changepoint_prior_scale=0.90, seasonality_prior_scale = 0.99) # keeping High sensitivity to seasonal variability and changing points
prc_prophet.fit(prc) 


# *Forecasting Model*

# In[ ]:


# Make a future dataframe for next 90 days (3 months)
prc_forecast = prc_prophet.make_future_dataframe(periods=90, freq= "d") 
# Make predictions
prc_forecast = prc_prophet.predict(prc_forecast)


# In[ ]:


prc_forecast = prc_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n=90) 
prc_forecast.tail()

# yhat --> forecasted Modal Price
# yhat_lower and yhat_upper --> forecasted 95% confidence interval range of modal price


# In[ ]:


prc_prophet.plot(prc_forecast, xlabel = 'Date', ylabel = 'Commodity_Price')
plt.ylim(ymin=0);
plt.title('Price Predictions');


# ##### Note:
# 
# From this model we can get the upper and lower bound rages. However, it is to be seen that the 95% confidence interval becomes weak exponentially as the number of days increases.

# In[ ]:




