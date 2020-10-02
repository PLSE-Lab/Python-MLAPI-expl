#!/usr/bin/env python
# coding: utf-8

# <h2>Complete exploratory data analysis</h2>
# 
# We will be using pandas, scipy, seaborn and plotly to explore the train.csv data. Interactive plots will be used to visualize the time series since we have many data points.
# 
# Some key factors about this dataset:
# * Number of rows: 913k
# * Just three columns: store, item and sales.
# * Fifty different items and ten stores
# * Sales are measured for each item, store and date (daily)
# * Five years time frame (2013/01/01 to 2017/12/31)
# * No missing data
# 
# I will be updating this notebook as possible. Please upvote if you find usefull, thanks.
# 

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import warnings
# Matplotlib e Seaborn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)
# Plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode()


# <h3>1. Statistics</h3>
# 
# As mentioned before there are no missing values in this dataset and all values are numeric (integers). Let's start by looking at some basic statistics: 

# In[ ]:


# Read train.csv file and set datatype
data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
df = pd.read_csv("../input/train.csv", parse_dates= ['date'], dtype= data_type)
df.describe()


# The store and item columns are in the range 1 to 10 and 1 to 50 respectively. Sales values are in the range 0 to 231 with 52.25 mean.
# 
# Let's plot the sales distribution:

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of sales - for each item, date and store")
ax = sns.distplot(df['sales'])


# Now we will compare our data to the normal distribution using Scipy normaltest:

# In[ ]:


print("p-value for sales distribution: {}".format(st.normaltest(df.sales.values)[1]))
plt.figure(figsize=(12,5))
plt.title("Distribution of sales vs best fit normal distribution")
ax = sns.distplot(df.sales, fit= st.norm, kde=True, color='g')


# In the above plot, the green line represents our sales distribution, while the black line is the best normal distribution we can fit to our data. The p-value indicates that the null hypothesis can be rejected and therefore our data dont fit a normal distribution. Now let's try to find the best distribution for sales based on the sum of square error (SSE):

# In[ ]:


# Code (function) adapted from https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
def best_fit_distribution(data, bins= 200):
    """Model data by finding best fit distribution to data"""
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    DISTRIBUTIONS = [        
        st.alpha,st.beta,st.chi,st.chi2, st.dgamma,st.dweibull,st.erlang,st.exponweib,
        st.f, st.genexpon,st.gausshyper,st.gamma, st.johnsonsb,st.johnsonsu, st.norm,
        st.rayleigh,st.rice,st.recipinvgauss, st.t, st.weibull_min,st.weibull_max
    ]

    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    for distribution in DISTRIBUTIONS:
        #print("Testing " + str(distribution))

        # Try to fit the distribution
        #try:
        # Ignore warnings from data that can't be fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # fit dist to data
            params = distribution.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))

            # identify if this distribution is better
            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse
        #except Exception:
        #    pass

    return (best_distribution.name, best_params)

dist_name, best_params = best_fit_distribution(df.sales.values)
print("Best distribution found: {}, with parameters: {}".format(dist_name, best_params))


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of sales vs Johnson-SB distribution (best fit)")
ax = sns.distplot(df.sales, fit= st.johnsonsb, kde=True, color='g')


# <h3>2. Total sales</h3>
# 
# Graph of <b>average monthly sales</b> for all stores and items:

# In[ ]:


monthly_df = df.groupby([df.date.dt.year, df.date.dt.month])['sales'].mean()
monthly_df.index = monthly_df.index.set_names(['year', 'month'])
monthly_df = monthly_df.reset_index()
x_axis = []
for y in range(13, 18):
    for m in range(1,12):
        x_axis.append("{}/{}".format(m,y))
trace = go.Scatter(x= x_axis, y= monthly_df.sales, mode= 'lines+markers', name= 'sales avg per month', line=dict(width=3))
layout = go.Layout(autosize=True, title= 'Sales - average per month', showlegend=True)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# <b>Total sales by year</b>

# In[ ]:


year_df = df.groupby(df.date.dt.year)['sales'].sum().to_frame()

trace = go.Bar(
    y= year_df.sales, x= ['2013','2014','2015','2016','2017'],
    marker=dict(color='rgba(179, 143, 0, 0.6)', line=dict(color='rgba(179, 143, 0, 1.0)', width=1)),
    name='Total sales by year', orientation='v'
)

layout = go.Layout(autosize=False, title= 'Total sales by year', showlegend=True, width=600, height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# Sales are slowing increasing each year and there is a clear seasonality effect

# <h3>3. Sales by store</h3>
# 
# Average sales per month and store:

# In[ ]:


monthly_df = df.groupby([df.date.dt.year, df.date.dt.month, 'store']).mean()
monthly_df.index = monthly_df.index.set_names(['year', 'month', 'store'])
monthly_df = monthly_df.reset_index()

traces = []
for i in range(1, 11):
    store_sales = monthly_df[monthly_df.store == i]
    trace = go.Scatter(x= x_axis, y= store_sales.sales, mode= 'lines+markers', name= 'Store '+str(i), line=dict(width=3))
    traces.append(trace)
layout = go.Layout(autosize=True, title= 'Sales - average per month', showlegend=True)
fig = go.Figure(traces, layout=layout)
iplot(fig)


# **Sales per store - bar chart**

# In[ ]:


store_total = df.groupby(['store'])['sales'].sum().to_frame().reset_index()
store_total.sort_values(by = ['sales'], ascending=True, inplace=True)
labels = ['Store {}'.format(i) for i in store_total.store]

trace = go.Bar(
    y= store_total.sales, x= labels,
    marker=dict(color='rgba(255, 65, 54, 0.6)', line=dict(color='rgba(255, 65, 54, 1.0)', width=1)),
    name='Total sales per store', orientation='v'
)

layout = go.Layout(autosize=True, title= 'Total sales by store')
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# **Boxplot**

# In[ ]:


store_sum = df.groupby(['store', 'date'])['sales'].sum()
traces = []

for i in range(1, 11):
    s = store_sum[i].to_frame().reset_index()
    trace = go.Box(y= s.sales, name= 'Store {}'.format(i), jitter=0.8, whiskerwidth=0.2, marker=dict(size=2), line=dict(width=1))
    traces.append(trace)

layout = go.Layout(
    title='Sales BoxPlot for each store',
    yaxis=dict(
        autorange=True, showgrid=True, zeroline=True,
        gridcolor='rgb(233,233,233)', zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2, gridwidth=1
    ),
    margin=dict(l=40, r=30, b=80, t=100), showlegend=False,
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig)


# **Sales for each store - time series**
# 
# It's possible to select the store in the dropdown menu and the time frame on the range slider

# In[ ]:


data = []
for i in range(1,11):
    s = store_sum[i].to_frame().reset_index()
    trace = go.Scatter(
        x= s.date,
        y= s.sales,
        name = "Store "+str(i),
        opacity = 0.9)
    data.append(trace)

# Buttons to select a specific store visualization
update_buttons = []
for i in range(10):
    visible = [True if j == i else False for j in range(10)]
    button= dict(label = 'Store ' + str(i+1), method= 'update', args= [{'visible': visible}])
    update_buttons.append(button)
# Button to return to all stores visualization
update_buttons.append(dict(label = 'All', method= 'update', args= [{'visible': [True]*10}]))

updatemenus = list([dict(active=-1, buttons=list(update_buttons))])

layout = dict(
    title='Sales by store and time',
    updatemenus= updatemenus,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=12, label='12m', step='month', stepmode='backward'),
                dict(count=24, label='24m', step='month', stepmode='backward'),
                dict(count=36, label='36m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(), type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, validate= False)


# <h3>4. Sales by item</h3>
# 
# We have 50 different products with total sales that goes from 335k for Item 5 to 1.6M for item 15.

# In[ ]:


item_total = df.groupby(['item'])['sales'].sum().to_frame().reset_index()
item_total.sort_values(by = ['sales'], ascending=False, inplace=True)
labels = ['Item {}'.format(i) for i in item_total.item]

trace = go.Bar(
    y= item_total.sales, x= labels,
    marker=dict(color='rgba(33, 33, 135, 0.6)', line=dict(color='rgba(33, 33, 135, 1.0)', width=1)),
    name='Total sales by item', orientation='v'
)
layout = go.Layout(autosize=True, title= 'Sales per item (all time)')
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# **Boxplot**

# In[ ]:


item_sum = df.groupby(['item', 'date'])['sales'].sum()
traces = []

for i in range(1, 51):
    s = item_sum[i].to_frame().reset_index()
    trace = go.Box(y= s.sales, name= 'Item {}'.format(i), jitter=0.8, whiskerwidth=0.2, marker=dict(size=2), line=dict(width=1))
    traces.append(trace)

layout = go.Layout(
    title='Sales BoxPlot for each item',
    yaxis=dict(
        autorange=True, showgrid=True, zeroline=True,
        gridcolor='rgb(233,233,233)', zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2, gridwidth=1
    ),
    margin=dict(l=40, r=30, b=80, t=100), showlegend=False,
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig)


# <h3>5. Sales for each item and store</h3>
# 
# To conclude, we will be ploting the time series for each store and item with a dropdown menu where it is possible to select a specific cobination.

# In[ ]:



data = []
default_visible = [False]*500
default_visible[0] = True
for i in range(1, 51):
    _df = df[df.item == i]
    for s in range(1,11):
        trace = go.Scatter(
            x= _df[_df.store == s].date,
            y= _df[_df.store == s].sales,
            name = "Store {} Item {} ".format(s, i),
            visible = False,
            opacity = 0.9)
        data.append(trace)

# Buttons to select a specific item and store visualization
update_buttons = []
for i in range(1, 51):
    for s in range(1, 11):
        visible = [True if k == i*s else False for k in range(1,501)]  
        button= dict(label = 'Store {} Item {}'.format(s,i), method= 'update', args= [{'visible': visible}])
        update_buttons.append(button)

updatemenus = list([dict(active=-1, buttons=list(update_buttons))])

layout = dict(
    title='Sales by store and item',
    #visible = default_visible,
    updatemenus= updatemenus,
    xaxis=dict(rangeslider=dict(), type='date')
)

fig = dict(data=data, layout=layout)
iplot(fig, validate= False)

