#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing Necessary libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import pycountry
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '../input/covid19-global-forecasting-week-4/'
os.listdir(path)


# In[ ]:


data = pd.read_csv(path + "train.csv")
data.head()


# In[ ]:


data.info()


# * ## Plotting countries with confirmed cases

# In[ ]:


subset_by_max_date = data[data['Date'] == data['Date'].max()]
subset_by_max_date.head()


# In[ ]:


country_wise_case = subset_by_max_date.groupby(by = 'Country_Region')['ConfirmedCases'].agg([np.sum]).reset_index().rename(columns = {'Country_Region': 'Country', 'sum': 'confirmed'})
country_wise_case.head()


# In[ ]:


X = country_wise_case['Country'].values
Y = country_wise_case['confirmed'].values
bar = go.Bar(x = Y, y = X, marker=dict(color=Y, colorscale='Reds', showscale=True), orientation = 'h')
layout = go.Layout(title = dict(text = 'Total no of confirmed cases country wise', x = 0.5), 
                   yaxis = dict(title = "Countries"),
                   xaxis = dict(title = "Confirmed Case"), height = 1500)
fig = go.Figure(data=[bar], layout = layout)
py.iplot(fig)


# ## Since the plot is not much visible let's focus on only top 20 countries with most confirmed cases

# In[ ]:


country_wise_case.sort_values('confirmed', ascending = False, inplace = True)
X = country_wise_case['Country'].values[:20]
Y = country_wise_case['confirmed'].values[:20]
bar = go.Bar(x = X, y = Y, marker = dict(color = Y, colorscale = 'Reds', showscale = True))
layout = go.Layout(title = dict(text = "Top 20 countries with most confirmed cases", x = 0.3), 
                   yaxis = dict(title = 'Confirmed Cases'), 
                  xaxis = dict(title = 'Country Names'))
fig = go.Figure(data = [bar], layout = layout)
py.iplot(fig)


# ## USA has the most confirmed cases
# 
# ### Let's Focus more on USA

# In[ ]:


usa = data[data['Country_Region'] == 'US']
usa.head()


# In[ ]:


# No null column
usa.info()


# ### Let's check for Top 10 Province wise distribution

# In[ ]:


province_wise_case = usa.groupby(by = 'Province_State')['ConfirmedCases'].agg([np.max]).reset_index()                        .rename(columns = {'Province_State': 'Province', 'amax':'Confirmed'}).sort_values('Confirmed', ascending = False)


# In[ ]:


labels = province_wise_case['Province'].values[:10]
values = province_wise_case['Confirmed'].values[:10]
pie = go.Pie(labels = labels, values = values, name = "Provinces", 
             hole = 0.4, domain = {'x': [0, 0.5]})
layout = dict(title = dict(text = 'Province wise distribution', x = 0.3, xanchor = 'center', yanchor = 'top'), 
              legend = dict(orientation = 'h'), 
              annotations = [dict(x = 0.2, y = 0.5, text='Provinces', showarrow=False, font=dict(size=20))])
fig = go.Figure(data = [pie], layout = layout)
py.iplot(fig)


# ## The most affected state turned out to be new york

# ## Let's check the growth rate of confirmed cases and death rate in New York

# In[ ]:


new_york = usa[usa['Province_State'] == 'New York']
new_york.head()


# In[ ]:


X = new_york['Date']
Y_confirm = new_york['ConfirmedCases']
Y_death = new_york['Fatalities']
line_confirm = go.Scatter(x = X, y = Y_confirm, mode = "lines+markers", 
                          name = "Confirmed", line = dict(color = 'firebrick', width = 4))
line_death = go.Scatter(x = X, y = Y_death, mode = "lines", 
                        name = "Fatalities", line = dict(color = 'royalblue', width = 2))

layout = dict(title = dict(text = 'Growth Line Graph of confirmed cases and fatalities in NY', x = 0.2), 
              xaxis = dict(title = "Days"),
              yaxis = dict(title = "People"))

fig = go.Figure(data = [line_confirm, line_death], layout = layout)
py.iplot(fig)


# ## Rate of spread is exponential whereas rate of fatalities is almost linear
# 
# ### Let's check it on log scale

# In[ ]:


X = new_york['Date']
Y_confirm = np.log10(new_york['ConfirmedCases'].values + 1)
Y_death = np.log10(new_york['Fatalities'].values + 1)
line_confirm = go.Scatter(x = X, y = Y_confirm, mode = "lines+markers", 
                          name = "Confirmed", line = dict(color = 'firebrick', width = 4))
line_death = go.Scatter(x = X, y = Y_death, mode = "lines", 
                        name = "Fatalities", line = dict(color = 'royalblue', width = 2))

layout = dict(title = dict(text = 'Growth Line Graph of confirmed cases and fatalities in NY (on log scale)', x = 0.2), 
              xaxis = dict(title = "Days"),
              yaxis = dict(title = "People"))

fig = go.Figure(data = [line_confirm, line_death], layout = layout)
py.iplot(fig)


# ## Similarly Let's Check for fatalities

# In[ ]:


country_wise_death = subset_by_max_date.groupby(by = 'Country_Region')['Fatalities'].agg([np.sum]).reset_index().rename(columns = {'Country_Region': 'Country', 'sum': 'Deaths'})
country_wise_death.head()


# In[ ]:


X = country_wise_death['Country'].values
Y = country_wise_death['Deaths'].values
bar = go.Bar(x = Y, y = X, marker=dict(color=Y, colorscale='Reds', showscale=True), orientation = 'h')
layout = go.Layout(title = 'Total no of Deaths country wise', yaxis = dict(title = "Countries"),
                   xaxis = dict(title = "Deaths"), height = 1500)
fig = go.Figure(data=[bar], layout = layout)
py.iplot(fig)


# ## Since the plot is not much visible let's focus on only top 20 countries with most Deaths

# In[ ]:


country_wise_death.sort_values('Deaths', ascending = False, inplace = True)
X = country_wise_death['Country'].values[:20]
Y = country_wise_death['Deaths'].values[:20]
bar = go.Bar(x = X, y = Y, marker = dict(color = Y, colorscale = 'Reds', showscale = True))
layout = go.Layout(title = "Top 20 countries with most Deaths", yaxis = dict(title = 'Deaths'), 
                  xaxis = dict(title = 'Country Names'))
fig = go.Figure(data = [bar], layout = layout)
py.iplot(fig)


# ## Not suprisingly USA has more deaths as well
# 
# ### Let's Focus more on Provinces

# ### Let's check for Top 10 Province wise distribution

# In[ ]:


province_wise_case = usa.groupby(by = 'Province_State')['Fatalities'].agg([np.max]).reset_index()                        .rename(columns = {'Province_State': 'Province', 'amax':'Deaths'}).sort_values('Deaths', ascending = False)


# In[ ]:


labels = province_wise_case['Province'].values[:10]
values = province_wise_case['Deaths'].values[:10]
pie = go.Pie(labels = labels, values = values, name = "Provinces", 
             hole = 0.4, domain = {'x': [0, 0.5]})
layout = dict(title = dict(text = 'Province wise distribution', x = 0.3, xanchor = 'center', yanchor = 'top'), 
              legend = dict(orientation = 'h'), 
              annotations = [dict(x = 0.2, y = 0.5, text='Provinces', showarrow=False, font=dict(size=20))])
fig = go.Figure(data = [pie], layout = layout)
py.iplot(fig)


# ## New York has much more death rate as compared to other provinces

# In[ ]:




