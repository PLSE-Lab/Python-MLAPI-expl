#!/usr/bin/env python
# coding: utf-8

# **Brazil's House of Deputies Reimbursements**
# 
# This notebook, makes an analysis of the  Brazilian deputies expenses from 2013 to 2017.

# ![](http://www.vermelho.org.br/admin/arquivos/biblioteca/cd99755.jpg)

# In[ ]:


import pandas as pd

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

init_notebook_mode(connected=True)


# **Load Data**

# In[ ]:


path2data = '../input/'


# In[ ]:


df_politicians = pd.read_csv(path2data + 'deputies_dataset.csv', low_memory=False)
df_dirty_politicians = pd.read_csv(path2data + 'dirty_deputies_v2.csv', low_memory=False)


# In[ ]:


print('Dataset shape: {}'.format(df_politicians.shape))


# In[ ]:


df_politicians['receipt_date'] = pd.to_datetime(df_politicians['receipt_date'])
df_politicians['month'] = df_politicians['receipt_date'].dt.month
df_politicians['year'] = df_politicians['receipt_date'].dt.year
df_politicians['day'] = df_politicians['receipt_date'].dt.day


# In[ ]:


df_politicians.head()


# In[ ]:


# get data from 2013 to 2017
df_politicians = df_politicians[ (df_politicians['year']>=2013) &  (df_politicians['year']<=2017)]


# In[ ]:


df_dirty_politicians['receipt_date'] = pd.to_datetime(df_dirty_politicians['refund_date'])
df_dirty_politicians['month'] = df_dirty_politicians['receipt_date'].dt.month
df_dirty_politicians['year'] = df_dirty_politicians['receipt_date'].dt.year
df_dirty_politicians['day'] = df_dirty_politicians['receipt_date'].dt.day


# In[ ]:


df_dirty_politicians.head()


# **Exploratory Data Analysis**

# **Receipt values by deputies of each state and Political**
# 
# As can be seen from the chart below, most of the expenses are from deputies from the state of Sao Paulo. With regard to the political party, the one that had the highest expenses in the analyzed period was the PT (146M).

# In[ ]:


trace1 = go.Bar(
            x=df_politicians.groupby(['state_code'])['receipt_value'].sum().sort_values(ascending=True).values,
            y=df_politicians.groupby(['state_code'])['receipt_value'].sum().sort_values(ascending=True).index,
            orientation = 'h'
)

trace2 = go.Bar(
            x=df_politicians.groupby(['political_party'])['receipt_value'].sum().sort_values(ascending=True).values,
            y=df_politicians.groupby(['political_party'])['receipt_value'].sum().sort_values(ascending=True).index,
            orientation = 'h'
)

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=600, width=1000, title='Receipt Value counter')
iplot(fig, filename='simple-subplot-with-annotations')


# In[ ]:


df_politicians.groupby(['deputy_name', 'political_party'])['receipt_value'].sum().sort_values(ascending=True)[-10:]


# # Individual analysis

# In[ ]:


deputy_name = 'Jair Bolsonaro' # change here to analyze other deputy


# In[ ]:


deputy_data = df_politicians[df_politicians['deputy_name'] == deputy_name]


# In[ ]:


deputy_data.head()


# **Trajectory of receipt value in each month**
# 
# As can be seen, December is the month in which the number of expenses is higher

# In[ ]:


receipt_value_month_year = deputy_data.groupby(['year','month'])['receipt_value'].sum().to_frame().unstack(level=-1)
receipt_value_month_year.columns = receipt_value_month_year.columns.droplevel(0)


# In[ ]:


receipt_value_month_year.head()


# In[ ]:


data = []
years = receipt_value_month_year.index

for idx in range(0, len(years)):

    trace = go.Bar(
        x=receipt_value_month_year.iloc[0].index,
        y=receipt_value_month_year.iloc[idx],
        name=str(years[idx])
    )
    data.append(trace)
    
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
fig['layout'].update(height=600, width=1000, title='Receipt Value in each month')
iplot(fig, filename='grouped-bar')


# In[ ]:


deputy_data['receipt_description'].unique()


# **Distribution of paid services**
# 
# 

# In[ ]:


fig = {
    'data': [{'labels': deputy_data.groupby(['receipt_description'])['receipt_value'].sum().index,
              'values': deputy_data.groupby(['receipt_description'])['receipt_value'].sum().values,
              'type': 'pie'}],
    'layout': {'title': 'Expenses distribution for each service',
              'height': 400, 
               'width':1000}
     }

iplot(fig, filename='basic_pie_chart')


# # Airline Tickets

# In[ ]:


deputy_data[deputy_data['receipt_description'] == 'Airline tickets'].groupby(['establishment_name'])['receipt_value'].sum().sort_values(ascending=False)


# In[ ]:


deputy_airline_df = deputy_data[deputy_data['receipt_description'] == 'Airline tickets']


# In[ ]:


data = [go.Scatter(
          x=deputy_airline_df['receipt_date'],
          y=deputy_airline_df['receipt_value'],
          mode = 'markers'
    )]

iplot(data)


# # Number of trips by day
# 
# As can be seen below, there are deputies who have made many trips in just one day.

# In[ ]:


nb_trips = deputy_airline_df.groupby(['receipt_date'])['receipt_date'].count().sort_values(ascending=False)


# In[ ]:


nb_trips.head(10)


# In[ ]:


deputy_airline_df[deputy_airline_df['receipt_date'] == nb_trips.index[0]]


# # Postal Services

# In[ ]:


aux = deputy_data[deputy_data['receipt_description'] == 'Postal Services'].groupby(['establishment_name'])['receipt_value'].sum().to_frame()
aux.sort_values('receipt_value', ascending=False, inplace=True)


# In[ ]:


aux


# In[ ]:


types_of_postal_services = deputy_data[deputy_data['receipt_description'] == 'Postal Services']['establishment_name'].unique()


# In[ ]:


print('Types of postal services: {}'.format(len(types_of_postal_services)))


# In[ ]:


data = []
for ps in types_of_postal_services:

    aux = deputy_data[(deputy_data['receipt_description'] == 'Postal Services') & (deputy_data['establishment_name'] == ps)]
    trace = go.Box(
        x=aux['receipt_value'].values,
        name = ps
    )
    
    data.append(trace)

layout = go.Layout(
    width=800,
    yaxis=dict(
        zeroline=False
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # Dissemination of the Parliamentary Activity.

# In[ ]:


dpa = deputy_data[deputy_data['receipt_description'] == 'Dissemination of the Parliamentary Activity.'].groupby(['establishment_name', 'year'])['receipt_value'].sum().to_frame().unstack(level=-1)
dpa.columns = dpa.columns.droplevel(0)


# In[ ]:


dpa

