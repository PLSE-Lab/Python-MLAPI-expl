#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Power is one of the most critical components of infrastructure crucial for the economic growth and welfare of nations. The existence and development of adequate infrastructure is essential for sustained growth of the Indian economy. India is the world's third largest producer and third largest consumer of electricity. Sustained economic growth continues to drive electricity demand in India.
# 
# 
# Consumption of electricity is known to follow economic activity closely. The industries that produce essential goods are operating at very low utilization levels. Hence, in such a scenario one expects electricity demands to go down.
# 
# Here is a notebook to give you a brief intro of the dataset that I created, through interactive visualizations which will allow you to browse through data visually. The intension is to build an intuition about the data thereby being able to answer questions of relevance. The date ranges from 28/10/2019 to 23/05/2020.
# 
# In this notebook I have put my hands on interactive plots which will enable anyone to browse the data with a few clicks. I hope you like it and get your hands on the dataset to build a notebook of your own. 
# 
# Do comment your suggestions and review of my work below. Hope you enjoy as much as I did while creating it. :)
# 

# In[ ]:


pip install bar_chart_race


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import HTML
import calendar
from plotly.subplots import make_subplots
import bar_chart_race as bcr


# In[ ]:


df = pd.read_csv('../input/state-wise-power-consumption-in-india/power data.csv')
df_long = pd.read_csv('../input/state-wise-power-consumption-in-india/long_data_.csv')


# In[ ]:


df.info()


# In[ ]:


df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
df_long['Dates'] = pd.to_datetime(df_long.Dates, dayfirst=True)


# # Region Wise Daily Power Consumption

# In[ ]:


df['NR'] = df['Punjab']+ df['Haryana']+ df['Rajasthan']+ df['Delhi']+df['UP']+df['Uttarakhand']+df['HP']+df['J&K']+df['Chandigarh']

df['WR'] = df['Chhattisgarh']+df['Gujarat']+df['MP']+df['Maharashtra']+df['Goa']+df['DNH']

df['SR'] = df['Andhra Pradesh']+df['Telangana']+df['Karnataka']+df['Kerala']+df['Tamil Nadu']+df['Pondy']

df['ER'] = df['Bihar']+df['Jharkhand']+ df['Odisha']+df['West Bengal']+df['Sikkim']

df['NER'] =df['Arunachal Pradesh']+df['Assam']+df['Manipur']+df['Meghalaya']+df['Mizoram']+df['Nagaland']+df['Tripura']


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.Date, y=df.NR,
    mode='lines+markers',
    name='Northern region',
    marker=dict(
            color='rgba(300, 50, 50, 0.8)',
            size=5,
            line=dict(
                color='DarkSlateGrey',
                width = 1
                     )
                )
))

fig.add_trace(go.Scatter(
    x=df.Date, y=df.SR,
    mode='lines+markers',
    name='Southern Region',
    marker=dict(
            color='rgba(50, 300, 50, 0.8)',
            size=5,
            line=dict(
                color='DarkSlateGrey',
                width = 1
                     )
                )
))

fig.add_trace(go.Scatter(
    x=df.Date, y=df.ER,
    mode='lines+markers',
    name='Eastern Region',
    marker=dict(
            color='rgba(50, 50, 300, 0.8)',
            size=5,
            line=dict(
                color='DarkSlateGrey',
                width = 1
                     )
                )
))

fig.add_trace(go.Scatter(
    x=df.Date, y=df.WR,
    mode='lines+markers',
    name='Western Region',
    marker=dict(
            color='rgba(300, 100, 200, 0.8)',
            size=5,
            line=dict(
                color='DarkSlateGrey',
                width = 1
                     )
                )
))

fig.add_trace(go.Scatter(
    x=df.Date, y=df.NER,
    mode='lines+markers',
    name='North-Eastern',
    marker=dict(
            color='rgba(100, 200, 300, 0.8)',
            size=5,
            line=dict(
                color='DarkSlateGrey',
                width = 1
                     )
                )
))


fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.update_layout(title='Power Consumption in Various Region')
fig.update_layout(width=800,height=500)
fig.show()


# # State-wise mean Power consumption

# In[ ]:


df1= df[['Date', 'Punjab', 'Haryana', 'Rajasthan', 'Delhi', 'UP',
       'Uttarakhand', 'HP', 'J&K', 'Chandigarh', 'Chhattisgarh', 'Gujarat',
       'MP', 'Maharashtra', 'Goa', 'DNH', 
       'Andhra Pradesh', 'Telangana', 'Karnataka', 'Kerala', 'Tamil Nadu',
       'Pondy', 'Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Sikkim',
       'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram',
       'Nagaland', 'Tripura']]

df1 = df1.set_index('Date')
bcr.bar_chart_race(df1, figsize=(4, 3.5),period_length =500,filename = None, title='power usage by states')


# # Monthly average Power Consumption

# In[ ]:


monthly_df = df_long.groupby([df_long.Dates.dt.year, df_long.Dates.dt.month,df_long.States,df_long.Regions, df_long.latitude,df_long.longitude])['Usage'].mean()
monthly_df.index = monthly_df.index.set_names(['year', 'month','State','Region','latitude','longitude'])
monthly_df = monthly_df.reset_index()
monthly_df['month'] = monthly_df['month'].apply(lambda x: calendar.month_abbr[x])


# In[ ]:


monthly_df.head()


# In[ ]:


fig = px.sunburst(monthly_df, path=['Region', 'State','month'], values='Usage',
                  color='Usage',
                  color_continuous_scale='RdBu')
fig.update_layout(title='Click various Regions/States to view power distribution')
fig.update_layout( width=800,height=600)
fig.show()


# In[ ]:


fig = px.bar(monthly_df, x="Region", y="Usage",color='State',animation_frame = 'month')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.update_layout(title='Region-wise Bar plots')
fig.show()


# # Before and After lockdown Scenarios

# In[ ]:


df_before = df.iloc[0:150,:]
df_after = df.iloc[151:,]


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Gujarat'], name='Gujarat before lockdown',fill='tonexty',
    line=dict(width=2,dash='dot',color='firebrick') 
))
fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Maharashtra'], name='Maharashtra before lockdown',fill='tonexty',
    line=dict(width=2,dash='dot',color='coral')
))

fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['MP'], name='MP before lockdown',fill='tozeroy',
    line=dict(width=2,dash='dot',color='darkred')
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Gujarat'],name='Gujarat after lockdown',fill='tozeroy',
    line=dict(color='firebrick', width=2)
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Maharashtra'],name='Maharashtra after lockdown',fill='tozeroy',
    line=dict(color='coral', width=2)
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['MP'],name='MP after lockdown',fill='tozeroy',
    line=dict(color='darkred', width=2)
))

fig.update_layout(title='Power Consumption in top 3 WR states')
fig.update_layout( width=800,height=500)
fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Karnataka'], name='Karnataka before lockdown',fill='tonexty',
    line=dict(width=2,dash='dot',color='skyblue') 
))
fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Tamil Nadu'], name='Tamil Nadu before lockdown',fill='tonexty',
    line=dict(width=2,dash='dot',color='lightblue')
))

fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Telangana'], name='Telangana before lockdown',fill='tozeroy',
    line=dict(width=2,dash='dot',color='midnightblue')
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Karnataka'],name='Karnataka after lockdown',fill='tozeroy',
    line=dict(color='skyblue', width=2)
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Tamil Nadu'],name='Tamil Nadu after lockdown',fill='tozeroy',
    line=dict(color='lightblue', width=2)
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Telangana'],name='Telangana after lockdown',fill='tozeroy',
    line=dict(color='midnightblue', width=2)
))

fig.update_layout(title='Power Consumption in top 3 WR states')
fig.update_layout( width=800,height=500)
fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Rajasthan'], name='Rajasthan before lockdown',fill='tonexty',
    line=dict(width=2,dash='dot',color='darkviolet') 
))

fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['UP'], name='UP before lockdown',fill='tonexty',
    line=dict(width=2,dash='dot',color='deeppink')
))


fig.add_trace(go.Scatter( x=df_before['Date'], y=df_before['Haryana'], name='Haryana before lockdown',fill='tozeroy',
    line=dict(width=2,dash='dot',color='indigo')
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Rajasthan'],name='Rajasthan after lockdown',fill='tozeroy',
    line=dict(color='darkviolet', width=2)
))

fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['UP'],name='UP after lockdown',fill='tonexty',
    line=dict(color='deeppink', width=2)
))


fig.add_trace(go.Scatter(x=df_after['Date'], y=df_after['Haryana'],name='Haryana after lockdown',fill='tozeroy',
    line=dict(color='indigo', width=2)
))

fig.update_layout(title='Power Consumption in top 3 NR states')
fig.update_layout( width=800,height=500)
fig.show()


# # Maximum value reached

# In[ ]:


WR_df = df_long[df_long['Regions']=='WR']
NR_df = df_long[df_long['Regions']=='NR']
SR_df = df_long[df_long['Regions']=='SR']
ER_df = df_long[df_long['Regions']=='ER']
NER_df = df_long[df_long['Regions']=='NER']


# In[ ]:



fig= go.Figure(go.Indicator(
mode = "gauge+number",
value = WR_df['Usage'].max(),
title = {'text': "Max Power Usage In WR:Maharashtra 13/05/2020"},
gauge = {
    'axis': {'range': [None, 500], 'tickwidth': 1},
    'threshold': {
        'line': {'color': "red", 'width': 4},
        'thickness': 0.75,
        'value': 490}}
))

fig.show()


# In[ ]:


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = NR_df['Usage'].max(),
    title = {'text': "Max Power Usage In NR :UP 09/05/2020"},
    gauge = {
        'axis': {'range': [None, 500], 'tickwidth': 1},
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 490}}
))
fig.update_layout(legend_title_text='State   Date::UP')
fig.show()


# In[ ]:


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = SR_df['Usage'].max(),
    title = {'text': "Max Power Usage In SR : Tamil Nadu  01/11/2019"},
    gauge = {
        'axis': {'range': [None, 500], 'tickwidth': 1},
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 490}}
))

fig.show()


# In[ ]:


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = ER_df['Usage'].max(),
    title = {'text': "Max Power Usage In ER: West Bangal 04/05/2020"},
    gauge = {
        'axis': {'range': [None, 500], 'tickwidth': 1},
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 490}}
))

fig.show()


# In[ ]:


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = NER_df['Usage'].max(),
    title = {'text': "Max Power Usage In NER: Assam 05/05/2020"},
    gauge = {
        'axis': {'range': [None, 500], 'tickwidth': 1},
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 490}}
))

fig.show()


# # Plotting on maps

# In[ ]:


df_long = pd.read_csv('../input/state-wise-power-consumption-in-india/long_data_.csv')
df_long.dropna(inplace = True)


# In[ ]:


fig = px.scatter_geo(df_long,'latitude','longitude', color="Regions",
                     hover_name="States", size="Usage",
                     animation_frame="Dates", scope='asia')
fig.update_geos(lataxis_range=[5,35], lonaxis_range=[65, 100])
fig.show()


# In[ ]:




