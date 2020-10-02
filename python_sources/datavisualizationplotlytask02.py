#!/usr/bin/env python
# coding: utf-8

# # $$Introduction$$
# 
# ## Covid-19 22 January 2020 to 11 April 2020 plotly data analysis
# 
# ###  Data Information
# - Number of rows 25353
# - Number of columns 6
#  - column_Name(Id)   Data_Type(int64) Records(25353)
#  - column_Name(Province_State)   Data_Type(object) Records(10773 )
#  - column_Name(Country_Region)   Data_Type(object) Records(25353)
#  - column_Name(Date)   Data_Type(datetime64[ns]) Records(25353)
#  - column_Name(ConfirmedCases)   Data_Type(float64) Records(25353)
#  - column_Name(Fatalities)   Data_Type(float64) Records(25353)

# ## Necessery Liberary Import

# In[ ]:


import pandas as pd # Load data
import numpy as np # Scientific Computing
import seaborn as sns
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode,plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings # Ignore Warnings
warnings.filterwarnings("ignore")
sns.set() # Set Graphs Background


# ## Load Data

# In[ ]:


data = pd.read_csv('../input/datafileee/train (3).csv')
data.head()


# ## Information Of Data

# In[ ]:


data.info()


# ## Unique Country

# In[ ]:


data['Country_Region'].unique()


# ## Unique country Count

# In[ ]:


data['Country_Region'].nunique()


# - There have 184 Unique Country present

# ## Country Records Count

# In[ ]:


data['Country_Region'].value_counts()


# ## Maximum & Minimum Date 

# In[ ]:


print(data['Date'].min())
print(data['Date'].max())


# - Start date 22 january 2020
# - End date 11 april 2020

# ## Group Country & Select Largest 15 Country by ConfirmedCases

# In[ ]:


data_15 = data.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()
data_15 = data_15.nlargest(15,'ConfirmedCases')
data_15


# ## Create Active/Recover Column

# In[ ]:


data_15['Active/Recover'] = data_15['ConfirmedCases'] - data_15['Fatalities']
data_15


# ## Bar Plot ConfirmedCases For New Data

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['ConfirmedCases']

data1 = go.Bar(x=p_15,y=q_15,name='ConfirmedCases')

layout = go.Layout(title='Country_Region VS ConfirmedCases',xaxis_title='Country_Region',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - The Highest Number of ConfirmedCases in US
# - The lowest Number of ConfirmedCases in Austria

# ## Bar Plot For Fatalities For New Data

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['Fatalities']

data1 = go.Bar(x=p_15,y=q_15)

layout = go.Layout(title='Country_Region VS Deaths',xaxis_title='Country_Region',yaxis_title='Deaths')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - The Highest Number of Deaths in Italy
# - The lowest Number of Deaths in Austria

# ## Bar Plot For Active/Recover For New Data

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['Active/Recover']

data1 = go.Bar(x=p_15,y=q_15)

layout = go.Layout(title='Country_Region VS Active/Recover',
                   xaxis_title='Country_Region',yaxis_title='Active/Recover')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - The Highest Number of Active/Recover in US
# - The lowest Number of Active/Recover in Austria

# ## Country_Region Vs ConfirmedCases & Deaths for 15 Country

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['Active/Recover']
z_15 = data_15['Fatalities']
m_15 = data_15['ConfirmedCases']

data1 = go.Bar(x=p_15,y=m_15,name='ConfirmedCases')
data2 = go.Bar(x=p_15,y=q_15,name='Active/Recover')
data3 = go.Bar(x=p_15,y=z_15,name='Deaths')

layout = go.Layout(barmode='group',title='ConfirmedCases & Active/Recover & Deaths VS Country_Region',
                   xaxis_title='Country_Region',
                   yaxis_title='ConfirmedCases & Active/Recover & Deaths')
data=[data1,data2,data3]
fig = go.Figure(data,layout)

iplot(fig)


# ## Scatter Plot For Country VS ConfirmedCases For New Data

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['ConfirmedCases']

data1 = go.Scatter(x=p_15,y=q_15,mode='markers')

layout = go.Layout(title='Country_Region VS ConfirmedCases',
                   xaxis_title='Country_Region',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - The Highest Number of ConfirmedCases in US.
# - The lowest Number of ConfirmedCases in Austria.

# ## Scatter Plot For Country VS Active/Recover For New Data

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['Active/Recover']

data1 = go.Scatter(x=p_15,y=q_15,mode='markers')

layout = go.Layout(title='Country_Region VS Active/Recover',
                   xaxis_title='Country_Region',yaxis_title='Active/Recover')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - The Highest Number of Active/Recover in US.
# - The lowest Number of Active/Recover in Austria.

# ## Scatter Plot For Country VS Deaths For New Data

# In[ ]:


p_15 = data_15['Country_Region']
q_15 = data_15['Fatalities']

data1 = go.Scatter(x=p_15,y=q_15,mode='markers')

layout = go.Layout(title='Country_Region VS Deaths',
                   xaxis_title='Country_Region',yaxis_title='Deaths')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - The Highest Number of Deaths in Italy.
# - The lowest Number of Deaths in Austria.

# ## Pie Chart For ConfirmedCases For New Data

# In[ ]:


pp = data_15['Country_Region']
qq = data_15['ConfirmedCases']

data = go.Pie(labels=pp,values=qq, 
              hoverinfo='label+percent', textinfo='value')

layout = go.Layout(title='Country_Region VS ConfirmedCases')
data =[data]
fig = go.Figure(data,layout)
iplot(fig)


# - The Highest Number of ConfirmedCases 23% in US & China .
# - The lowest Number of ConfirmedCases 1% in Belgium, Netherlands, Canada & Austria.

# ## Pie Chart For Deaths For New Data

# In[ ]:


pp = data_15['Country_Region']
qq = data_15['Fatalities']

data = go.Pie(labels=pp,values=qq, 
              hoverinfo='label+percent', textinfo='value')

layout = go.Layout(title='Country_Region VS Fatalities')
data =[data]
fig = go.Figure(data,layout)
iplot(fig)


# - The Highest Number of Deaths 25% in Italy.
# - The lowest Number of Deaths 0.2% in Austria.

# ## The Date Convert Into YYYY-MM-DD Format

# In[ ]:


data1 = pd.read_csv('../input/datafileee/train (3).csv')
data1['Date'] = pd.to_datetime(data1['Date'])


# ## Create New Dataset For Individual Date

# In[ ]:


data_81 = data1.groupby('Date', as_index=False)['ConfirmedCases','Fatalities'].sum()
data_81 = data_81.nlargest(81,'ConfirmedCases')
data_81


# ## Create Active/Recover Column for Individual date

# In[ ]:


data_81['Active/Recover'] = data_81['ConfirmedCases'] - data_81['Fatalities']
data_81


# ## Scatter Plot Date VS ConfirmedCases For New Dataset

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['ConfirmedCases']

data1 = go.Scatter(x=p_81,y=q_81)

layout = go.Layout(title='Date VS ConfirmedCases',
                   xaxis_title='Date',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The ConfirmedCases

# ## Scatter Plot Date VS Active/Recover For New Dataset

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['Active/Recover']

data1 = go.Scatter(x=p_81,y=q_81)

layout = go.Layout(title='Date VS Active/Recover',
                   xaxis_title='Date',yaxis_title='Active/Recover')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Active/Recover

# ## Scatter Plot Date VS Deaths For New Dataset

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['Fatalities']

data1 = go.Scatter(x=p_81,y=q_81,mode='markers')

layout = go.Layout(title='Date VS Fatalities',
                   xaxis_title='Date',yaxis_title='Fatalities')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Deaths

# ## ConfirmedCases & Active/Recover & Deaths By Date

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['ConfirmedCases']
z_81 = data_81['Active/Recover']
m_81 = data_81['Fatalities']

trace0 = go.Scatter(x=p_81,y=q_81,mode='markers')
trace1 = go.Scatter(x=p_81,y=z_81,mode='markers')
trace2 = go.Scatter(x=p_81,y=m_81,mode='markers')

layout = go.Layout(title='Total ConfirmedCases & Active/Recover & Deaths By Date',
                   xaxis_title='Date',yaxis_title='ConfirmedCases Active/Recover & Deaths')
data=[trace0,trace1,trace2]
fig = go.Figure(data,layout)

iplot(fig)


# - ConfirmedCases & Active/Recover highly increases after 18th March
# - Deaths are increase after 29th March.

# ## Bar Plot Date VS ConfirmedCases For New Dataset

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['ConfirmedCases']

data1 = go.Bar(x=p_81,y=q_81)

layout = go.Layout(title='Date VS ConfirmedCases',
                   xaxis_title='Date',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The ConfirmedCases.

# ## Bar Plot Date VS Deaths For World

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['Fatalities']

data1 = go.Bar(x=p_81,y=q_81)

layout = go.Layout(title='Date VS Deaths',
                   xaxis_title='Date',yaxis_title='Deaths')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Deaths.

# ## Line Plot Date VS ConfirmedCases For World

# In[ ]:


p_81 = data_81['Date']
q_81 = data_81['ConfirmedCases']

data1 = go.Scatter(x=p_81,y=q_81)

layout = go.Layout(title='Date VS ConfirmedCases',
                   xaxis_title='Date',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The ConfirmedCases.

# ## Create New Dataset For Individual Date & Individual Country

# In[ ]:


data2 = pd.read_csv('../input/datafileee/train (3).csv')
data_all = data2.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
data_all


# ## Create Active/Recover Column for Country &  date

# In[ ]:


data_all['Active/Recover'] = data_all['ConfirmedCases'] - data_all['Fatalities']
data_all


# ## Select All United State Data From New Dataset

# In[ ]:


data_usa = data_all.query("Country_Region=='US'")
data_usa


# ## Scatter Plot Date VS ConfirmedCases For United State

# In[ ]:


p_usa = data_usa['Date']
q_usa = data_usa['ConfirmedCases']

data1 = go.Scatter(x=p_usa,y=q_usa,mode='markers')

layout = go.Layout(title='Total ConfirmedCases By Date For United State',
                   xaxis_title='Date',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Number of ConfirmedCases In US.

# ## Scatter Plot Date VS Deaths For United State

# In[ ]:


p_usa = data_usa['Date']
q_usa = data_usa['Fatalities']

data1 = go.Scatter(x=p_usa,y=q_usa,mode='markers')

layout = go.Layout(title='Total Fatalities By Date For United State',
                   xaxis_title='Date',yaxis_title='Fatalities')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Number of deaths In US.

# ## ConfirmedCases & Active/Recover & Deaths By Date For United State

# In[ ]:


p_usa = data_usa['Date']
q_usa = data_usa['ConfirmedCases']
z_usa = data_usa['Active/Recover']
m_usa = data_usa['Fatalities']

trace0 = go.Scatter(x=p_usa,y=q_usa,mode='markers')
trace1 = go.Scatter(x=p_usa,y=z_usa,mode='markers')
trace2 = go.Scatter(x=p_usa,y=m_usa,mode='markers')

layout = go.Layout(title='Total ConfirmedCases & Active/Recover & Deaths By Date',
                   xaxis_title='Date',yaxis_title='ConfirmedCases Active/Recover & Deaths')
data=[trace0,trace1,trace2]
fig = go.Figure(data,layout)

iplot(fig)


# - ConfirmedCases & Active/Recover highly increases
# - Deaths are increase after 9th April.

# ## Bar Plot Date VS ConfirmedCases For United State

# In[ ]:


p_usa = data_usa['Date']
q_usa = data_usa['ConfirmedCases']

data1 = go.Bar(x=p_usa,y=q_usa)

layout = go.Layout(title='Date VS ConfirmedCases US',
                   xaxis_title='Date',yaxis_title='ConfirmedCases')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Number of ConfirmedCases In US.

# ## Bar Plot Date VS Deaths For United State

# In[ ]:


p_usa = data_usa['Date']
q_usa = data_usa['Fatalities']

data1 = go.Bar(x=p_usa,y=q_usa)

layout = go.Layout(title='Date VS Deaths US',
                   xaxis_title='Date',yaxis_title='Deaths')
data=[data1]
fig = go.Figure(data,layout)

iplot(fig)


# - Day By Day Increase The Number of Deaths In US.
