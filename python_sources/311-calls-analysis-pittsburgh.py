#!/usr/bin/env python
# coding: utf-8

# # What are 311 calls ?
# 
# 311 is a non-emergency phone number that people can call in many cities to find information about services, make complaints, or report problems like graffiti or road damage. Even in cities where a different phone number is used, 311 is the generally recognized moniker for non-emergency phone systems.
# 
# Since its inception, 311 has evolved with technological advances into a multi-channel service that connects citizen with government, while also providing a wealth of data that improves how cities are run. 

# ## INDEX:
# 
# 
# [1. Peek into the Data.](#1)
# 
# 
# [2. Distributions:](#2)
# - Distribution of requests amongst request origins.
# - Distribution of requests amongst Departments.
# - Distribution of requests amongst council Districts.
# - Distribution of requests amongst Wards.
# - Distribution of requests amongst police zones.
# - Distribution of requests amongst public work divisions.
# - Distribution of requests amongst PLI divisions.
# - Distribution of requests amongst ploice zones.
# - Distribution of requests amongst fire zones.
# 
# 
# [3. Potholes, Weed...](#3)
# 
# 
# [4. Request Status](#4)
# 
# 
# [5. Request Status of top reported Request Types](#5)
# 
# 
# [6. Trend of 311 calls (by week)](#6)
# 
# 
# [7. Month, Day, Hour Distribution :](#7)
# - Requests distribution amongst Months.
# - Request Distribution amongst Day of Month.
# - Requests Distribution amongst Weekdays.
# - Request Distriution amongst Hour of Day.
# 
# 
# [8. Monthly trend in Top 20 Request Type](#8)
# 
# 
# [9. Time-Series Analys](#9)
# - Distribution of requests amongst Request Origins Throught time.
# - Distribution of requests amongst top 20 Request Types Throught time.

# In[ ]:


## importing libraries:
import numpy as np # Linear Algebra
import pandas as pd # To work with data
from plotly.offline import init_notebook_mode, iplot # Offline visualizations
import plotly.graph_objects as go # Visualizations
import plotly.express as px # Visualizations
import matplotlib.pyplot as plt # Just in case
from plotly.subplots import make_subplots # Subplots in plotly
from wordcloud import WordCloud, STOPWORDS # WordCloud


# In[ ]:


df = pd.read_csv("../input/311-service-requests-pitt/ServreqPitt.csv") # Loading the Data


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum() # Too many null values


# In[ ]:


df.dropna(inplace=True)


# <a class='anchor' id='1'></a>
# ## Peek into the Data

# In[ ]:


df.head() # a look into the dataset.


# In[ ]:


# Converting the attribute into datetime object will help in many ways.
df.loc[:,'CREATED_ON'] = pd.to_datetime(df['CREATED_ON']) 

df['Month'] = df['CREATED_ON'].dt.month_name()
df['Day'] = df['CREATED_ON'].dt.day
df['Hour'] = df['CREATED_ON'].dt.hour
df['Weekday'] = df['CREATED_ON'].dt.weekday_name


# In[ ]:


## Let's have a look at the unique value that each attribute has.
for i in df.columns :
    print(i,':', len(df[i].unique()))


# <a class='anchor' id='2'></a>
# ## Distrinbutions

# In[ ]:


temp = df['REQUEST_ORIGIN'].value_counts().reset_index()
temp.columns=['Request_Origin', 'Count']

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "xy"},{"type": "domain"}]]
)

fig.add_trace(go.Bar(x=temp['Request_Origin'], y=temp['Count']), 1,1)
fig.add_trace(go.Pie(labels=temp['Request_Origin'], values=temp['Count'], pull=[0.1,0,0,0,0,0,0,0]), 1,2)
fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title='Distribution of requests amongst Request Origins:')
iplot(fig)


# ### Insights :
# - More than half of the total requests are reported on call centres. 
# - Request made on QAlert Mobile IOS, Government Website and over Emails are less than 1% of total requests. So, they can consider stopping there services.

# In[ ]:


temp = df['DEPARTMENT'].value_counts().reset_index()
temp.columns=['Department', 'Count']

fig = px.bar(temp, 'Department', 'Count')
fig.update_layout(title='Number of requests from departments:', xaxis_tickangle=-25)
iplot(fig)


# In[ ]:


temp = df['COUNCIL_DISTRICT'].value_counts().reset_index().head(20)
temp.columns=['Council_District', 'Count']

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])
fig.add_trace(go.Bar(x=temp['Council_District'], y=temp['Count']), 1,1)
fig.add_trace(go.Pie(labels=temp['Council_District'], values=temp['Count']), 1,2)
fig.update_traces(textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title='Distribution of requests amongst council districts:')
iplot(fig)


# In[ ]:


temp = df['WARD'].value_counts().reset_index()
temp.columns=['Ward', 'Count']

fig = px.bar(temp, 'Ward', 'Count', color='Count')
fig.update_layout(title='Request distributions amongst Wards:', xaxis_tickangle=-25)
iplot(fig)


# In[ ]:


temp = df['PUBLIC_WORKS_DIVISION'].value_counts().reset_index().head(20)
temp.columns=['Public Work Division', 'Count']

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])
fig.add_trace(go.Bar(x=temp['Public Work Division'], y=temp['Count']), 1,1)
fig.add_trace(go.Pie(labels=temp['Public Work Division'], values=temp['Count']), 1,2)
fig.update_traces(textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title='Distribution of requests amongst Public Work Divisions:')
iplot(fig)


# In[ ]:


temp = df['POLICE_ZONE'].value_counts().reset_index().head(20)
temp.columns=['Police Zone', 'Count']

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])
fig.add_trace(go.Bar(x=temp['Police Zone'], y=temp['Count']), 1,1)
fig.add_trace(go.Pie(labels=temp['Police Zone'], values=temp['Count'], textinfo='label+percent'), 1,2)
fig.update_traces(textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title='Distribution of requests amongst Police Zones:')
iplot(fig)


# In[ ]:


temp = df['PUBLIC_WORKS_DIVISION'].value_counts().reset_index()
temp.columns=['Public Work Division', 'Count']
fig = go.Figure(data=[
    go.Bar(x=temp['Count'], y=temp['Public Work Division'],
           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),
          orientation='h'
          )
])
fig.update_layout(title='Distrbution of request amongst Public Work Divisions')
iplot(fig)


# In[ ]:


temp = df['PLI_DIVISION'].value_counts().reset_index()
temp.columns=['PLI_DIVISION', 'Count']
fig = go.Figure(data=[
    go.Bar(x=temp['Count'], y=temp['PLI_DIVISION'],
           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),
          orientation='h'
          )
])
fig.update_layout(title='Distrbution of request amongst PLI divisions')
iplot(fig)


# In[ ]:


temp = df['POLICE_ZONE'].value_counts().reset_index()
temp.columns=['Police Zone', 'Count']
fig = go.Figure(data=[
    go.Bar(x=temp['Count'], y=temp['Police Zone'],
           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),
          orientation='h'
          )
])
fig.update_layout(title='Distrbution of request amongst Police Zone')
iplot(fig)


# In[ ]:


temp = df['FIRE_ZONE'].value_counts().reset_index().head(20)
temp
temp.columns=['Fire Zone', 'Count']
fig = go.Figure(data=[
    go.Bar(x=temp['Count'], y=temp['Fire Zone'],
           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),
          orientation='h'
          )
])
fig.update_layout(title='Fire Zones that recieve most requests.')
iplot(fig)


# <a class='anchor' id='3'></a>
# ## Potholes, Weed...

# In[ ]:


temp = df['REQUEST_TYPE'].value_counts().reset_index()
temp.columns=['Request_Type', 'Count']

fig = px.bar(temp.head(20), 'Request_Type', 'Count')
fig.update_layout(title='Most reported Service Requests:')
iplot(fig)


# <a class='anchor' id='4'></a>
# ## Request Status

# In[ ]:


temp = df['STATUS'].value_counts().reset_index()
temp.columns=['Status', 'Count']
fig = go.Figure(data=[
    go.Pie(labels=temp['Status'], values=temp['Count'])
])
fig.update_traces(textfont_size=20, marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title='Distribution of request status:')
iplot(fig)


# <a class='anchor' id='5'></a>
# ## Request Status of top reported Request Types

# In[ ]:


temp = df['REQUEST_TYPE'].value_counts().reset_index()
temp.columns=['Request_Type', 'Count']
top_requests = temp.head(20)['Request_Type'].tolist()
temp=df[df['REQUEST_TYPE'].isin(top_requests)]

temp=temp.groupby(by=['REQUEST_TYPE', 'STATUS'])['REQUEST_ID'].count()
temp = temp.unstack().fillna(0).reset_index()
temp.columns=['Request Type','0','1','3']
temp = pd.melt(temp, id_vars='Request Type', value_vars=['0','1','3'])
fig = px.bar(temp, 'Request Type', 'value', color='variable')
fig.update_layout(title='Request Status of top reported Request Types')
iplot(fig)


# <a class='anchor' id='6'></a>
# ## Trend of 311 calls (by week)

# In[ ]:


temp = df.set_index('CREATED_ON').sort_index()
temp = temp.resample('W')['REQUEST_ID'].count().reset_index()
fig = px.line(temp, 'CREATED_ON', 'REQUEST_ID')
fig.update_layout(title = 'Trend of 311 calls. (By Week)')
iplot(fig)


# <a class='anchor' id='7'></a>
# ## Month, Day, Hour Distribution

# In[ ]:


temp = df['Month'].value_counts().reset_index()
temp.columns=['Month', 'Count']
temp.sort_values(by='Count', inplace=True)
fig = px.scatter(temp, 'Month', 'Count', size='Count', color='Count')
fig.update_layout(title='Requests by months:')
iplot(fig)


# ### Insights:
# - May, June, July and August have highest number of complaints recorded. That is summers.
# - December has the lowest recorded service requests. That can be because of the Chritsmas and New Year celebrations.

# In[ ]:


temp = df['Day'].value_counts().reset_index()
fig = px.scatter(temp, 'index', 'Day', color='Day', size='Day',
          labels={'index':'Day', 'Day':'Requests'})
fig.update_layout(title='Requests by day of month')
iplot(fig)


# ### Insights :
# - Day '31' has least requests registerd.That is because not every month has '31'st day in hem. So, this gives no idea about distribution.
# - After '15'th day of any month, we can see a downward trend in count of registered requests.
# - This means that people tend to register more requests in the first half of the month than the next half.

# In[ ]:


temp = df['Weekday'].value_counts().reset_index()
fig = px.scatter(temp, 'index', 'Weekday', color='Weekday', size='Weekday',
          labels={'index':'Weekday', 'Weekday':'Requests'})
fig.update_layout(title='Requests by Weekday')


# ### Insights :
# - We see a downward trend with the weekdays and number of reported requests.
# - People register significantly less complains on weekend than weekdays. Requests reported on weekends are around 20% of the requests reported on weekdays.
# - Saturday is the least busy day as people report least requests on Saturday.

# In[ ]:


temp = df['Hour'].value_counts().reset_index()
fig = px.scatter(temp, 'index', 'Hour', color='Hour', size='Hour',
          labels={'index':'Hour', 'Hour':'Requests'})
fig.update_layout(title='Requests by hour of day')
iplot(fig)


# ### Insights:
# - We have two major trends: 1.Towards midnight. 2.Towards Noon.
# - Number of requests go up as the day begins till the noon.
# - After noon, the number of requests go down as the passes by the evening and reaches midnight.

# <a class='anchor' id='8'></a>
# ## Monthly trend in top 20 Request Type

# In[ ]:


temp = df['REQUEST_TYPE'].value_counts().reset_index()
top_types = temp.head(20)['index'].tolist()
del temp
df1 = df[df['REQUEST_TYPE'].isin(top_types)]
df1 = df1.groupby(by=['Month', 'REQUEST_TYPE'])['REQUEST_ID'].count().unstack().reset_index()
vars_list = list(df1.columns)[1:]
df1 = pd.melt(df1, id_vars='Month', value_vars=vars_list)
df1.columns=['Month','Request Type', 'Requests']

fig = px.scatter(df1, x='Request Type', y='Requests', color='Month')
fig.update_layout(title='Distribution of Requests of top 20 most reported Tequest Types :')
iplot(fig)


# ### Insights:
# - Pothole requests are maximum in the months of Feb, April and May. and Least in the month of December
# - Snow/Ice removal requests are maximum in the months of January, February and December. Which is winters in Pitsburh. So, this distribution justifies the request type.
# - Weed/Debris requests are also highest in the months of May, Jun, Jul and Aug.

# <a class='anchor' id='9'></a>
# ## Time Series Analysis

# In[ ]:


df1 = df.set_index('CREATED_ON')
df1 = df1[['REQUEST_ORIGIN']]
df1 = pd.get_dummies(df1)
df1 = df1.resample('M').sum()
df1 = df1.cumsum()
df1.reset_index(inplace=True)
df1 = pd.melt(df1, id_vars=['CREATED_ON'], value_vars=['REQUEST_ORIGIN_Call Center',
 'REQUEST_ORIGIN_Control Panel',
 'REQUEST_ORIGIN_Email',
 'REQUEST_ORIGIN_QAlert Mobile iOS',
 'REQUEST_ORIGIN_Report2Gov Android',
 'REQUEST_ORIGIN_Report2Gov Website',
 'REQUEST_ORIGIN_Report2Gov iOS',
 'REQUEST_ORIGIN_Text Message',
 'REQUEST_ORIGIN_Twitter',
 'REQUEST_ORIGIN_Website'])
df1.columns = ['Created_On', 'Request_Origin', 'Requests']
df1.loc[:,'Request_Origin'] = df1['Request_Origin'].apply(lambda x : str(x).split('REQUEST_ORIGIN_')[1])
df1.loc[:,'Created_On'] = df1.loc[:,'Created_On'].dt.strftime('%Y-%m-%d')

fig = px.bar(df1, 'Request_Origin', 'Requests', animation_frame='Created_On')
fig.update_layout(title='Requests distribution amongst Request Origins throughout time:')
iplot(fig)


# #### Looking at the above animation, we can see that Email and QAlert on IOS mobile had never been populer choice amongst users to report requests. These services shoul've had stopped already.

# In[ ]:


temp = df['REQUEST_TYPE'].value_counts().reset_index()
top_types = temp.head(20)['index'].tolist()
del temp
df1 = df.set_index('CREATED_ON')
df1 = df1[df1['REQUEST_TYPE'].isin(top_types)]['REQUEST_TYPE']
df1 = pd.get_dummies(df1)
df1 = df1.resample('M').sum()
df1 = df1.cumsum()
df1.reset_index(inplace=True)
df1 = pd.melt(df1, id_vars='CREATED_ON', value_vars=top_types)
df1.columns = ['Created_On', 'Request_Type', 'Requests']
df1.loc[:,'Created_On'] = df1.loc[:,'Created_On'].dt.strftime('%Y-%m-%d')

fig = px.bar(df1, 'Request_Type', 'Requests', animation_frame='Created_On')
fig.update_layout(title='Requests distribution amongst most reported Request Types throughout time:')
iplot(fig)


# ### Insights:
# - Potholes requests has been reported most since the very beginning.
# - Weed requests were not very popular before 2016. It was June 2016 when it exceeded all other requests(except Potholes) and became 2nd most reported request.
# - There were almost no(total 9) reported requests about Snow/Ice removal before 2016.
# - There were no requests regarding Referral before may 2016. First request regarding Referrals was reported in june 2016.

# In[ ]:




