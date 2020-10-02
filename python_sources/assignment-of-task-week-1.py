#!/usr/bin/env python
# coding: utf-8

# Task 1 & Task 2 are both accomodated in this Notebook

# In[ ]:


# storing & analysis
import pandas as pd
import numpy as np
import ppscore as pps # pip install ppscore

# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# # Exploratory Data Analysis

# In[ ]:


# Loading the dataset
df = pd.read_csv("../input/adeconvid19/train.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.nunique()


# In[ ]:


df.isnull().sum()


# * Null values are only to be found in the Province_State collumn

# In[ ]:


df['Date'].unique()


# * Timespan of the dataset is from the Month of January to March

# In[ ]:


df['Country_Region'].unique()


# In[ ]:


# Predictive Power Score
df1 = df.drop(['Id'], axis=1)
test_data = pps.matrix(df1)
sns.heatmap(test_data, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)


# * Correlation only defined for numeric columns, so Predictive Power Score is used to establish more meaningingful relation between the dataset features
# * Fatalities is a strong predictor for the ConfirmedCases (0.57 PPS). However ConfirmedCases only predicts the Fatalities with a PPS of 0.4

# # Time-Series-Analysis of ConfirmedCases & Fatalities

# In[ ]:


# Cases Over Time
temp1 = df.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()
temp1 = temp1.melt(id_vars="Date", value_vars=['ConfirmedCases','Fatalities'],
                 var_name='Case', value_name='Count')
temp1.head()


fig = px.area(temp1, x="Date", y="Count", color='Case',
             title='Cases over time')
fig.update_layout(margin=dict(t=80,l=0,r=0,b=0))
fig


# In[ ]:


# Mortality Rate Over Time
temp2 = df.groupby('Date').sum().reset_index()

temp2['No. of Deaths to 100 Confirmed Cases'] = round(temp2['Fatalities']/temp2['ConfirmedCases'], 3)*100

temp2 = temp2.melt(id_vars='Date', value_vars=['No. of Deaths to 100 Confirmed Cases'],var_name='Ratio',
                  value_name='Value')

fig = px.line(temp2, x="Date", y="Value", color='Ratio', log_y=True, title='Mortality Rate Over The Time')
fig.update_layout(legend=dict(orientation="h", y=1, x=0, xanchor="left", yanchor="top"),
                  margin=dict(t=80,l=0,r=0,b=0))
fig


# # Cases & Fatalities Across the Countries

# In[ ]:


# ConfirmedCases & Fatalities Group by Latest Date
latest_date = df[df['Date'] == max(df['Date'])].reset_index()
latest_date.head()


# In[ ]:


# ConfirmedCases & Fatalities Group by Country on the Latest Date
Country_latest_date = latest_date.groupby('Country_Region')['ConfirmedCases','Fatalities'].sum().reset_index()
Country_latest_date.head()


# In[ ]:


# Top 10 Countries proportion to the COVID-19 cases
data_cases = Country_latest_date.sort_values('ConfirmedCases', ascending=False).head(10)
sizes = data_cases['ConfirmedCases']
labels = data_cases['Country_Region']
explode = [0.1,0,0,0,0,0,0,0,0,0.3]

plt.figure(figsize = (7,7))
plt.pie(sizes, labels=labels, explode=explode,autopct='%1.1f%%', shadow=True,startangle=90, radius=1, 
        labeldistance=1.1,pctdistance=0.7,frame=False)

plt.axis('equal')
plt.show()


# In[ ]:


# Top 20 Countries With the Highest Confirm Casses 
fig = px.bar(Country_latest_date.sort_values('ConfirmedCases', ascending=False).head(20).sort_values('ConfirmedCases', ascending=True), 
             x="ConfirmedCases", y="Country_Region", title='Confirmed Cases Per Country', text='ConfirmedCases',
             orientation='h')
fig.update_traces(textposition='outside')
fig.update_layout(margin=dict(t=80,l=0,r=0,b=0))
fig


# In[ ]:


# Top 20 Countries With the Highest Fatalities
fig = px.bar(Country_latest_date.sort_values('Fatalities', ascending=False).head(20).sort_values('Fatalities', ascending=True), 
             x="Fatalities", y="Country_Region", title='Fatalities Per Country', text='Fatalities',
             orientation='h')
fig.update_traces(textposition='outside')
fig.update_layout(margin=dict(t=80,l=0,r=0,b=0))
fig


# In[ ]:


sns.pairplot(Country_latest_date.sort_values('ConfirmedCases', ascending=False).head(20).sort_values('ConfirmedCases', ascending=True),
             hue='Country_Region', size=3);


#  # Motality Rate Trend Across the Countries

# The following trend shows the motality rate across the countres with the highest ConfirmedCases and Fatalitites

# In[ ]:


plt.figure(figsize=(25,25))
sns.relplot(x="ConfirmedCases",y="Fatalities", hue='Country_Region',kind="line", height=10,
            data=df.sort_values('ConfirmedCases', ascending=False).head(500).sort_values('ConfirmedCases', ascending=True))


# In[ ]:


# Time-Series-Analysis for the Provinces of Each Country
df_grouped_date1 = df.groupby(['Country_Region','Date','Province_State'])['ConfirmedCases','Fatalities'].sum().reset_index()
df_grouped_date2 = df.groupby(['Country_Region','Province_State','Date'])['ConfirmedCases','Fatalities'].sum().reset_index()


# # Infected Rate Across the Countries (in their local Epicenter)

# In[ ]:



df_grouped_date3 = df.groupby(['Country_Region','Province_State'])['ConfirmedCases','Fatalities'].max().reset_index()
df_grouped_date3.head()

fig = px.pie(df_grouped_date3.sort_values('ConfirmedCases', ascending=False), values='ConfirmedCases', names='Country_Region',
             title='Infected rates across countries',
              hover_data=['Province_State'], labels={'Province_State':'Province_State'})
fig.update_traces(textposition='inside')
fig.show()


# In[ ]:


World_latest = latest_date[latest_date['Country_Region']!='US']
df_grouped_date4 = World_latest.groupby(['Country_Region','Province_State'])['ConfirmedCases','Fatalities'].max().reset_index()
df_grouped_date4.head()

fig = px.pie(df_grouped_date4.sort_values('ConfirmedCases', ascending=False), values='ConfirmedCases', names='Country_Region',
             title='Infected rates Across Countries',
              hover_data=['Province_State'], labels={'Province_State':'Province_State'})
fig.update_traces(textposition='inside')
fig.show()


# In[ ]:


US_latest = latest_date[latest_date['Country_Region']=='US']
US_latest.head()
US_latest_grouped = US_latest.groupby('Province_State')['ConfirmedCases','Fatalities'].sum().reset_index()
US_latest_grouped.head()


# In[ ]:


# Infected Rate Across the States of US
fig = px.pie(US_latest_grouped.sort_values('ConfirmedCases', ascending=False), values='ConfirmedCases', names='Province_State',
             title='Infected rates across Province State of US')
fig.update_traces(textposition='inside')
fig.show()


# * Major Propotion of Cases are in New York around 34.4% but Many Province States are affected by the COVID-19 Cases in US

# In[ ]:


# Fatality Rate Across the States of US
fig = px.pie(US_latest_grouped.sort_values('Fatalities', ascending=False), values='Fatalities', names='Province_State',
             title='Fatality rates across Province State of US')
fig.update_traces(textposition='inside')
fig.show()


# In[ ]:


China_latest = latest_date[latest_date['Country_Region']=='China']
China_latest.head()
China_latest_grouped = China_latest.groupby('Province_State')['ConfirmedCases','Fatalities'].sum().reset_index()
China_latest_grouped.head()

# Infected Rate Across the States of US
fig = px.pie(China_latest_grouped.sort_values('ConfirmedCases', ascending=False), values='ConfirmedCases', names='Province_State',
             title='Infected rates across Province State of China')
fig.update_traces(textposition='inside')
fig.show()


# * Most of the cases were saturated **(81.7)** in single Province State of Hubei in China unlike US.

# In[ ]:


US_df = df.groupby(['Date', 'Country_Region'])['ConfirmedCases','Fatalities'].max().reset_index()

temp4 = US_df[US_df['Country_Region']=='US'].reset_index()
temp4 = temp4.melt(id_vars='Date', value_vars=['ConfirmedCases','Fatalities'],
                var_name='Case', value_name='Count')
fig = px.bar(temp4, x="Date", y="Count", color='Case', facet_col="Case",
            title='US Cases Over Time')
fig.update_layout(margin=dict(t=80,l=0,r=0,b=0))
fig


# * Slope of ConfirmedCases is still linearly increasing in US 
# * Slope of Fatalities rate is increasing with the decreasing rate
# 
# **Observation:**
# 
# While a steady growth in the virus spread can be observed in US, a rapid increase in number of cases within the past few days can be seen in some other countries.

# In[ ]:


China_df = df.groupby(['Date', 'Country_Region'])['ConfirmedCases','Fatalities'].max().reset_index()

temp5 = China_df[China_df['Country_Region']=='China'].reset_index()
temp5 = temp5.melt(id_vars='Date', value_vars=['ConfirmedCases','Fatalities'],
                var_name='Case', value_name='Count')
fig = px.bar(temp5, x="Date", y="Count", color='Case', facet_col="Case",
            title='China Cases Over Time')
fig.update_layout(margin=dict(t=80,l=0,r=0,b=0))
fig


# * Slope of ConfirmedCases is nearly flat in the China 
# * Slope of Fatalities rate is Stable and not increasing anymore in China
# 
# **Observation:**
# 
# The growth in the virus spread can be observed in China has come to a halt as it cab been from the dataset, the number cases in the month of March remains same.
