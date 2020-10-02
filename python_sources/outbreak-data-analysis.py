#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'> Corona Virus OutBreak Analyis</h1>
# 
# 

# # Import Libraries

# In[ ]:


# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium


# # Import Dataset

# In[ ]:


conf_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recv_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')


# In[ ]:


conf_df.head()


# # Data Wrangling

# In[ ]:


dates = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', 
         '1/29/20', '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', 
         '2/5/20', '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', 
         '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',
         '2/20/20','2/21/20','2/22/20','2/23/20','2/24/20','2/25/20','2/26/20',
'2/27/20','2/28/20','2/29/20','3/1/20','3/2/20','3/3/20','3/4/20','3/5/20','3/6/20',
'3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20','3/13/20','3/14/20','3/15/20',
'3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20','3/23/20','3/24/20','3/25/20','3/26/20']

conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Confirmed')

deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Deaths')

recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
                            value_vars=dates, var_name='Date', value_name='Recovered')

full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 
                       axis=1, sort=False)
full_table.head()


# # Data Cleaning and Preprocessing

# In[ ]:


# converting to proper data format
full_table['Date'] = pd.to_datetime(full_table['Date'])
full_table['Recovered'] = full_table['Recovered'].astype('float')

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values with 0 in columns ('Confirmed', 'Deaths', 'Recovered')
full_table[['Confirmed', 'Deaths', 'Recovered']] = full_table[['Confirmed', 'Deaths', 'Recovered']].fillna(0)
full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')

# cases in the Diamond Princess cruise ship
ship = full_table[full_table['Province/State']=='Diamond Princess cruise ship']

# full table
full_table = full_table[full_table['Province/State']!='Diamond Princess cruise ship']
full_table.head()


# In[ ]:


# derived dataframes
china = full_table[full_table['Country/Region']=='China']
row = full_table[full_table['Country/Region']!='China']

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
row_latest = full_latest[full_latest['Country/Region']!='China']

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()


# # EDA

# ## Current Situation

# In[ ]:


temp = full_latest.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()
temp.style.background_gradient(cmap='Pastel1_r')


# ## Top 10 Countries with most no. of reported cases

# In[ ]:


temp_f = full_latest_grouped[['Country/Region', 'Confirmed']]
temp_f = temp_f.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.head(10).style.background_gradient(cmap='Pastel1_r')


# * Massive number of cases are reported in Mainland China Compared to reset of the world
# * The next few countries are infact are the neighbours of China

# ## Countries with deaths reported

# In[ ]:


temp_flg = full_latest_grouped[['Country/Region', 'Deaths']]
temp_flg = temp_flg.sort_values(by='Deaths', ascending=False)
temp_flg = temp_flg.reset_index(drop=True)
temp_flg = temp_flg[temp_flg['Deaths']>0]
temp_flg.style.background_gradient(cmap='Pastel1_r')


# * Outside China, there has been a lot of deaths due to COVID-19 has reported particularly in Italy and Spain

# ## Most Recent Stats

# In[ ]:


full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
full_table.head()


# In[ ]:


# Defining COVID-19 cases as per classifications 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Defining Active Case: Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# Renaming Mainland china as China in the data table
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[cases] = full_table[cases].fillna(0)

# cases in the ships
ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]

# china and the row
china = full_table[full_table['Country/Region']=='China']
row = full_table[full_table['Country/Region']!='China']

# latest
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
row_latest = full_latest[full_latest['Country/Region']!='China']

# latest condensed
full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
#Step 3: Creating a consolidated table , which gives the country wise total defined cases

temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Pastel1')


# * There are more recovered cases than deaths at this point of time

# ## Hubei - China - World

# In[ ]:


get_ipython().run_line_magic('pinfo', 'fig.update_traces')


# In[ ]:


def location(row):
    if row['Country/Region']=='China':
        if row['Province/State']=='Hubei':
            return 'Hubei'
        else:
            return 'Other Chinese Provinces'
    else:
        return 'Rest of the World'

temp = full_latest.copy()
temp['Region'] = temp.apply(location, axis=1)
temp = temp.groupby('Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
temp = temp.melt(id_vars='Region', value_vars=['Confirmed', 'Deaths', 'Recovered'], 
                 var_name='Case', value_name='Count').sort_values('Count')
temp.head()

fig = px.bar(temp, y='Region', x='Count', color='Case', barmode='group', orientation='h',
             height=500, width=1000, text='Count', title='Hubei - China - World', 
             color_discrete_sequence= ['#EF553B', '#00CC96', '#636EFA'])
fig.update_traces(textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# ## Count of Cases

# In[ ]:


# Reading the dataset
data= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.head()


# In[ ]:


# Let's look at the various columns
data.info()


# In[ ]:


data.describe()


# In[ ]:


# Convert Last Update column to datetime64 format

data['ObservationDate'] = data['ObservationDate'].apply(pd.to_datetime)
data.drop(['SNo'],axis=1,inplace=True)

#Set Date column as the index column.
#data.set_index('Last Update', inplace=True)
data.head()


# ## Countries which have been affected by the Coronavirus(2019-nCoV)till now

# In[ ]:


countries = data['Country/Region'].unique().tolist()
print(countries)

print("\nTotal countries affected by virus: ",len(countries))


# In[ ]:


#Combining China and Mainland China cases

data['Country/Region'].replace({'Mainland China':'China'},inplace=True)
countries = data['Country/Region'].unique().tolist()
print(countries)
print("\nTotal countries affected by virus: ",len(countries))


# ## Current status worldwide

# In[ ]:


# Creating a dataframe with total no of confirmed cases for every country
Number_of_countries = len(data['Country/Region'].value_counts())


cases = pd.DataFrame(data.groupby('Country/Region')['Confirmed'].sum())
cases['Country/Region'] = cases.index
cases.index=np.arange(1,Number_of_countries+1)

global_cases = cases[['Country/Region','Confirmed']]
#global_cases.sort_values(by=['Confirmed'],ascending=False)
global_cases


# ## A Closer look at China's condition

# In[ ]:


#Mainland China
China = data[data['Country/Region']=='China']
China


# In[ ]:


##Let's look at the Confirmed vs Recovered figures of Provinces of China other than Hubei
f, ax = plt.subplots(figsize=(16, 12))

sns.set_color_codes("dark")
sns.barplot(x="Confirmed", y="Province/State", data=China[1:],
            label="Confirmed", color="r")

sns.set_color_codes("deep")
sns.barplot(x="Recovered", y="Province/State", data=China[1:],
            label="Recovered", color="g")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 400), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)


# ## Coorelation between different attributes

# In[ ]:


#Get Correlation between different variables
corr = data.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)


# In[ ]:


'''A Function To Plot Pie Plot using Plotly'''

def pie_plot(cnt_srs, colors, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.7,
                   showlegend=True,
                   marker=dict(colors=colors,
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace


# In[ ]:


data.head()


# In[ ]:


'''Plotly visualization .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook


# ## State

# In[ ]:


py.iplot([pie_plot(data['Province/State'].value_counts(), ['cyan', 'gold'], 'State')])


# ## Country

# In[ ]:


py.iplot([pie_plot(data['Country/Region'].value_counts(), ['cyan', 'gold'], 'Country')])


# ## Confirmed Cases

# In[ ]:


py.iplot([pie_plot(data['Confirmed'].value_counts(), ['cyan', 'gold'], 'Confirmed')])


# ## Death Cases

# In[ ]:


py.iplot([pie_plot(data['Deaths'].value_counts(), ['cyan', 'gold'], 'Deaths')])


# ## Recovered Cases

# In[ ]:


py.iplot([pie_plot(data['Recovered'].value_counts(), ['cyan', 'gold'], 'Recovered')])


# In[ ]:


# Location
sns.countplot(data['Country/Region'])
sns.countplot(data['Country/Region']).set_xticklabels(sns.countplot(data['Country/Region']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(23,8)
plt.title('Location')


# # Global Spread of the Coronavirus Over Time 

# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')


# In[ ]:


from datetime import date
data['ObservationDate'] = data['ObservationDate'].dt.date
spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
#fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))

trace1 = go.Scatter(
                x=spread_gl['ObservationDate'],
                y=spread_gl['Confirmed'],
                name="Confirmed",
                line_color='orange',
                opacity=0.9)
data1 = [trace1];
layout = dict(title = '<b>Confirmed</b>',
              xaxis= dict(title= 'Date',ticklen= 10,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data1, layout = layout)
iplot(fig)


# In[ ]:


trace2 = go.Scatter(
                x=spread_gl['ObservationDate'],
                y=spread_gl['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.9)
data2 = [trace2];
layout = dict(title = '<b>Deaths</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data2, layout = layout)
iplot(fig)


# In[ ]:


trace3 = go.Scatter(
                x=spread_gl['ObservationDate'],
                y=spread_gl['Recovered'],
                name="Recovered",
                line_color='green',
                opacity=0.9)
data3 = [trace3];
layout = dict(title = '<b>Recovered</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data3, layout = layout)
iplot(fig)


# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')


# In[ ]:


data['ObservationDate'] = data['ObservationDate'].dt.date
spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))


china_data = spread[spread['Country/Region']=='China']
date_con_ch = china_data.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum().reset_index()


# In[ ]:


spread


# In[ ]:


china_data


# In[ ]:


date_con_ch


# ## Spread of the Coronavirus Over Time In China

# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')


# In[ ]:


data['ObservationDate'] = data['ObservationDate'].dt.date
spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
#fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))

trace4 = go.Scatter(
                x=date_con_ch['ObservationDate'],
                y=date_con_ch['Confirmed'],
                name="Confirmed",
                line_color='orange',
                opacity=0.8)
data4 = [trace4];
layout = dict(title = '<b>Confirmed</b>',
              xaxis= dict(title= 'Date',ticklen= 10,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data4, layout = layout)
iplot(fig)


# In[ ]:


trace5 = go.Scatter(
                x=date_con_ch['ObservationDate'],
                y=date_con_ch['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8)
data5 = [trace5];
layout = dict(title = '<b>Recovered</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data5, layout = layout)
iplot(fig)


# In[ ]:


trace6 = go.Scatter(
                x=date_con_ch['ObservationDate'],
                y=date_con_ch['Recovered'],
                name="Recovered",
                line_color='green',
                opacity=0.8)
data6 = [trace6];
layout = dict(title = '<b>Recovered</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data6, layout = layout)
iplot(fig)


# ### Please follow this repository to follow these project: https://github.com/chiragsamal/CoronaVirus-Outbreak-Analysis

# ## End of the Notebook

# 

# ## Last Update 27th March 2020

# ## Spread of the Coronavirus Over Time In Italy

# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')


# In[ ]:


data['ObservationDate'] = data['ObservationDate'].dt.date
spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))


italy_data = spread[spread['Country/Region']=='Italy']
date_con_ch1 = italy_data.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum().reset_index()


# In[ ]:


italy_data


# In[ ]:


date_con_ch1


# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')


# In[ ]:


data['ObservationDate'] = data['ObservationDate'].dt.date
#spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
#spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
#fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))

trace7 = go.Scatter(
                x=date_con_ch1['ObservationDate'],
                y=date_con_ch1['Confirmed'],
                name="Confirmed",
                line_color='orange',
                opacity=0.8)
data7 = [trace7];
layout = dict(title = '<b>Confirmed</b>',
              xaxis= dict(title= 'Date',ticklen= 10,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data7, layout = layout)
iplot(fig)


# In[ ]:


trace8 = go.Scatter(
                x=date_con_ch1['ObservationDate'],
                y=date_con_ch1['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8)
data8 = [trace8];
layout = dict(title = '<b>Deaths</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data8, layout = layout)
iplot(fig)


# In[ ]:


trace9 = go.Scatter(
                x=date_con_ch1['ObservationDate'],
                y=date_con_ch1['Recovered'],
                name="Recovered",
                line_color='green',
                opacity=0.8)
data9 = [trace9];
layout = dict(title = '<b>Recovered</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data9, layout = layout)
iplot(fig)


# ## Spread of the Coronavirus Over Time In USA

# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')
data['ObservationDate'] = data['ObservationDate'].dt.date
spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))


us_data = spread[spread['Country/Region']=='US']
date_con_ch2 = us_data.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum().reset_index()


# In[ ]:


us_data


# In[ ]:


date_con_ch2


# In[ ]:


data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], errors='coerce')
data['ObservationDate'] = data['ObservationDate'].dt.date
#spread = data[data['ObservationDate'] > pd.Timestamp(date(2020,1,21))]
#spread_gl = spread.groupby('ObservationDate')["Confirmed", "Deaths", "Recovered"].sum().reset_index()
from plotly.subplots import make_subplots
#fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))

trace10 = go.Scatter(
                x=date_con_ch2['ObservationDate'],
                y=date_con_ch2['Confirmed'],
                name="Confirmed",
                line_color='orange',
                opacity=0.8)
data10 = [trace10];
layout = dict(title = '<b>Confirmed</b>',
              xaxis= dict(title= 'Date',ticklen= 10,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data10, layout = layout)
iplot(fig)


# In[ ]:


trace11 = go.Scatter(
                x=date_con_ch2['ObservationDate'],
                y=date_con_ch2['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8)
data11 = [trace11];
layout = dict(title = '<b>Deaths</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data11, layout = layout)
iplot(fig)


# In[ ]:


trace12 = go.Scatter(
                x=date_con_ch2['ObservationDate'],
                y=date_con_ch2['Recovered'],
                name="Recovered",
                line_color='green',
                opacity=0.8)
data12 = [trace12];
layout = dict(title = '<b>Recovered</b>',
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis= dict(title = 'No. of Cases', ticklen=5, zeroline = False)
             )
fig = dict(data = data12, layout = layout)
iplot(fig)

