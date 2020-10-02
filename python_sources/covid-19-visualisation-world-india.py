#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import requests
import warnings
import json
import folium
warnings.filterwarnings('ignore')
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
py.init_notebook_mode(connected=True)
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot


# # Worldwide Trend

# In[ ]:


full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
full_table.sample(6)


# In[ ]:


# Cleaning data

# fixing Country values
full_table.loc[full_table['Province/State']=='Greenland', 'Country/Region'] = 'Greenland'

# Active Case = confirmed - deaths - recovered
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
full_table[['Province/State']] = full_table[['Province/State']].fillna('')
full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)

# fixing datatypes
full_table['Recovered'] = full_table['Recovered'].astype(int)

full_table.sample(6)


# In[ ]:


# Visualise Active, Deaths and Recovered
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])
fig = px.treemap(tm, path=["variable"], values="value", height=225, width=1200,
                 color_discrete_sequence=['blue', 'green', 'red'])
fig.data[0].textinfo = 'label+text+value'
fig.show()
################################################################
temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
             title='Cases over time', color_discrete_sequence = ['green', 'red', 'blue'])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
################################################################
# World wide Map
temp = full_table[full_table['Date'] == max(full_table['Date'])]
m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=2, max_zoom=4, zoom_start=2)

for i in range(0, len(temp)):
    folium.Circle(
        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],
        color='crimson', fill='crimson',
        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(temp.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths']),
        radius=int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)
m


# In[ ]:


# Grouped by day, country

full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

# new cases 
temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)
temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

# renaming columns
temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']

# merging new values
full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])

# filling na with 0
full_grouped = full_grouped.fillna(0)

# fixing data types
cols = ['New cases', 'New deaths', 'New recovered']
full_grouped[cols] = full_grouped[cols].astype('int')
full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)

################################################################

# Over the time
fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names', color=np.log(full_grouped["Confirmed"]), 
                    hover_name="Country/Region", animation_frame=full_grouped["Date"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.Purp)
fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[ ]:


fig = px.bar(full_grouped, x="Date", y="Confirmed", color='Country/Region', height=600,
             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

################################################################

fig = px.bar(full_grouped, x="Date", y="Deaths", color='Country/Region', height=600,
             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

################################################################

fig = px.bar(full_grouped, x="Date", y="New cases", color='Country/Region', height=600,
             title='New cases', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()


# In[ ]:


# Country wise

# getting latest values
country_wise = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)

# group by country
country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()

# per 100 cases
country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)
country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)
country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)

cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']
country_wise[cols] = country_wise[cols].fillna(0)

country_wise.head()


# In[ ]:


# confirmed - deaths
confirmed_plot = px.bar(country_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="Country/Region", 
               text='Confirmed', orientation='h', color_discrete_sequence = ['blue'])
deaths_plot = px.bar(country_wise.sort_values('Deaths').tail(15), x="Deaths", y="Country/Region", 
               text='Deaths', orientation='h', color_discrete_sequence = ['red'])
# recovered - active
recovered_plot = px.bar(country_wise.sort_values('Recovered').tail(15), x="Recovered", y="Country/Region", 
               text='Recovered', orientation='h', color_discrete_sequence = ['green'])
active_plot = px.bar(country_wise.sort_values('Active').tail(15), x="Active", y="Country/Region", 
               text='Active', orientation='h', color_discrete_sequence = ['#333333'])

# plot
fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,
                    subplot_titles=('Confirmed cases', 'Deaths reported', 'Recovered', 'Active cases'))
fig.add_trace(confirmed_plot['data'][0], row=1, col=1)
fig.add_trace(deaths_plot['data'][0], row=1, col=2)
fig.add_trace(recovered_plot['data'][0], row=2, col=1)
fig.add_trace(active_plot['data'][0], row=2, col=2)
fig.show()


# In[ ]:


fig = px.scatter(country_wise.sort_values('Deaths', ascending=False).iloc[:25, :], 
                 x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=700,
                 text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed (Scale is in log10)')
fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()


# In[ ]:


full_latest = full_table[full_table['Date'] == max(full_table['Date'])]
                         
fig = px.treemap(full_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 
                 path=["Country/Region", "Province/State"], values="Confirmed", height=700,
                 title='Number of Confirmed Cases',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
fig.data[0].textinfo = 'label+text+value'
fig.show()

fig = px.treemap(full_latest.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 
                 path=["Country/Region", "Province/State"], values="Deaths", height=700,
                 title='Number of Deaths reported',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# Source 
# https://app.flourish.studio/visualisation/1571387/edit

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')


# # COVID-19 in India 

# In[ ]:


# Load Data
individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
age_groups = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
covid19India = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
hospitalBeds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
# icmrTestDetails = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')
icmrTestLabs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
indiaCencus = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')
stateDetails = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')


# In[ ]:


individual_details.head()


# In[ ]:


age_groups.head(10)


# Age Group Distribution

# In[ ]:


# Age has multiple values, so split column
age_split_data=individual_details['age'].str.split("-",expand=True)
age_split_data.columns=['age','age2']
age_split_data.head()
age_split_data['age'] = pd.to_numeric(age_split_data['age'])

plt.figure(figsize=[10,8])
sns.distplot(age_split_data['age'],bins=10,kde=False)
plt.xlabel("Age", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()


# Gender Distribution

# In[ ]:


plt.figure(figsize=(14,8))
sns.countplot(data=individual_details,x='gender',order=individual_details['gender'].value_counts().index)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Total Cases')
order2 = individual_details['gender'].value_counts()

for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / individual_details.gender.dropna().count() )
    plt.text(i,count+2,strt,ha='center')


# In[ ]:


# Fill Gender NA values with Missing
individual_details.gender.fillna('Missing',inplace = True)
plt.figure(figsize=(14,8))
sns.countplot(data=individual_details,x='gender',order=individual_details['gender'].value_counts().index)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Total Cases')
order2 = individual_details['gender'].value_counts()

for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / individual_details.gender.dropna().count() )
    plt.text(i,count+2,strt,ha='center')


# This shows that there is a large number or records with missing values.

# In[ ]:


covid19India.head(10)


# In[ ]:


covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
df1=covid19India.groupby('Date').sum()
df1.reset_index(inplace=True)
print(df1)


# **Total Cases breakdown** - Confirmed,Active,Deaths

# In[ ]:


plt.figure(figsize= (14,8))

ax1 = sns.lineplot(data=df1, x="Date", y="Confirmed",markers=True,color = 'blue',label="Confirmed")
ax2 = sns.lineplot(data=df1, x="Date",y='Cured',markers=True,color = 'green',label="Cured")
ax3 = sns.lineplot(data=df1, x="Date",y='Deaths',markers=True,color = 'red',label="Deaths")
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total Cases',fontsize = 20)
plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)


# In[ ]:


df2=df1.tail(30)
df2['Date'] = df2['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
plt.figure(figsize=(20,8))
sns.barplot(data=df2,x='Date',y='Confirmed',color=sns.color_palette('Set3')[5],label='Confirmed')
sns.barplot(data=df2,x='Date',y='Cured',color=sns.color_palette('Set3')[2],label='Cured')
sns.barplot(data=df2,x='Date',y='Deaths',color=sns.color_palette('Set3')[3],label='Deaths')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation = 90)
plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)
plt.legend(frameon=True,fontsize=12);


# In[ ]:


state_cases=covid19India.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
state_cases['Active'] = state_cases['Confirmed'] - abs((state_cases['Deaths']- state_cases['Cured']))
state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)
(state_cases.sort_values('Confirmed', ascending= False)
.fillna(0)
.style
.background_gradient(cmap='Greys',subset=["Confirmed"])
.background_gradient(cmap='Blues',subset=["Deaths"])
.background_gradient(cmap='Greens',subset=["Cured"])
.background_gradient(cmap='Purples',subset=["Active"])
.background_gradient(cmap='Reds',subset=["Death Rate (per 100)"])
.background_gradient(cmap='Oranges',subset=["Cure Rate (per 100)"]))


# In[ ]:


state_cases=state_cases.sort_values('Confirmed', ascending= False).fillna(0)
state_cases=state_cases.head(15)
state_cases
plt.figure(figsize=(20,8))
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Confirmed',color=sns.color_palette('Set3')[3],label='Confirmed')
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Active',color=sns.color_palette('Set3')[7],label='Active')
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Cured',color=sns.color_palette('Set3')[8],label='Cured')
sns.barplot(data=state_cases,x='State/UnionTerritory',y='Deaths',color=sns.color_palette('Set3')[9],label='Deaths')
plt.xticks(rotation=90)
plt.legend();


# In[ ]:


fig = px.treemap(state_cases,path=['State/UnionTerritory'],values='Active',hover_data=['Confirmed','Deaths','Cured'],color='Active',
                 color_continuous_scale='Reds')
fig.show()


# In[ ]:


df3=individual_details.groupby(['detected_state','detected_district']).count()
df3.reset_index(inplace=True)
states_list=['Maharashtra','Gujarat','Delhi','Rajasthan','Madhya Pradesh','Tamil Nadu','Uttar Pradesh','Telangana','Andhra Pradesh',
            'West Bengal','Karnataka','Kerala','Jammu and Kashmir','Punjab','Haryana']
plt.figure(figsize=(20,60))
for i,state in enumerate(states_list):
    plt.subplot(8,2,i+1)
    df4=df3[df3['detected_state']==state].sort_values('id',ascending=False)
    df4=df4.head(10)
    sns.barplot(data=df4,x='id',y='detected_district')
    plt.xlabel('Number of Cases')
    plt.ylabel('')
    plt.title(state)
plt.tight_layout()
plt.show()


# In[ ]:


states_list=['Maharashtra','Gujarat','Delhi','Rajasthan','Madhya Pradesh','Tamil Nadu','Uttar Pradesh','Andhra Pradesh',
            'West Bengal','Karnataka','Kerala','Jammu and Kashmir','Punjab','Haryana']
df5=covid19India[covid19India['Date']>'2020-04-07']
df5=df5.groupby(['Date','State/UnionTerritory']).sum()
df5.reset_index(inplace=True)
df5['Date'] = df5['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
plt.figure(figsize=(20,60))

for i,state in enumerate(states_list):
    plt.subplot(7,2,i+1)
    df4=df5[df5['State/UnionTerritory']==state]
    plt.bar(df4.Date,df4.Confirmed,label='Confirmed')
    plt.bar(df4.Date,df4.Cured,label='Cured')
    plt.bar(df4.Date,df4.Deaths,label='Death')
    plt.xticks(rotation=90)
    plt.title(state)
    plt.ylabel('Total Cases')
    plt.xlabel('Date')
    plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
data=covid19India.groupby(['Date','State/UnionTerritory'])['Confirmed','Cured','Deaths'].sum()
data.reset_index(inplace=True)
data['Date']=data['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))


state_cases=covid19India.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()

with open('../input/india-geojson-file/state/india_state.geojson') as file:
    geojsonData = json.load(file)
    
for i in geojsonData['features']:
    if(i['properties']['NAME_1']=='Orissa'):
        i['properties']['NAME_1']='Odisha'
    elif(i['properties']['NAME_1']=='Uttaranchal'):
        i['properties']['NAME_1']='Uttarakhand'
    
for i in geojsonData['features']:
    i['id'] = i['properties']['NAME_1']
    
data = state_cases

map_choropleth = folium.Map(location = [20.5937,78.9629], zoom_start = 4)

folium.Choropleth(geo_data=geojsonData,
                 data=data,
                 name='CHOROPLETH',
                 key_on='feature.id',
                 columns = ['State/UnionTerritory','Confirmed'],
                 fill_color='YlOrRd',
                 fill_opacity=0.7,
                 line_opacity=0.8,
                 legend_name='Confirmed Cases',
                 highlight=True).add_to(map_choropleth)

folium.LayerControl().add_to(map_choropleth)

display(map_choropleth)


# **Testing and healthcare insights**

# In[ ]:


fig = px.treemap(icmrTestLabs, path=['state','city'],
                  color='city', hover_data=['lab','address'],
                  color_continuous_scale='Greens')
fig.show()


# In[ ]:


state=list(icmrTestLabs['state'].value_counts().index)
count=list(icmrTestLabs['state'].value_counts())
plt.figure(figsize=(20,8))
sns.barplot(x=state,y=count,color=sns.color_palette('Set2')[6])
plt.xlabel('State')
plt.ylabel('Count')
plt.title('ICMR Test labs per States')
plt.xticks(rotation=90)
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(20,60))
plt.subplot(4,1,1)
hospitalBeds=hospitalBeds.sort_values('NumUrbanHospitals_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanHospitals_NHP18',color=sns.color_palette('Pastel2')[6])
plt.title('Urban Hospitals per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumUrbanHospitals_NHP18']
    plt.text(count+100,i,count,ha='center',va='center')

plt.subplot(4,1,2)
hospitalBeds=hospitalBeds.sort_values('NumRuralHospitals_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralHospitals_NHP18',color=sns.color_palette('Pastel2')[7])
plt.title('Rural Hospitals per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumRuralHospitals_NHP18']
    plt.text(count+500,i,count,ha='center',va='center')

plt.subplot(4,1,3)
hospitalBeds=hospitalBeds.sort_values('NumUrbanBeds_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanBeds_NHP18',color=sns.color_palette('Pastel2')[3])
plt.title('Rural Beds per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumUrbanBeds_NHP18']
    plt.text(count+3000,i,count,ha='center',va='center')

plt.subplot(4,1,4)
hospitalBeds=hospitalBeds.sort_values('NumRuralBeds_NHP18', ascending= False)
sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralBeds_NHP18',color=sns.color_palette('Pastel2')[2])
plt.title('Rural Beds per states')
plt.xlabel('Count')
plt.ylabel('States')
for i in range(hospitalBeds.shape[0]):
    count = hospitalBeds.iloc[i]['NumRuralBeds_NHP18']
    plt.text(count+3000,i,count,ha='center',va='center')

plt.show()
plt.tight_layout()


# **Prediction by Prophet Model**

# In[ ]:


pred = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
pred = pred[pred["Country/Region"]=="India"]
pred = pred.fillna(0)
predgrp = pred.groupby("Date")[["Confirmed","Recovered","Deaths"]].sum().reset_index()

#Model
pred_cnfrm = predgrp.loc[:,["Date","Confirmed"]]
pr_data = pred_cnfrm
pr_data.columns = ['ds','y']
m=Prophet()
m.fit(pr_data)
future=m.make_future_dataframe(periods=15)
forecast=m.predict(future)

fig = plot_plotly(m, forecast)
py.iplot(fig)

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')

