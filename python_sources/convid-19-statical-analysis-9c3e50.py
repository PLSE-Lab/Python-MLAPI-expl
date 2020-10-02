#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebrimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import os
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df= pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# **We will convert ObservationDate and Last Update to datetime since they are currently taken as object<br><br>
#  The counts for 'Confirmed', 'Deaths' and 'Recovered' will be converted to int**

# In[ ]:


df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df['Last Update'] = pd.to_datetime(df['Last Update'])
df['Confirmed']=df['Confirmed'].astype('int')
df['Deaths']=df['Deaths'].astype('int')
df['Recovered']=df['Recovered'].astype('int')


#  **Since the final count of cases is present on 17th May 2020 we will create a separate dataframe for the same**

# In[ ]:


from datetime import date
recent=df[['ObservationDate']][-1:].max()
df_update=df.loc[df.ObservationDate==pd.Timestamp(recent['ObservationDate'])]
df_update


# In[ ]:


df_update.isnull().sum()


# **Out of the 25959 records we are now left with only 377 and many of the records dont have a Province defined.<br>
# These are mostly provinces that are not part of China**

# **Wherever Province is null, we replace it with the Country name and we group Mainland China and China together in China**

# In[ ]:


df_update['Province/State']=df_update.apply(lambda x: x['Country/Region'] if pd.isnull(x['Province/State']) else x['Province/State'],axis=1)
df['Province/State']=df.apply(lambda x: x['Country/Region'] if pd.isnull(x['Province/State']) else x['Province/State'],axis=1)


# In[ ]:


df_update['Country/Region']=df_update.apply(lambda x:'China' if x['Country/Region']=='Mainland China' else x['Country/Region'],axis=1)
df['Country/Region']=df.apply(lambda x:'China' if x['Country/Region']=='Mainland China' else x['Country/Region'],axis=1)


# ** We perform encoding of the Country to CountryID and Province to ProvinceID**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_update['ProvinceID'] = le.fit_transform(df_update['Province/State'])
df_update['CountryID']=le.fit_transform(df_update['Country/Region'])
df_update.head()


# In[ ]:


corr= df_update.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


num_plot_global=num_plot.reset_index()
num_plot_global['Death Case Increase']=0
num_plot_global['Confirmed Case Increase']=0
num_plot_global['Confirmed Case Increase'][0]=0
num_plot_global['Death Case Increase'][0]=0
for i in range(1,num_plot_global.shape[0]):
    num_plot_global['Confirmed Case Increase'][i]=-(num_plot_global.iloc[i-1][1]-num_plot_global.iloc[i][1])
    num_plot_global['Death Case Increase'][i]=-(num_plot_global.iloc[i-1][3]-num_plot_global.iloc[i][3])
num_plot_global.tail()


# # India

# **Observing the cases in India. Confirmed cases are increasing in India each day. There is a need to get a flatter curve for confirmed cases which currently is in upswing with a steep increase since past few days.**

# In[ ]:


india_cases_complete=df.loc[df['Country/Region']=='India']
india_cases_complete['date'] = india_cases_complete['ObservationDate'].dt.date
india_cases_complete['date']=pd.to_datetime(india_cases_complete['date'])
india_cases_complete = india_cases_complete[india_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]
num_plot = india_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()
num_plot.plot(figsize=(8,8),colormap='winter',title='Per Day statistics for India',marker='o')
num_plot_india=num_plot.reset_index()


# In[ ]:


num_plot_india['Confirmed Case Increase']=0
num_plot_india['Death Case Increase']=0
num_plot_india['Confirmed Case Increase'][0]=0
num_plot_india['Death Case Increase'][0]=0
for i in range(1,num_plot_india.shape[0]):
    num_plot_india['Confirmed Case Increase'][i]=-(num_plot_india.iloc[i-1][1]-num_plot_india.iloc[i][1])
    num_plot_india['Death Case Increase'][i]=-(num_plot_india.iloc[i-1][3]-num_plot_india.iloc[i][3])
num_plot_india.tail()


# **17th May has recorded highest number of COVID19 confirmed cases in India in a day (5050). We notice a peak in every 4-5 days**

# In[ ]:


num_plot_india['Confirmed Case Increase'].plot(kind='bar',width=0.95,colormap='winter',figsize=(20,6),title='Confirmed Case Increase')
plt.show()


# **There seems to be an issue with an extra death reported on Day 50 due to which on Day 51 we see a downtrend.Highest number of deaths reported in a day is 175 for India which was on 4th May.**

# In[ ]:


num_plot_india['Death Case Increase'].plot(kind='bar',width=0.95,colormap='winter',figsize=(20,6),title='Death Case Increase')
plt.show()


# # Trajectories for some of the countries

# **Lets look at some of country graphs together and check the trajectory being followed. As seen below all the countries are following the same trajectory. South Korea was able to break the chain quiet early as compared to the other nations. Germany and Italy are making the downturn move. The good news for India is that it made some downward movement from the earlier trajectory but there is a drop to be seen yet.**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from plotly.offline import iplot, init_notebook_mode
import math
import bokeh 
import matplotlib.pyplot as plt
import plotly.express as px
from urllib.request import urlopen
import json
from dateutil import parser
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
from bokeh.resources import INLINE
from bokeh.io import output_notebook
from bokeh.models import Span
import warnings
warnings.filterwarnings("ignore")
output_notebook(resources=INLINE)
le=LabelEncoder()

df.rename(columns={'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)
df = df.fillna('unknown')
df['Country'] = df['Country'].str.replace('US','United States')
df['Country'] = df['Country'].str.replace('UK','United Kingdom') 
df['Country'] = df['Country'].str.replace('Mainland China','China')
df['Code']=le.fit_transform(df['Country'])
virus_data = df
#print(virus_data.head())
#print(len(virus_data))

top_country = virus_data.loc[virus_data['Date'] == virus_data['Date'].iloc[-1]]
top_country = top_country.groupby(['Code','Country'])['Confirmed'].sum().reset_index()
top_country = top_country.sort_values('Confirmed', ascending=False)
top_country = top_country[:50]
top_country_codes = top_country['Country']
top_country_codes = list(top_country_codes)
#print(top_country)

countries = virus_data[virus_data['Country'].isin(top_country_codes)]
countries_day = countries.groupby(['Date','Code','Country'])['Confirmed','Deaths','Recovered'].sum().reset_index()
#print(countries_day)


exponential_line_x = []
exponential_line_y = []
for i in range(16):
    exponential_line_x.append(i)
    exponential_line_y.append(i)

china = countries_day.loc[countries_day['Code']==43]

new_confirmed_cases_china = []
new_confirmed_cases_china.append( list(china['Confirmed'])[0] - list(china['Deaths'])[0] 
                           - list(china['Recovered'])[0] )

for i in range(1,len(china)):

    new_confirmed_cases_china.append( list(china['Confirmed'])[i] - 
                                     list(china['Deaths'])[i] - 
                                     list(china['Recovered'])[i])
    
    
italy = countries_day.loc[countries_day['Code']==102]

new_confirmed_cases_ita = []
new_confirmed_cases_ita.append( list(italy['Confirmed'])[0] - list(italy['Deaths'])[0] 
                           - list(italy['Recovered'])[0] )

for i in range(1,len(italy)):
    
    new_confirmed_cases_ita.append( list(italy['Confirmed'])[i] - 
                                  list(italy['Deaths'])[i] - 
                                  list(italy['Recovered'])[i])
    
    
skorea = countries_day.loc[countries_day['Code']==186]

new_confirmed_cases_skorea = []
new_confirmed_cases_skorea.append( list(skorea['Confirmed'])[0] - list(skorea['Deaths'])[0] 
                           - list(skorea['Recovered'])[0] )

for i in range(1,len(skorea)):
    
    new_confirmed_cases_skorea.append( list(skorea['Confirmed'])[i] - 
                                     list(skorea['Deaths'])[i] - 
                                    list(skorea['Recovered'])[i])
    
    
india = countries_day.loc[countries_day['Code']==96]

new_confirmed_cases_india = []
new_confirmed_cases_india.append( list(india['Confirmed'])[0] - list(india['Deaths'])[0] 
                           - list(india['Recovered'])[0] )

for i in range(1,len(india)):
    
    new_confirmed_cases_india.append( list(india['Confirmed'])[i] - 
                                     list(india['Deaths'])[i] - 
                                    list(india['Recovered'])[i])
    

spain = countries_day.loc[countries_day['Code']==188]

new_confirmed_cases_spain = []
new_confirmed_cases_spain.append( list(spain['Confirmed'])[0] - list(spain['Deaths'])[0] 
                           - list(spain['Recovered'])[0] )

for i in range(1,len(spain)):
    
    new_confirmed_cases_spain.append( list(spain['Confirmed'])[i] - 
                                     list(spain['Deaths'])[i] - 
                                    list(spain['Recovered'])[i])
    

us = countries_day.loc[countries_day['Code']==211]

new_confirmed_cases_us = []
new_confirmed_cases_us.append( list(us['Confirmed'])[0] - list(us['Deaths'])[0] 
                           - list(us['Recovered'])[0] )

for i in range(1,len(us)):
    
    new_confirmed_cases_us.append( list(us['Confirmed'])[i] - 
                                     list(us['Deaths'])[i] - 
                                    list(us['Recovered'])[i])
    
    
german = countries_day.loc[countries_day['Code']==77]

new_confirmed_cases_german = []
new_confirmed_cases_german.append( list(german['Confirmed'])[0] - list(german['Deaths'])[0] 
                           - list(german['Recovered'])[0] )

for i in range(1,len(german)):
    
    new_confirmed_cases_german.append( list(german['Confirmed'])[i] - 
                                     list(german['Deaths'])[i] - 
                                    list(german['Recovered'])[i])
    
p1=figure(plot_width=800, plot_height=550, title="COVID 2019 Trajectories for Countries")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Total number of Confirmed Cases (Log scale)'
p1.yaxis.axis_label = 'Total number of active cases (Log scale)'


p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=1)

p1.line(np.log(list(china['Confirmed'])), np.log(new_confirmed_cases_china), color='red', 
        legend_label='China', line_width=3)
p1.circle(np.log(list(china['Confirmed'])[-1]), np.log(new_confirmed_cases_china[-1]), size=5)

p1.line(np.log(list(italy['Confirmed'])), np.log(new_confirmed_cases_ita), color='blue', 
        legend_label='Italy', line_width=3)
p1.circle(np.log(list(italy['Confirmed'])[-1]), np.log(new_confirmed_cases_ita[-1]), size=5)



p1.line(np.log(list(skorea['Confirmed'])), np.log(new_confirmed_cases_skorea), color='violet', 
        legend_label='South Korea', line_width=3)
p1.circle(np.log(list(skorea['Confirmed'])[-1]), np.log(new_confirmed_cases_skorea[-1]), size=5)


p1.line(np.log(list(india['Confirmed'])), np.log(new_confirmed_cases_india), color='orange', 
        legend_label='India', line_width=3)
p1.circle(np.log(list(india['Confirmed'])[-1]), np.log(new_confirmed_cases_india[-1]), size=5)

p1.line(np.log(list(spain['Confirmed'])), np.log(new_confirmed_cases_spain), color='brown', 
        legend_label='Spain', line_width=3)
p1.circle(np.log(list(spain['Confirmed'])[-1]), np.log(new_confirmed_cases_spain[-1]), size=5)

p1.line(np.log(list(us['Confirmed'])), np.log(new_confirmed_cases_us), color='green', 
        legend_label='United States', line_width=3)
p1.circle(np.log(list(us['Confirmed'])[-1]), np.log(new_confirmed_cases_us[-1]), size=5)

p1.line(np.log(list(german['Confirmed'])), np.log(new_confirmed_cases_german), color='black', 
        legend_label='Germany', line_width=3)
p1.circle(np.log(list(german['Confirmed'])[-1]), np.log(new_confirmed_cases_german[-1]), size=5)

p1.legend.location = "bottom_right"
#output_file("coronavirus.html", title="COVID2019 Trajectory")
show(p1)



# **Adding other datasources for further analysis of India on State-level**

# In[ ]:


import requests
import io
age_group = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
india_covid_19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')
ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
state_testing = pd.read_csv('../input/statewisetestingdetailsindiacsv/statewise_tested_numbers_data.csv')


# In[ ]:


#Removal of 'Unassigned' State/UnionTerritory
india_covid_19.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered'}, inplace=True)
unassigned=india_covid_19[india_covid_19['State']=='Unassigned'].index
india_covid_19.drop(unassigned,axis=0,inplace=True)
unassigned1=india_covid_19[india_covid_19['State']=='Nagaland#'].index
india_covid_19.drop(unassigned1,axis=0,inplace=True)
unassigned2=india_covid_19[india_covid_19['State']=='Jharkhand#'].index
india_covid_19.drop(unassigned2,axis=0,inplace=True)
unassigned3=india_covid_19[india_covid_19['State']=='Madhya Pradesh#'].index
india_covid_19.drop(unassigned3,axis=0,inplace=True)


# **Statewise Confirmed Cases in India for COVID-2019**

# In[ ]:



statewise_cases = pd.DataFrame(india_covid_19.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())
statewise_cases["Country"] = "India" 
fig = px.treemap(statewise_cases, path=['Country','State'], values='Confirmed',color='Confirmed', hover_data=['State'])
fig.show()


# **Gender-wise distribution of the COVID-2019 cases**<br>
# The figure shows the cases are more than double in Males than Females

# In[ ]:


labels = ['Male', 'Female']
sizes = []
sizes.append(list(individual_details['gender'].value_counts())[0])
sizes.append(list(individual_details['gender'].value_counts())[1])
explode = (0.05, 0)
colors = ['#ffcc99','#66b3ff']
plt.figure(figsize= (8,8))
plt.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f',startangle=90)
plt.title('Percentage of Gender (Ignoring the Missing Values)',fontsize = 10)
plt.show ()


# **Age-wise distribution of the COVID-2019 cases**<br>
# The cases are more common in elderly population as seen in the graph below

# In[ ]:


fig = plt.figure(figsize=(10,10))
age_dist_india = age_group.groupby('AgeGroup')['Sno'].sum().sort_values(ascending=False)
def absolute_value(val):
    a  = val
    return (np.round(a,2))
age_dist_india.plot(kind="pie",title='Case Distribution by Age',autopct=absolute_value,colormap='Paired',startangle=90)

plt.show ()


# **Statewise Recovery and Death Rate in India**<br>
# 1)Maharashtra has the highest number of Confirmed cases in India<br>
# 2) Kerala has highest recovery rate among the top 20 hotspot states in India <br>
# 3)We have 3 Green states (COVID-19-free) now in India since Goa and Manipur have reported fresh cases <br>
# 4)West Bengal has the highest death rate.

# In[ ]:


india_covid_19['Deaths']=india_covid_19['Deaths'].astype('int')


# In[ ]:


state_details = pd.pivot_table(india_covid_19, values=['Confirmed','Deaths','Recovered'], index='State', aggfunc='max')
state_details['Recovery Rate'] = round(state_details['Recovered'] / state_details['Confirmed'],2)
state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)
state_details = state_details.sort_values(by='Confirmed', ascending= False)
state_details.style.background_gradient(cmap='Purples')


# **Statewise Testing done so far in India till 15th May**<br>
# Tamil Nadu has done maximum number of tests with 3% of tested cases being positive<br>
# Maharashta has the highest positive rate (11%) showing that the virus has spread widely in this area.

# In[ ]:


testing=state_testing.groupby('State')['Total Tested'].max().sort_values(ascending=False).reset_index()
fig = px.bar(testing, 
             x="Total Tested",
             y="State", 
             orientation='h',
             height=800,
             title='Statewise Testing',
            color='State')
fig.show()


# In[ ]:


state_test_details = pd.pivot_table(state_testing, values=['Total Tested','Positive','Negative'], index='State', aggfunc='max')
state_test_details['Positive Test Rate'] = round(state_test_details['Positive'] / state_test_details['Total Tested'],2)
state_test_details['Negative Test Rate'] = round(state_test_details['Negative'] /state_test_details['Total Tested'], 2)
state_test_details = state_test_details.sort_values(by='Total Tested', ascending= False)
state_test_details.style.background_gradient(cmap='Blues')


# **Laboratories available for testing in Each of the states**

# In[ ]:


values = list(ICMR_labs['state'].value_counts())
states = list(ICMR_labs['state'].value_counts().index)
labs = pd.DataFrame(list(zip(values, states)), 
               columns =['values', 'states'])
fig = px.bar(labs, 
             x="values",
             y="states", 
             orientation='h',
             height=1000,
             title='Statewise Labs',
            color='states')
fig.show()


# **Hospital Infrastructure in India**

# **As seen below Uttar Pradesh has many hospitals followed mostly by Maharshtra**

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
hospital_beds_states =hospital_beds.drop([36])
cols_object = list(hospital_beds_states.columns[2:8])
for cols in cols_object:
    hospital_beds_states[cols] = hospital_beds_states[cols].astype(int,errors = 'ignore')
top_5_primary = hospital_beds_states.nlargest(5,'NumPrimaryHealthCenters_HMIS')
top_5_community = hospital_beds_states.nlargest(5,'NumCommunityHealthCenters_HMIS')
top_5_district_hospitals = hospital_beds_states.nlargest(5,'NumDistrictHospitals_HMIS')
top_5_public_facility = hospital_beds_states.nlargest(5,'TotalPublicHealthFacilities_HMIS')
top_5_public_beds = hospital_beds_states.nlargest(5,'NumPublicBeds_HMIS')
top_rural_hos = hospital_beds_states.nlargest(5,'NumRuralHospitals_NHP18')
top_rural_beds = hospital_beds_states.nlargest(5,'NumRuralBeds_NHP18')
top_urban_hos = hospital_beds_states.nlargest(5,'NumUrbanHospitals_NHP18')
top_urban_beds = hospital_beds_states.nlargest(5,'NumUrbanBeds_NHP18')

plt.figure(figsize=(30,30))
plt.suptitle('Health Facilities in Top 5 States',fontsize=30)
plt.subplot(231)
plt.title('Primary Health Centers',fontsize=25)
plt.barh(top_5_primary['State/UT'],top_5_primary['NumPrimaryHealthCenters_HMIS'],color ='blue');

plt.subplot(232)
plt.title('Community Health Centers',fontsize=25)
plt.barh(top_5_community['State/UT'],top_5_community['NumCommunityHealthCenters_HMIS'],color = 'blue');

plt.subplot(233)
plt.title('Public Health Facilities',fontsize=25)
plt.barh(top_5_public_facility['State/UT'],top_5_public_facility['TotalPublicHealthFacilities_HMIS'],color='blue');

plt.subplot(234)
plt.title('District Hospitals',fontsize=25)
plt.barh(top_5_district_hospitals['State/UT'],top_5_district_hospitals['NumDistrictHospitals_HMIS'],color = 'orange');

plt.subplot(235)
plt.title('Rural Hospitals',fontsize=25)
plt.barh(top_rural_hos['State/UT'],top_rural_hos['NumRuralHospitals_NHP18'],color = 'orange');
plt.subplot(236)
plt.title('Urban Hospitals',fontsize=25)
plt.barh(top_urban_hos['State/UT'],top_urban_hos['NumUrbanHospitals_NHP18'],color = 'orange');
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# **Number of Beds Facility-wise**

# In[ ]:



plt.figure(figsize=(27,15))
plt.suptitle('Number of Beds in Top 5 States',fontsize=30);
plt.subplot(131)
plt.title('Rural Beds',fontsize=25)
plt.barh(top_rural_beds['State/UT'],top_rural_beds['NumRuralBeds_NHP18'],color = 'orange');

plt.subplot(132)
plt.title('Urban Beds',fontsize=25)
plt.barh(top_urban_beds['State/UT'],top_urban_beds['NumUrbanBeds_NHP18'],color = 'blue');
plt.subplot(133)
plt.title('Public Beds',fontsize=25)
plt.barh(top_5_public_beds['State/UT'],top_5_public_beds['NumPublicBeds_HMIS'],color = 'purple');
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# # Predictions for India

# In[ ]:


#Current number of confirmed cases
ax = num_plot_india['Confirmed'].plot(title="Confirmed Cases in India",figsize=(8,8));
ax.set(xlabel="Date", ylabel="Confirmed Cases");


# **Making Predictions for India based on the current scenario **

# **Using Prophet**

# In[ ]:


train = num_plot_india.iloc[:-3,:2]
test = num_plot_india.iloc[-3:,:2]


# In[ ]:


train.rename(columns={"date":"ds","Confirmed":"y"},inplace=True)
test.rename(columns={"date":"ds","Confirmed":"y"},inplace=True)
test = test.set_index("ds")
test = test['y']


# In[ ]:


from fbprophet import Prophet
pd.plotting.register_matplotlib_converters()
model = Prophet(changepoint_prior_scale=0.4, changepoints=['2020-04-14','2020-04-25','2020-05-09','2020-05-14'])
model.fit(train)


# **We can see that by 31st May (End of Lockdown 4) more than 1.51L confirmed cases are predicted as per this model with upper limit of around 1.59L. As more data comes in these values will keep changing**

# In[ ]:


future_dates = model.make_future_dataframe(periods=20)
forecast =  model.predict(future_dates)
ax = forecast.plot(x='ds',y='yhat',label='Predicted Confirmed Case',legend=True,figsize=(10,10))
test.plot(y='y',label='Actual Confirmed Cases',legend=True,ax=ax)


# **Accuracy Metrics for Prophet**

# In[ ]:


from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(model, initial='60 days', period='20 days', horizon = '3 days')
df_cv.head()
df_p = performance_metrics(df_cv)
df_p.head()


# In[ ]:


forecast.tail(20)


# **Prediction considering till when entire population of India will be impacted**<br>
# If we consider logistic growth, between mid July to December the daily number of confirmed cases will be rising and by Jan 2021 we will be at the peak with daily increase being constant thereafter.

# In[ ]:


from fbprophet import Prophet
model_india = Prophet(growth="logistic",changepoint_prior_scale=0.4,changepoints=['2020-04-14','2020-04-25','2020-05-09','2020-05-14'])
pop = 1380004385 #from worldometers
train['cap'] = pop
model_india.fit(train)
# Future Prediction
future_dates = model_india.make_future_dataframe(periods=300)
future_dates['cap'] = pop
forecast =  model_india.predict(future_dates)
# Plotting
ax = forecast.plot(x='ds',y='yhat',label='Predicted Confirmed Cases',legend=True,figsize=(10,10))
test.plot(y='y',label='Actual Confirmed Counts',legend=True,ax=ax)
ax.set(xlabel="Date", ylabel="Confirmed Cases");


# In[ ]:


forecast.iloc[100:150]


# **Using ARIMA**

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
import datetime
arima = ARIMA(train['y'], order=(3, 1, 0))
arima = arima.fit(trend='nc', full_output=True, disp=True)
forecast = arima.forecast(steps= 30)
pred = list(forecast[0])
start_date = train['ds'].max()
prediction_dates = []
for i in range(30):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
plt.figure(figsize= (20,10))
plt.xlabel("Dates",fontsize = 10)
plt.ylabel('Total cases',fontsize = 10)
plt.title("Predicted Values for the next 25 Days" , fontsize = 20)

plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = 'blue',label = 'Predicted')
plt.plot_date(y=train['y'].tail(15),x=train['ds'].tail(15),linestyle = '-',color = 'orange',label = 'Actual')


# **Arima Predictions**

# **ARIMA is predicting 1.42L confirmed cases on 31st May**

# In[ ]:


pred=pd.DataFrame(forecast[0],columns=['Predicted'])
dates=pd.DataFrame(prediction_dates,columns=['Date'])
arima_df=pd.merge(dates,pred,right_index=True,left_index=True)
arima_df.tail(30)


# In[ ]:


test=test.reset_index()


# In[ ]:


df1=pd.DataFrame(forecast[0],columns=['yhat'])
df2=pd.DataFrame(prediction_dates,columns=['ds'])
df3=test['y']
df4=pd.merge(df2,df3,right_index=True,left_index=True)
df5=pd.merge(df4,df1,right_index=True,left_index=True)


# In[ ]:


df5['mse'],df5['rmse'],df5['mae'],df5['mape'],df5['mdape']=[0,0,0,0,0]


# In[ ]:


for t in range(len(test)):
    mape =  np.mean(np.abs(df5['yhat'][t] - df5['y'][t])/np.abs(df5['y'][t]))
    df5['mape'][t]="{:.5f}".format(mape)
    mdape =  np.median(np.abs(df5['yhat'][t] - df5['y'][t])/np.abs(df5['y'][t]))
    df5['mdape'][t]="{:.5f}".format(mdape)
    mae = np.mean(np.abs(df5['yhat'][t] - df5['y'][t]))
    df5['mae'][t]=mae
    mse = np.mean((df5['yhat'][t] - df5['y'][t])**2)
    df5['mse'][t]=mse
    rmse = np.mean((df5['yhat'][t] - df5['y'][t])**2)**.5
    df5['rmse'][t]=rmse


# **Accuracy Metrics for ARIMA**

# In[ ]:


df5


# **Prediction for Bed capacity in India**

# In[ ]:


num_plot_india['Active']=0
for i in range(len(num_plot_india)):
    num_plot_india['Active'][i]=num_plot_india['Confirmed'][i]-num_plot_india['Recovered'][i]-num_plot_india['Deaths'][i]
num_plot_india


# In[ ]:


train_bed=pd.DataFrame(columns=['ds','y'])
test_bed=pd.DataFrame(columns=['ds','y'])
train_bed_y= num_plot_india.iloc[:-5,-1:]
train_bed_ds = num_plot_india.iloc[:-5,:1]
train_bed=pd.merge(train_bed_ds,train_bed_y,right_index=True,left_index=True)
train_bed.rename(columns={'date': 'ds', 'Active': 'y'}, inplace=True)
test_bed_y = num_plot_india.iloc[-5:,-1:]
test_bed_ds = num_plot_india.iloc[-5:,:1]
test_bed=pd.merge(test_bed_ds,test_bed_y,right_index=True,left_index=True)
test_bed.rename(columns={'date': 'ds', 'Active': 'y'}, inplace=True)


# In[ ]:


test_bed = test_bed.set_index("ds")
test_bed = test_bed['y']


# **Considering current number of active cases, between mid May and mid-August we will see a drastic increase in the number of active cases and by end September all the available hospital beds in India will be occupied <br>
# if we donot lower the increase of cases or increase the number of beds**

# In[ ]:


num_bed=hospital_beds.iloc[36][7]+hospital_beds.iloc[36][9]+hospital_beds.iloc[36][11]
model_bed = Prophet(growth = "logistic",changepoints=['2020-04-10','2020-04-20','2020-05-02','2020-05-10'])
bed_cap = num_bed 
train_bed['cap'] = bed_cap
model_bed.fit(train_bed)
# Future Prediction
future_dates = model_bed.make_future_dataframe(periods=200)
future_dates['cap'] = bed_cap
forecast =  model_bed.predict(future_dates)
# Plotting
ax = forecast.plot(x='ds',y='yhat',label='Predicted Active Cases',legend=True,figsize=(10,10))
test_bed.plot(y='y',label='Actual Active Counts',legend=True,ax=ax)
ax.set(xlabel="Date", ylabel="Active Cases");


# In[ ]:


forecast.iloc[230:240]


# In[ ]:




