#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Basic Libraries
import pandas as pd
import numpy as np 
import math 
import re 

#Visualization 
import seaborn as sns
import matplotlib.pyplot as plt 
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
import folium
cf.go_offline()

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# In[ ]:


#Reading Data
df = pd.read_csv('../input/covid_19_clean_complete.csv')
df1 = pd.read_csv('../input/tests.csv')
df2 =  pd.read_csv('../input/usa_county_wise.csv')


# This is the first analysis I did about the COVID-19 , I am trying to used polynomial regression to predict future cases, "This data isnt mine" however I could not upload it from Kaggell
# 
# 1. In the First stage I will try to have some insight from the Data
# 2. I will try to have same insight from Nicaragua - Central America
# 3. Use polynomial Regression to predict cases using USA Data

# In[ ]:


df.tail(5) 


# In[ ]:


fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (20,5))
sns.heatmap(data = df.isna() ,yticklabels=False , cmap ="plasma", ax= ax1)
sns.heatmap(data = df1.isna() ,yticklabels=False ,cmap ="plasma", ax= ax2)
sns.heatmap(data = df2.isna() ,yticklabels=False ,cmap ="plasma", ax= ax3)
print("Missing Data")


# **COVID-19 World Wide**

# > "Checking df1 = Covid_all Arround the World

# In[ ]:


wd = df   # wd = World
wd.info()


# In[ ]:


wd['Date_c'] = pd.to_datetime(wd['Date'])  #wd World 
UptoDate =wd[wd['Date_c'] == max(wd['Date_c'])] # max because it will filter and show the latest day so It would be updated

#Create canvas map 
World = folium.Map(location = [0,0], tiles='OpenStreetMap', #tite = typed of map https://python-visualization.github.io/folium/quickstart.html
               min_zoom=2, max_zoom=5, zoom_start=2)

#Adding points and Circle 
for date in range (0 , len(UptoDate)):# the len od UptoDate is 265
    folium.Circle(
        radius=int(UptoDate.iloc[date]['Confirmed'])*0.5, # Go to the most updated case and them bring the Confirmed case 
        location=[UptoDate.iloc[date]['Lat'],UptoDate.iloc[date]['Long']], # provide the lat and long of the most updated value
        popup='The Waterfront',color='crimson',
        tooltip=    # her Will show the Legend of all Data 
        '<li><bold>Country : ' + str(UptoDate.iloc[date]['Country/Region'])+  # Podes ponerlos en etiqueta HTML 
        '<li><bold>Confirmed : ' + str(UptoDate.iloc[date]['Confirmed'])+
        '<li><bold>Deaths : ' + str(UptoDate.iloc[date]['Deaths'])+
        '<li><bold>Recovered : ' + str(UptoDate.iloc[date]['Recovered']) + 
         str(" Country :" + str(UptoDate.iloc[date]['Country/Region'])),  # str in python 
        fill=True,fill_color='#3186cc').add_to(World)


# **Mapa Mundi (COVID-19)**

# In[ ]:


World


# **COVID-19 SPREAD**

# In[ ]:


# Create a New Columns to Show Active cases arround the world 
wd['Active Cases'] = wd['Confirmed'] - wd['Recovered'] - wd['Deaths']

#Analazing Total Confirmed , Recovered, Active cases and Deaths (CRAD)
CRAD = wd[wd['Date_c'] == max(wd['Date_c'])][['Confirmed','Deaths','Recovered','Active Cases']].sum()

#plotting the CRAD
CRAD.iplot(kind= 'barh', color= "turquoise", title = 'COVID-19 Arround the World by Million Cases')


# In[ ]:


#Tracking the Spread Per day (SPD)

SPD = wd[['Date_c','Confirmed','Deaths','Recovered','Active Cases']].groupby('Date_c').sum()
SPD.iplot(kind= 'scatter', colors=['blue','Darkred','green','Purple'], title = 'COVID-19 Arround the World by Million Cases', xTitle = 'Date' , yTitle = 'People per Million')


# In[ ]:


import plotly.express as px
fig = px.choropleth(df, locations="Country/Region", locationmode='country names', color=np.log(df["Confirmed"]), 
                    hover_name="Country/Region", animation_frame=df["Date_c"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.Viridis)
fig.update(layout_coloraxis_showscale=False)
fig.show()


# **Recovery Rate**

# The recovery rate was Calculated based on number of confirmed vs Active cases and Death , please keep on Mind the following
# 
# 1. Specials thanks to #Devakumar kp for collecting the Data
# 2. Some countries like Canada, Mozambique and other just reported Confirmed and deaths so it is hard to stimate a Recovery Rate
# 3. Recovery Rate formula might not be the best one ( as far as I know they use Sigma to stimate some number)
# 4. ** This is for study/Practicce purpose, this might not be taken as official reports
# 

# In[ ]:


#Recovery Rate UpToDate = (Confirmed - (Infected + Deaths) / Confimed ) * 100
wd['RecRate']= round((((wd['Confirmed'] - (wd['Deaths']+wd['Active Cases'])) / (wd['Confirmed']))*100).fillna(0),2)
wd.tail(5)


# In[ ]:


NRR = wd[(wd['Date_c'] == max(wd['Date_c'])) & (wd['RecRate'] >= 0 )][['Country/Region','RecRate']].groupby('Country/Region').max().sort_values('RecRate')
NRR.iplot(kind= 'barh', xTitle = ' Recovery Rate 0- 100 %' , yTitle = 'Countries')


# In[ ]:


#Top Countries Confirmed cases
Cases_confirmed = wd.groupby('Country/Region',as_index= False)[['Confirmed']].max().sort_values('Confirmed')
CC = px.bar(Cases_confirmed.tail(7), x="Confirmed", y="Country/Region",  text='Confirmed', orientation='h', color_discrete_sequence = ['Green'])
#Top Countries Deaths reported
Deaths = wd.groupby('Country/Region',as_index= False)[['Deaths']].max().sort_values('Deaths')
D = px.bar(Deaths.tail(7), x="Deaths", y="Country/Region",  text='Deaths', orientation='h', color_discrete_sequence = ['Darkred'])
#Top Countries Recovered
Recovered = wd.groupby('Country/Region',as_index= False)[['Recovered']].max().sort_values('Recovered')
R = px.bar(Recovered.tail(7), x="Recovered", y="Country/Region",  text='Recovered', orientation='h', color_discrete_sequence = ['Orange'])
#Top Countries Active Cases
Active_cases = wd.groupby('Country/Region',as_index= False)[['Active Cases']].max().sort_values('Active Cases')
AC = px.bar(Active_cases.tail(7), x="Active Cases", y="Country/Region",  text='Active Cases', orientation='h', color_discrete_sequence = ['Skyblue'])


fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,
                    subplot_titles=('Top Countries Confirmed cases', 'Top Countries Deaths reported' , 'Top Countries Recovered','Top Countries Active Cases'  ))

fig.add_trace(CC['data'][0], row=1, col=1) #Remember hacerlo a un lista para que lo pueda leer

fig.add_trace(D['data'][0], row=1, col=2)
fig.add_trace(R['data'][0], row=2, col=1)
fig.add_trace(AC['data'][0], row=2, col=2)


fig.update_layout(height=700)


# **Conclusions**:
# 
# 1. US is top in All Categories 
# 2. Today is 6/9/2020 and top 5 Countries are US, Germany, Brazil , spain , Italy ,turkey 
# 3. Germany is doing awesome After US it has the 2nd position on Recovery Cases , Following by Brazil and Spain
# 4. In term of death UK is in Second position but up to today They have less Active Cases - considering they were addected before USA(graph #4)
# 5. Data might not be accurate because WHO reported some countries are not reporting all cases ***For study Purpose******

# **COVID-  19 in Nicaragua**

# In[ ]:


#Extracting Nic Information
nic = df[df["Country/Region"] == "Nicaragua"].reset_index()
nic.head(2)


# In[ ]:


#Cleaning Data
df_nic= nic.drop(['index','Province/State'], axis = 1) 
df_nic.head()


# **In this point I will add the state(called apartments in Nic) randomly,**
# 
# 1. It will add some noise and the data will not be accurate in the map * but it is only for educational purpose*
# 2. Because I want to learn how to plot a map with full information
# 3. It will be a great way to develope my skills
# 4. * Under any circustances this could be take it as real information *
# 5. Yes I know Nicaragua

# In[ ]:


df_nic.tail(5)


# In[ ]:


# Spread COVD- 19 per Day 
NICSPREAD = df_nic[['Date_c','Confirmed','Deaths','Recovered','Active Cases']].groupby('Date_c').sum()
NICSPREAD.iplot(kind='scatter', title = 'Cases in Nicaragua', xTitle = 'Date' , yTitle = 'Verified Cases')


# **it might looks that the numbers are wrong or something wrong but NO check this out **https://www.bbc.com/mundo/noticias-america-latina-52716064
# 
# 1. it means "goverment tries to hide the covid-19 impact" ** I post this for references for you to know what happened with the data trend

# In[ ]:


#CRAD_NIC (Confirmed, Recovered, Actived, Death)
CRAD_NIC = df_nic[df_nic['Date_c'] == max(df_nic['Date_c'])][['Confirmed','Deaths','Recovered','Active Cases']].sum() # max or Sum will give the same value
CRAD_NIC.iplot(kind= 'barh', color= "Green", title = 'COVID-19 in Nicaragua')


# In[ ]:


#COvid 19 - Spread Distribution 
fig,(ax1) = plt.subplots(1,1 , figsize = (35,15))
n = sns.scatterplot(x= 'Date' , y = 'Confirmed', data = df_nic, ax=ax1)
n.set_xticklabels(n.get_xticklabels(), rotation=45)
n.set_title('CoronaVirus in Nic')


# **Creating Nic Maps With Covid-19 Cases**
# 1. The Following Data was recollected and created by me to pratice Choropleth plots
# 2. The Data it is not correct (covid-19 cases)
# 3. The total Covid - 19 cases (From the originall Data Set) was distributed randomly among the 17 departments/states

# In[ ]:


# Adding Long and long
nic_dep = pd.read_excel('../input/lat_long_nic.xlsx')
#nic_dep = Nicaragua + Departments
nic_dep['Lat']= nic_dep['Lat'].apply(lambda x : round(x,4))
nic_dep['Long']= nic_dep['Long'].apply(lambda x : round(x,4))
nic_dep['Date'] = df_nic ['Date_c']  # Date Extracted From the Orginal Data
nic_dep.tail(2)


# In[ ]:


# Nicaragua map
NIC = folium.Map(location = [12.8654, -85.2072], tiles='OpenStreetMap', 
               min_zoom=7, max_zoom=8, zoom_start=7)
  
for (index,row) in nic_dep.iterrows():
    folium.Circle(
        radius=int(row.loc['Cases'])**7, # Go to the Row case and them bring the Confirmed case 
        location=[row.loc['Lat'], row.loc['Long']], # provide the lat and long according to index
        popup='The Waterfront',color='crimson',
        tooltip=    # this Will show the Legend of the markets 
        '<li><bold>Deparment : ' + str(row.loc['Deparments'])+  # HTML  <li><> is a bullet point
        '<li><bold>Confirmed : ' + str(row.loc['Cases']),   
        fill=True,fill_color='#ccfa00').add_to(NIC)
NIC


# **Conclusions**
# 1. The Given Data cannot be used to Create a Polinomial Regresion

# **Polynomial Regresion using USA data set**

# **COVID - 19 in USA Spread**

# In[ ]:


usa = df2.drop(['UID','iso2','iso3','Admin2','code3','FIPS','Combined_Key'], axis = 1)
# converting date into time format 
usa['Date'] =pd.to_datetime(usa['Date'])

# Grouping by State #The United States of America has 50 states, 1 Federal District, 5 Territories.
#District of Columbia(D.C) is a Federal District, not a state
SBD = usa[['Date','Confirmed','Deaths']].groupby('Date').sum()  #SD = Spread by Day

#Spread by day 
SBD.iplot(kind = 'spread', title = 'COVID-19 in USA')


# ****Confirmed / Deaths  by State****

# In[ ]:


CD = usa.groupby('Province_State')[['Confirmed','Deaths']].max()
CD.iplot(kind = 'barh', title = 'Confirmed / Death by State')


# In[ ]:


#Top 10 State Confirmed cases
USA_CONFIRMED = usa.groupby('Province_State',as_index= False)[['Confirmed']].max().sort_values('Confirmed') # USACC = USA CASE CONFIRMED
USA_CC = px.bar(USA_CONFIRMED.tail(15), x="Confirmed", y="Province_State",  text='Confirmed', orientation='h', color_discrete_sequence = ['Green'])

#Top 10 State Death 
USA_DEATH = usa.groupby('Province_State',as_index= False)[['Deaths']].max().sort_values('Deaths') # USACC = USA CASE CONFIRMED
USA_D = px.bar(USA_DEATH.tail(15), x="Deaths", y="Province_State",  text='Deaths', orientation='h', color_discrete_sequence = ['Blue'])

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,
                    subplot_titles=('USA COVID - 19 Confirmed Cases ', 'USA COVID - 19 Deaths'))

fig.add_trace(USA_CC['data'][0], row=1, col=1) #Remember hacerlo a un lista para que lo pueda leer
fig.add_trace(USA_D['data'][0], row=1, col=2)
fig.update_layout(height=700)


# **Data for Polynomial Regresion** 
# 1. in this section im trying to use polinomial regression why (because Im new in Machine learning and i Want to try) so any feedback it is more than welcome 

# In[ ]:


import datetime as dt
usa['Date_ordinal'] = usa['Date'].map(dt.datetime.toordinal) #changing Datetime to ordinal (I am not sure if it is the best way, but i couldnt find any other solution)
data = usa.drop(['Province_State','Lat','Long_','Date','Deaths','Country_Region'],axis = 1)
polydata = data.groupby('Date_ordinal',as_index = False).sum()
polydata.tail(5)


# In[ ]:


# importing Data Set  
y = polydata.iloc[:,1].values
X = polydata.iloc[:,0:1].values
#plotting for better understanding 
plt.scatter(X, y)
print('Cases per day')


# In[ ]:


#Splitting Data   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Transforming to degree 2 
poly_reg = PolynomialFeatures(degree = 2)   #because polynomial 2nd degree
X_train_poly = poly_reg.fit_transform(X_train)  # transform the data according to the degree selected
X_test_poly = poly_reg.fit_transform(X_test)

#Predicting 
PL =LinearRegression()
PL.fit(X_train_poly,y_train)
y_pred = PL.predict(X_test_poly)
y_pred


# In[ ]:


print("r2=" ,r2_score(y_test, y_pred))
print("Coef=", PL.coef_)


# **The accurancy of the model it is up to .98 % this could be because :**
# 1. The data follow and ascendent trend
# 2. This is the beggining so the curve is growing 
# 3. The distance between the data is not that significant (It is not disperse/shattered )

# **I create new Data to predict the new cases of COVID-19 until Jun 10**

# In[ ]:


X1 = np.arange (737570 , 737587) # arrange of numbers to pass to datetime
X1_pol = poly_reg.fit_transform(X1.reshape(-1, 1))
Predictions =PL.predict(X1_pol)


Till_Jun = pd.DataFrame(data = X1 , columns = ['Day'])
Prediction_jun = pd.DataFrame(Predictions , columns = ['Deaths'])
Jun = pd.concat([Till_Jun,Prediction_jun], axis = 1)
Jun ['Date'] = Jun['Day'].map(dt.datetime.fromordinal)
Jun.tail(3)


# In[ ]:


Jun.iplot(kind = 'bar' , x = 'Date' , y ='Deaths', title = 'Cases till Jun 10th')


# ****Conclusion** the model predict pretty well the new cases, the **https://www.worldometers.info/coronavirus/?utm_campaign=homeAdvegas1?
# 1. Showed  USA	2,026,597 total Cases
# 2. The Model Forcasts 2,514,077 total cases (+400K)
# 3. Thank you very much, I learn a lot with this Data set, Please provide any feddback or if i did something wrong please let me know. 

# In[ ]:




