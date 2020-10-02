#!/usr/bin/env python
# coding: utf-8

# # Tracing Covid19 Spread across USA & Europe - Data Analysis & Visualisation
# ![](https://www.genengnews.com/wp-content/uploads/2020/02/Feb27_2020_CDC_Coronavirus-1068x601.jpg)
# ![](https://media.arxiv-vanity.com/render-output/2249880/images/omni/XDL.png)
# ## Data Source
# ### 1. European Centre for Disease Prevention and Control
# The dataset contains the latest available public data on COVID-19 including a daily situation update, the epidemiological curve and the global geographical distribution (EU/EEA and the UK, worldwide). On 12 February 2020, the novel coronavirus was named severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) while the disease associated with it is now referred to as COVID-19. ECDC is closely monitoring this outbreak and providing risk assessments to guide EU Member States and the EU Commission in their response activities.
# #### Geographical Coverage
# Slovakia, Finland, Sweden, United Kingdom, Poland, Portugal, Romania, Slovenia, China, North Macedonia, Bulgaria, Belgium, Denmark, Czechia, Estonia, Germany, Greece, Ireland, France, Spain, Italy, Croatia, Latvia, Cyprus, Luxembourg, Lithuania, Malta, Hungary, Austria, Netherlands, Russia, Norway, Switzerland
# #### [Source Link](https://data.europa.eu/euodp/en/data/dataset/covid-19-coronavirus-data)
# ### 2. Data Repository by Johns Hopkins CSSE
# This is the data repository by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE).
# #### [Source Link](https://github.com/CSSEGISandData/COVID-19)

# # Download & Install Prerequisite 
# **Install:**
# * pycountry_convert 
# * folium

# In[ ]:


# Installs
get_ipython().system('pip install pycountry_convert ')
get_ipython().system('pip install folium')


# # Import Packages
# 
# * Pandas - for dataset handeling
# * Numpy - Support for Pandas and calculations
# * Matplotlib - for visualization (Platting graphas)
# * pycountry_convert - Library for getting continent (name) to from their country names
# * folium - Library for Map
# * Seaborn - for data visualisation
# * plotly - for interative plots
# * PandasProfiling - Python package to help understand data quickly

# In[ ]:


import pycountry_convert as pc
import folium
import branca
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling 
import matplotlib.pyplot as plt
from matplotlib import ticker 
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import json, requests
import seaborn as sns
sns.set()
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from datetime import datetime

from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam


# ## Load data

# In[ ]:


# Load Europe CDC data
df = pd.read_csv('/kaggle/input/covid19-european-cdc/download')

# Load Data Repository by Johns Hopkins CSSE
global_confirmed_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
global_deaths_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
#countrywise_cases = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_full = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])
df_full.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)


# ## Data Analysis
# 
# ### 1. Europe CDC Data Analysis

# In[ ]:


df.profile_report()


# # Preprocessing
# ## 1. Europe CDC :

# In[ ]:


df['Date'] = df.apply(lambda row: datetime.strptime(f"{int(row.month)}/{int(row.day)}/{int(row.year)}", '%m/%d/%Y'), axis=1)
df = df.rename(columns={"countriesAndTerritories": "Country","deaths": "Deaths","cases":"Cases"})
# Create Europe data subset
df_europe = df.copy()
countries = ["United_States_of_America","Slovakia", "Finland", "Sweden", "United Kingdom", "Poland", "Portugal", "Romania", "Slovenia", "North Macedonia", "Bulgaria", "Belgium", "Denmark", "Czechia", "Estonia", "Germany", "Greece", "Ireland", "France", "Spain", "Italy", "Croatia", "Latvia", "Cyprus", "Luxembourg", "Lithuania", "Malta", "Hungary", "Austria", "Netherlands", "Russia", "Norway", "Switzerland"]
df_europe= df_europe[df_europe.Country.isin(countries)]
# Filter based on Europe dataset
df_countries_cases = df_europe.copy().drop(['day','month','year','popData2018','dateRep'],axis =1)
df_countries_cases = df_countries_cases.groupby(["Country"]).sum()
df_countries_cases.sort_values(['Cases','Deaths'], ascending= False).style.background_gradient(cmap='Wistia')


# ## 2. John Hopkins:

# In[ ]:


global_confirmed_cases = global_confirmed_cases.rename(columns={"Province/State":"state","Country/Region": "country"})
# Create Europe data subset
df_europe_JK = global_confirmed_cases.copy()
countries = ["United_States_of_America","Slovakia", "Finland", "Sweden", "United Kingdom", "Poland", "Portugal", "Romania", "Slovenia", "North Macedonia", "Bulgaria", "Belgium", "Denmark", "Czechia", "Estonia", "Germany", "Greece", "Ireland", "France", "Spain", "Italy", "Croatia", "Latvia", "Cyprus", "Luxembourg", "Lithuania", "Malta", "Hungary", "Austria", "Netherlands", "Russia", "Norway", "Switzerland"]
df_europe_JK= df_europe_JK[df_europe_JK.country.isin(countries)]


# In[ ]:


# Filter based on Europe confirmed cases dataset
confirmed_cases_report = df_europe_JK.copy()
confirmed_cases_report = confirmed_cases_report.groupby("country").sum().drop(["Lat","Long"],axis =1)
confirmed_cases_report.loc["Total"] = confirmed_cases_report.sum()
global_confirmed_cases = confirmed_cases_report.groupby(level =0).diff(axis =1)
global_confirmed_cases = global_confirmed_cases.replace(np.nan, 0, regex=True)


# In[ ]:


global_deaths_cases.head()


# In[ ]:


global_deaths_cases = global_deaths_cases.rename(columns={"Province/State":"state","Country/Region": "country"})
# Create Europe data subset
df_europe_deaths = global_deaths_cases.copy()
countries = ["Slovakia", "Finland", "Sweden", "United Kingdom", "Poland", "Portugal", "Romania", "Slovenia", "North Macedonia", "Bulgaria", "Belgium", "Denmark", "Czechia", "Estonia", "Germany", "Greece", "Ireland", "France", "Spain", "Italy", "Croatia", "Latvia", "Cyprus", "Luxembourg", "Lithuania", "Malta", "Hungary", "Austria", "Netherlands", "Russia", "Norway", "Switzerland"]
df_europe_deaths= df_europe_deaths[df_europe_deaths.country.isin(countries)]


# In[ ]:


# Filter based on Europe deaths dataset
deaths_report = df_europe_deaths.copy()
deaths_report = deaths_report.groupby("country").sum().drop(["Lat","Long"],axis =1)
deaths_report.loc["Total"] = deaths_report.sum()
global_deaths_cases = deaths_report.groupby(level =0).diff(axis =1)
global_deaths_cases = global_deaths_cases.replace(np.nan, 0, regex=True)


# # Data Visualisation
# ## 1.Europe CDC:
# ### Top 10 Europe countries (Confirmed Cases and Deaths)

# In[ ]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)
plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Cases')["Cases"].index[-10:],df_countries_cases.sort_values('Cases')["Cases"].values[-10:],color="darkred")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)


# In[ ]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)
plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Deaths')["Deaths"].index[-10:],df_countries_cases.sort_values('Deaths')["Deaths"].values[-10:],color="darkorange")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Death Cases",fontsize=18)
plt.title("Top 10 Countries ( Death Cases)",fontsize=20)
plt.grid(alpha=0.3,which='both')


# ### Death Rate of Each European Country Over Time

# In[ ]:


df_data = df_europe.groupby(['Date', 'Country'])['Cases', 'Deaths'].max().reset_index()
df_data["Date"] = pd.to_datetime( df_data["Date"]).dt.strftime('%m/%d/%Y')

fig = px.scatter(df_data, y=df_data["Deaths"],
                    x= df_data["Cases"]+1,
                    range_y = [1,df_data["Deaths"].max()+1000],
                    range_x = [1,df_data["Cases"].max()+10000],
                    color= "Country", hover_name="Country",
                    hover_data=["Cases","Deaths"],
                    range_color= [0, max(np.power(df_data["Deaths"],0.3))], 
                    animation_frame="Date", 
                    animation_group="Country",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title='COVID-19: Change in Deaths vs Cases per Country Over Time',
                    size = np.power(df_data["Deaths"]+1,0.3)-0.5,
                    size_max = 30,
                    log_x=True,
                    log_y=True,
                    height =700,
                    )
fig.update_coloraxes(colorscale="hot")
fig.update(layout_coloraxis_showscale=False)
fig.update_xaxes(title_text="Confirmed Cases (Log Scale)")
fig.update_yaxes(title_text="Deaths Rate (Log Scale)")
fig.show()


# ## 2. John Hopkins Data:
# ### COVID-19 Confirmed Cases: Europe

# In[ ]:


f = plt.figure(figsize=(15,8))
ax1 = f.add_subplot(111)

ax1.bar(confirmed_cases_report[confirmed_cases_report.index == "Italy"].columns,global_confirmed_cases[global_confirmed_cases.index == "Italy"].values[0], label = "Italy (New)",color='dodgerblue')
ax1.bar(confirmed_cases_report[confirmed_cases_report.index == "Spain"].columns,global_confirmed_cases[global_confirmed_cases.index == "Spain"].values[0], bottom=global_confirmed_cases[global_confirmed_cases.index == "Spain"].values[0],label = "Spain (New)",color='orangered')

# Labels
ax1.set_xlabel("Dates",fontsize=17)
ax1.set_ylabel("New Cases Reported",fontsize =17)

ax1.tick_params(size=10,labelsize=15)
ax1.set_xticks(np.arange(0.5, len(confirmed_cases_report.columns), 6))
ax1.set_xticklabels([datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in confirmed_cases_report.columns][::6],fontsize=15)
ax1.set_yticks(np.arange(0, confirmed_cases_report.max(axis = 1)[2]/10+10000, 5000))


ax2 = ax1.twinx()
marker_style = dict(linewidth=6, linestyle='-', marker='o',markersize=10, markerfacecolor='#ffffff')

ax2.plot(confirmed_cases_report[confirmed_cases_report.index == "Total"].columns ,confirmed_cases_report[confirmed_cases_report.index == "Total"].values[0],**marker_style,label = "Europe Total (Cumulative)",color="darkorange",clip_on=False)
ax2.plot(confirmed_cases_report[confirmed_cases_report.index == "Italy"].columns ,confirmed_cases_report[confirmed_cases_report.index == "Italy"].values[0],**marker_style,label = "Italy (Cumulative)",color="limegreen",clip_on=False)
ax2.plot(confirmed_cases_report[confirmed_cases_report.index == "Spain"].columns ,confirmed_cases_report[confirmed_cases_report.index == "Spain"].values[0],**marker_style,label ="Spain (Cumulative)",color="darkviolet",clip_on=False)
ax2.bar([0],[0])

# Label
ax2.tick_params(labelsize=15)
ax2.set_ylabel("Cumulative",fontsize =17)
ax2.set_xticks(np.arange(0.5, len(confirmed_cases_report.columns), 6))
ax2.set_xticklabels([datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in confirmed_cases_report.columns][::6])
ax2.set_yticks(np.arange(0, confirmed_cases_report.max(axis = 1)[2]+50000, 50000))

f.tight_layout()
f.legend(loc = "upper left", bbox_to_anchor=(0.1,0.95))
plt.title("COVID-19 Confirmed Cases: Europe",fontsize = 22)
plt.show()


# ### COVID-19 Deaths across Europe

# In[ ]:


f = plt.figure(figsize=(15,8))
ax1 = f.add_subplot(111)

ax1.bar(deaths_report[deaths_report.index == "Italy"].columns,global_deaths_cases[global_deaths_cases.index == "Italy"].values[0], label = "Italy (New)",color='dodgerblue')
ax1.bar(deaths_report[deaths_report.index == "Spain"].columns,global_deaths_cases[global_deaths_cases.index == "Spain"].values[0], bottom=global_deaths_cases[global_deaths_cases.index == "Spain"].values[0],label = "Spain (New)",color='orangered')

# Labels
ax1.set_xlabel("Dates",fontsize=17)
ax1.set_ylabel("New Deaths Reported",fontsize =17)

ax1.tick_params(size=10,labelsize=15)
ax1.set_xticks(np.arange(0.5, len(deaths_report.columns), 6))
ax1.set_xticklabels([datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in deaths_report.columns][::6],fontsize=15)
ax1.set_yticks(np.arange(0, deaths_report.max(axis = 1)[2]/10+10000, 5000))


ax2 = ax1.twinx()
marker_style = dict(linewidth=6, linestyle='-', marker='o',markersize=10, markerfacecolor='#ffffff')

ax2.plot(deaths_report[deaths_report.index == "Total"].columns ,deaths_report[deaths_report.index == "Total"].values[0],**marker_style,label = "Europe Total (Cumulative)",color="darkorange",clip_on=False)
ax2.plot(deaths_report[deaths_report.index == "Italy"].columns ,deaths_report[deaths_report.index == "Italy"].values[0],**marker_style,label = "Italy (Cumulative)",color="limegreen",clip_on=False)
ax2.plot(deaths_report[deaths_report.index == "Spain"].columns ,deaths_report[deaths_report.index == "Spain"].values[0],**marker_style,label ="Spain (Cumulative)",color="darkviolet",clip_on=False)
ax2.bar([0],[0])

# Label
ax2.tick_params(labelsize=15)
ax2.set_ylabel("Cumulative",fontsize =17)
ax2.set_xticks(np.arange(0.5, len(deaths_report.columns), 6))
ax2.set_xticklabels([datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in deaths_report.columns][::6])
ax2.set_yticks(np.arange(0, deaths_report.max(axis = 1)[2]+50000, 50000))

f.tight_layout()
f.legend(loc = "upper left", bbox_to_anchor=(0.1,0.95))
plt.title("COVID-19 Death Cases: Europe",fontsize = 22)
plt.show()


# In[ ]:


df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]
df_temp = df_full.copy()
df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)
df_latlong = pd.merge(df_temp, df_confirmed, on=["Country", "Province/State"])


# ## Europe Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered

# In[ ]:


fig = px.density_mapbox(df_latlong, 
                        lat="Lat", 
                        lon="Long", 
                        hover_name="Country", 
                        hover_data=["Confirmed","Deaths","Recovered"], 
                        animation_frame="Date",
                        color_continuous_scale="Portland",
                        radius=1, 
                        zoom=2,height=700)
fig.update_layout(title='Europe Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered',
                  font=dict(family="Courier New, monospace",
                            size=18,
                            color="#7f7f7f")
                 )
fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


fig.show()


# ## Prediction Model:
# 

# In[ ]:


# Visible = Input(shape=(1,))
# Dense_l1 = Dense(80,name="Dense_l1")(Visible)
# LRelu_l1 = LeakyReLU(name = "LRelu_l1")(Dense_l1)
# Dense_l2 = Dense(80,name = "Dense_l2")(LRelu_l1)
# LRelu_l2 = LeakyReLU(name = "LRelu_l2")(Dense_l2)
# Dense_l3 = Dense(1,name="Dense_l3")(LRelu_l2)
# LRelu_l3 = LeakyReLU(name = "Output")(Dense_l3)
# model = models.Model(inputs=Visible, outputs=LRelu_l3)
# model.compile(optimizer=Adam(lr=0.001), 
#               loss='mean_squared_error',
#               metrics=['accuracy'])
# model.summary()


# In[ ]:


# epochs = 500
# model.fit(data_x.reshape([data_y.shape[0],1]),data_y.reshape([data_y.shape[0],1]),epochs=epochs)


# ## Please note that this notebook is work in progress. So far I hope you liked this post covering COVID 19 spread across Europe.Greatly appreciate to leave your comments/views/upvote.
# 
# ### Keep watching my posts in the coming days .

# In[ ]:




