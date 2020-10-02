#!/usr/bin/env python
# coding: utf-8

# # Storytelling COVID-19
# 
# Initially, I just wanted to know how bad the situation in my country, the Netherlands, was compared to other countries as we had high numbers of confirmed cases and only 17 million inhabitants. In order to do so, I used the [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset) and I uploaded my own dataset: [Countries of the World; ISO codes and population](https://www.kaggle.com/erikbruin/countries-of-the-world-iso-codes-and-population), and made plotly world maps based on the numbers of confirmed cases and deaths per million inhabitants.
# 
# After having done that, the pandemic unfortunately became worse and worse, and I wanted to investigate more. This version includes time series trends and also time series trends that start at the date of the first confirmed case/death reported ("Day Zero"). On April 12th, I wanted to add more insight on the situation in the US, and therefore added dataset [COVID-19 US County JHU Data & Demographics](https://www.kaggle.com/headsortails/covid19-us-county-jhu-data-demographics).
# 
# **Observations on May 5th**
# *(Please be aware that this may be different if you fork and run this kernel, as the dataset is updated daily)*
# 
# - When looking at the number of death per million, Belgium is now the country with the highest number of Deaths per Million. However, if we would treat the US states as countries, the ranking would be as follows:
#     - 1. New York State, 2. New Jersey 3. Belgium, 4. Connecticut, 5. Massachusetts, 6. Spain, 7 Italy
# - The relative Death rates of New York are terrible. While countries such as Belgium, Italy and Spain have rates around 500-700 Death per Million, the state of New York has 1259 Deaths per Million , and New York City has 2242 Death per Million!
# 
# - Qatar is the country with most confirmed cases per million inhabitants. Most other countries in the "top" 10  of countries with the highest numbers of cases per million inhabitants are Western European countries, with the US at 5th place.
# 
# - The US is the country with the highest (absolute) number of confirmed cases and unfortunately the trend is steep upward. Over the past weeks the US has also passed Spain and Italy regarding the total number of Deaths.
# 
# - When looking the the time series of the cumulative numbers of deaths while taking the day of the first reported death as "Day Zero", we can see that China managed to flatten the curve while in Italy and Spain the number of victims really exploded after 15-20 days after the first casualty. Although more "delayed", the trend is now looking very bad for especially the US.
# 
# - Belarus, Saudi Arabia and United Arab Emirates are doing best when looking at the number of deadly victims relative to the number of confirmed cases.
# 
# # Table of contents
# * [1. Corona figures relative to country population](# 1.Corona-figures-relative-to-country-population)
#   * [1.1 Adding country population to the COVID-19 figures](#1.1-Adding-country-population-to-the-COVID-19-figures)
#   * [1.2 "Top" 10 countries with relatively most confirmed cases](#1.2-"Top"-10-countries-with-relatively-most-confirmed-cases)
#   * [1.3 World map with Cases per Million for each country](#1.3-World-map-with-Cases-per-Million-for-each-country)
#   * [1.4 "Top" 10 countries with relatively most deaths](#1.4-"Top"-10-countries-with-relatively-most-deaths)
#   * [1.5 World map with Deaths per Million for each country](#1.5-World-map-with-Deaths-per-Million-for-each-country)
# * [2. Bubble charts](#2.-Bubble-charts)
#   * [2.1 World map: Bubble chart showing Confirmed Cases by Province/state](#2.1-World-map:-Bubble-chart-showing-Confirmed-Cases-by-Province/state)
#   * [2.2 World map: Bubble chart showing Deaths by Province/state](#2.2-World-map:-Bubble-chart-showing-Deaths-by-Province/state)
# * [3. Time series plots](#3.-Time-series-plots)
#   * [3.1 Time series plot of the countries with most Confirmed cases](#3.1-Time-series-plot-of-the-countries-with-most-Confirmed-cases)
#   * [3.2 Time series plot of the countries with most Deaths](#3.2-Time-series-plot-of-the-countries-with-most-Deaths)
#   * [3.3 Time series plot of Deaths since day of first victim](#3.3-Time-series-plot-of-Deaths-since-day-of-first-victim)
# * [4. Deaths relative to the number of confirmed cases](#4.-Deaths-relative-to-the-number-of-confirmed-cases)
# * [5. US figures](#5.-US-figures)
#   * [5.1 US figures by state](#5.1-US-figures-by-state)
#   * [5.2 Relative Deaths by country when treating US states as countries too](#5.2-Relative-Deaths-by-country-when-treating-US-states-as-countries-too)
#   * [5.3 New York City figures](#5.3-New-York-City-figures)

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

import matplotlib.style as style
style.use('fivethirtyeight')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 150 #set figure size

from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
#import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

import folium

#df = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
#COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates=['Last Update'])
df.rename(columns={'Country/Region':'Country'}, inplace=True)
df = df.drop(columns = ['SNo', "Last Update"]) #only confuses

df_conf = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_conf.rename(columns={'Country/Region':'Country'}, inplace=True)

df_death = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
df_death.rename(columns={'Country/Region':'Country'}, inplace=True)
# time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
countries = pd.read_csv("../input/countries-of-the-world-iso-codes-and-population/countries_by_population_2019.csv")
countries_iso = pd.read_csv("../input/countries-of-the-world-iso-codes-and-population/country_codes_2020.csv")

us_covid = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/covid_us_county.csv')
us_county = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/us_county.csv')


# # 1. Corona figures relative to country population
# 
# ## 1.1 Adding country population to the COVID-19 figures
# 
# File "covid_19_data.csv" from dataset "Novel Corona Virus 2019 Dataset" contains info by day. Below, you can see a sample of this info. In most cases, the info is provided on the country level. However, especially for some large countries such as China and the US, numbers are specified on the Province/State level.

# In[ ]:


df.sample(5)


# What I am looking for is the most recent, overall numbers for each country. If I just keep the most recent line, I do get the most recent, cumulative numbers per State/Country. All I have to do then is to consolidate those per Country (for instance add up all numbers of the US States).
# 
# Dataset [Countries of the World; ISO codes and population](https://www.kaggle.com/erikbruin/countries-of-the-world-iso-codes-and-population), uploaded by me, contains the number of inhabitants by country. As I want to use built-in geometries of plotly.express later on to plot numbers on a world map, I am also merging the three-letter ISO country code to the dataframe.
# 
# This enables me to add the population numbers and ISO codes to the dataframe with Corona figures by country, and calculate the 'Cases per million inhabitants' and 'Deaths per million inhabitants'. Below you can see a sample of the resulting dataframe.

# In[ ]:


#strip white spaces are there is one country (Azerbaijan) with a whitespace observation
df['Country'] = df['Country'].str.strip()

#fill missing Province/State with Country
df.loc[df['Province/State'].isnull(), 'Province/State'] = df.loc[df['Province/State'].isnull(), 'Country']

#keep most recent line per Province/State and Country
df.sort_values(['Country', 'Province/State', 'ObservationDate'], ascending = [True,True,False], inplace = True)
df = df.drop_duplicates(['Country', 'Province/State'], keep = "first")

#keep a copy for later on
df_state = df.copy()

df = df.drop(columns = "ObservationDate")

#groupby Country
df_country = df.groupby(['Country'], as_index=False)['Confirmed', 'Deaths'].sum()

#drop some columns
cols_to_drop = ['Rank', 'pop2018','GrowthRate', 'area', 'Density']
countries = countries.drop(columns = cols_to_drop)

#add ISO Alpha 3 code that I uploaded in another CSV
countries = countries.merge(countries_iso[['name', 'cca3']], on = ['name'], how = "left")

cols_to_rename = {'name': 'Country', 'pop2019': 'Population', 'cca3': 'ISO'}
countries = countries.rename(columns = cols_to_rename)

#just fixing the most important mismatches
countries_to_rename = {'US': 'United States',                       'Mainland China': 'China',                       'UK': 'United Kingdom',                       'Congo (Kinshasa)': 'DR Congo',                       'North Macedonia': 'Macedonia',                       'Republic of Ireland': 'Ireland',                       'Congo (Brazzaville)': 'Republic of the Congo'}

df_country['Country'] = df_country['Country'].replace(countries_to_rename)

df_country = df_country.merge(countries, on = "Country", how = "left")

#check mismatches
#df_country[df_country.ISO.isnull()].sort_values(['Confirmed'], ascending = False)

#dropping not matching countries, only small islands left
df_country = df_country.dropna()

#rounding population to millions with 2 digits, and creating two new columns
df_country['Population'] = round((df_country['Population']/1000),2)
df_country = df_country.rename(columns = {'Population': 'Population (million)'})
df_country['Cases per Million'] = round((df_country['Confirmed']/df_country['Population (million)']),2)
df_country['Deaths per Million'] = round((df_country['Deaths']/df_country['Population (million)']),2)

#filter out countries with less than a million population as for instance San Marino has extremely high figures on a very small population
df_country = df_country[(df_country['Population (million)'] > 1)]

df_country.sample(5)


# # 1.2 "Top" 10 countries with relatively most confirmed cases
# 
# Final relative ranking sorted on 'Cases per Million' (10 countries with most cases per million only). In this list, countries with less than a million inhabitants are excluded.
# 
# On May 5th, Qatar is the country with most confirmed cases per million inhabitants. Most other countries in the "top" 10 of countries with the highest numbers of cases per million inhabitants are Western European countries, with the US at 5th place.

# In[ ]:


df_country = df_country.sort_values(['Cases per Million'], ascending = False).reset_index(drop=True)
df_country.drop(columns = ['ISO', 'Deaths', 'Deaths per Million']).head(10).style.background_gradient(cmap='Reds', subset = ['Cases per Million'])


# ## 1.3 World map with Cases per Million for each country
# 
# **Hovering over the map below shows the info in a tooltip. You can also use the plotly icons to zoom in at for instance Europe**
# 
# This map is bigger and therefore better after running in the editor. Somehow, the width of the rendered notebooks is small on Kaggle. Please let me know if you know a way to increase this width!

# In[ ]:


fig = px.choropleth(df_country, locations="ISO",
                    color="Cases per Million",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.YlOrRd)

layout = go.Layout(
    title=go.layout.Title(
        text="Corona confirmed cases per million inhabitants",
        x=0.5
    ),
    font=dict(size=14),
    width = 750,
    height = 350,
    margin=dict(l=0,r=0,b=0,t=30)
)

fig.update_layout(layout)

fig.show()


# ## 1.4 "Top" 10 countries with relatively most deaths
# 
# Final relative ranking sorted on 'Deaths per Million'. Again, countries with less than a million inhabitants are excluded in this list.
# 
# Belgium has now passed the Spain and Italy, the countries with most Deaths per Million in the past weeks.
# 
# 

# In[ ]:


df_country = df_country.sort_values(['Deaths per Million'], ascending = False).reset_index(drop=True)

#save a copy
countries = df_country.copy()

df_country.drop(columns = ['ISO', 'Confirmed', 'Cases per Million']).head(10).style.background_gradient(cmap='Reds', subset = ['Deaths per Million'])


# ## 1.5 World map with Deaths per Million for each country
# 
# **Hovering over the map below shows the info in a tooltip. You can also use the plotly tool to zoom in at for instance Europe**

# In[ ]:


fig = px.choropleth(df_country, locations="ISO",
                    color="Deaths per Million",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.YlOrRd)

layout = go.Layout(
    title=go.layout.Title(
        text="Corona deaths per million inhabitants",
        x=0.5
    ),
    font=dict(size=14),
    width = 750,
    height = 350,
    margin=dict(l=0,r=0,b=0,t=30)
)

fig.update_layout(layout)

fig.show()


# # 2. Bubble charts
# 
# As mentioned before, for some large countries such as China, the information is collected on the Province/State level. In the maps with the numbers per million inhabitants, China did not come out with very high averages. However, we all know that Hubei had very high numbers and I want to give some insight on this using a Bubble Chart.
# 
# ## 2.1 World map: Bubble chart showing Confirmed Cases by Province/state
# 
# The CSV with the time series of confirmed cases contains coordinates that I can use to plot on a map, but I first want to check if the file is as up-to-date as the dataframe that I have used so far. To do so, I am only displaying the last 5 columns added and filtering on the Netherlands. It turns out that this file is updated less frequently. Depending on the day that I run this kernel the last column may be a couple of days back. Also, the last day/column is not always updated for each country (last column same numbers as day before).

# In[ ]:


#get names of first 4 and last 5 columns
cols_to_select = list(df_conf.columns[0:4]) + list(df_conf.columns[-6:])
df_conf.loc[(df_conf['Country'] == "Netherlands"), cols_to_select]


# Below I am preparing a dataframe in which I only keep the last date.

# In[ ]:


#only keep last date available
cols_to_keep = list(df_conf.columns[0:4]) + list(df_conf.columns[-1:])
df_conf_last = df_conf[cols_to_keep]
df_conf_last.columns.values[-1] = "Confirmed"

df_conf_last.head()


# Below you can see the resulting bubble chart. The map is centered on China, but it is in fact a world map. You can for instance also move towards the Caribean to see the figures for some small islands that belong to countries such as France or the Netherlands.

# In[ ]:


#float required
df_conf_last['Confirmed'] = df_conf_last['Confirmed'].astype(float)

map1 = folium.Map(location=[30.6, 114], zoom_start=3) #US=[39,-98] Europe =[45, 5]

for i in range(0,len(df_conf_last)):
   folium.Circle(
      location=[df_conf_last.iloc[i]['Lat'], df_conf_last.iloc[i]['Long']],
      tooltip = "Country: "+df_conf_last.iloc[i]['Country']+"<br>Province/State: "+str(df_conf_last.iloc[i]['Province/State'])+"<br>Confirmed cases: "+str(df_conf_last.iloc[i]['Confirmed'].astype(int)),
      radius=df_conf_last.iloc[i]['Confirmed']*5,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(map1)

map1


# ## 2.2 World map: Bubble chart showing Deaths by Province/state

# Now, I am making the same visuals for the numbers of Deaths by country.

# In[ ]:


#only keep last date available
cols_to_keep = list(df_death.columns[0:4]) + list(df_death.columns[-1:])
df_death_last = df_death[cols_to_keep]
df_death_last.columns.values[-1] = "Death"

#float required
df_death_last['Death'] = df_death_last['Death'].astype(float)

map2 = folium.Map(location=[30.6, 114], zoom_start=3)

for i in range(0,len(df_death_last)):
   folium.Circle(
      location=[df_death_last.iloc[i]['Lat'], df_death_last.iloc[i]['Long']],
      tooltip = "Country: "+df_death_last.iloc[i]['Country']+"<br>Province/State: "+str(df_death_last.iloc[i]['Province/State'])+"<br>Deaths: "+str(df_death_last.iloc[i]['Death'].astype(int)),
      radius=df_death_last.iloc[i]['Death']*100,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(map2)

map2


# # 3. Time series plots
# 
# ## 3.1 Time series plot of the countries with most Confirmed cases
# 
# Below, you can see this time series for the Top-x countries with most confirmed cases. Nothing is hard-coded, so the Top-x (x is specified in head(x) in the code) may show different countries if I run the notebook again tomorrow.
# 
# The most noticable country is now the US, as the number of confirmed cases shows a very steep upward curve and the US is now the country with the highest (absolute) number of confirmed cases.

# In[ ]:


ts_country = df_conf.drop(columns = ['Lat', 'Long', 'Province/State'])
ts_country = ts_country.groupby(['Country']).sum()

#get countries with most cases on last date in dataframe
ts_country = ts_country.sort_values(by = ts_country.columns[-1], ascending = False).head(7)
#drop last date as not always updated
#ts_country.drop(ts_country.columns[len(ts_country.columns)-1], axis=1, inplace=True)

ts_country.transpose().iplot(title = 'Time series of confirmed cases of countries with most confirmed cases')


# ## 3.2 Time series plot of the countries with most Deaths
# 
# Unfortunately, the steep upward trend of most countries is much worse than China and therefore very worrying. Especially for the US, the trend is steep upward over the past weeks.

# In[ ]:


ts_country = df_death.drop(columns = ['Lat', 'Long', 'Province/State'])
ts_country = ts_country.groupby(['Country']).sum()

#get countries with most cases on last date in dataframe
ts_country = ts_country.sort_values(by = ts_country.columns[-1], ascending = False).head(7)
#drop last date as not always updated
ts_country.drop(ts_country.columns[len(ts_country.columns)-1], axis=1, inplace=True)

ts_country.transpose().iplot(title = 'Time series of deaths of countries with most victims')


# ## 3.3 Time series plot of Deaths since day of first victim
# 
# As you can see, China managed to flatten the curve while in Italy and Spain the number of victims really exploded after 15-20 days after the first casualty. In addition, although more "delayed", the trend is also looking bad for countries such as France, the UK, and especially the US.

# In[ ]:


ts_country = ts_country.transpose()

df1 = ts_country.iloc[:,0].to_frame()
df1 = df1[df1.iloc[:,0] !=0].reset_index(drop=True)

for i in range(1,ts_country.shape[1]):
    df = ts_country.iloc[:,i].to_frame()
    df = df[df.iloc[:,0] !=0].reset_index(drop=True)
    df1 = pd.concat([df1, df], join='outer', axis=1)

    
df1.iplot(title = 'Time series of deaths since first victim', xTitle = 'Days since first reported Death', yTitle = 'Number of Deaths')


# # 4. Deaths relative to the number of confirmed cases
# 
# Tables are filtered on countries with at least 100 Deaths. Italy has the highest ratio, and countries like Algeria and Mexico are somewhat surprisingly in the Top 10. I assume that this may be due to relatively low testing activity.

# In[ ]:


df_country = df_country.drop(columns = ['Population (million)', 'ISO', 'Cases per Million', 'Deaths per Million'])
df_country['Percent Death'] = round(((df_country.Deaths / df_country.Confirmed)*100),2)
#filter countries with at least 100 deaths
df_country = df_country[(df_country.Deaths >= 100)]

#set font size for plotting
#plt.rcParams.update({'font.size': 12})

#create barplot
se = df_country[['Country', 'Percent Death']].sort_values(by = "Percent Death", ascending = False).set_index("Country")
se = se[0:10].sort_values(by = "Percent Death", ascending = True)
se.plot.barh()
plt.title("Countries with worst ratio confirmed cases vs. Deaths")
plt.xticks(rotation=0);


# Belarus, Saudi Arabia and United Arab Emirates are doing best when looking at the number of deadly victims relative to the number of confirmed cases. Regarding the rich countries, Saudi Arabia and the United Arab Emirates, I suspect that this may be due to a high level of testing.

# In[ ]:


#create barplot
se = df_country[['Country', 'Percent Death']].sort_values(by = "Percent Death", ascending = False).set_index("Country")
se = se[-10:]
se.plot.barh()
plt.title("Countries doing best regarding confirmed cases vs. Deaths")
plt.xticks(rotation=0);


# # 5. US figures
# 
# ## 5.1 US figures by state
# 
# As New York is mentioned in the news as the new epicentre of the pandemic, I wanted to look into this is more detail. When I started this kernel, the 'Novel Corona Virus 2019 Dataset' had numbers by US county. However, as you can see in my Bubble charts, those are not maintained anymore. Fortunately, Heads or Tails uploaded a dataset on the US that he updates daily: [COVID-19 US County JHU Data & Demographic](COVID-19 US County JHU Data & Demographic).
# 
# The dataset consists of two csv's and a shapefile of the US counties. In the next version, I will use the shapefile to make maps, but for now I will focus on extracting some key figures. The cases and deaths are cumulative. Therefore, I will only keep the most recent date for each county/state combination.

# In[ ]:


#fips of 2 counties are missing (Dukes and Nantucket, Kansas City)
#quick fix for now
us_covid = us_covid[us_covid.fips.notnull()]
us_covid['fips'] = us_covid['fips'].astype(object)
us_county['fips'] = us_county['fips'].astype(object)

#add popultation from second csv
us_covid = us_covid.merge(us_county[['fips', 'population']], on = ['fips'], how = "left")

#keep latest date only
us_cum = us_covid.sort_values(by = ['county', 'state', 'date'], ascending = [True, True, False])
us_cum = us_cum.drop_duplicates(subset = ['county', 'state'], keep = "first")

#save a copy
counties_us = us_cum.copy()

#groupby State
us_cum = us_cum.groupby(['state', 'date'], as_index=False)['cases', 'deaths', 'population'].sum()

us_cum['population'] = us_cum['population'].astype(int)

#rounding population to millions with 2 digits, and creating two new columns
us_cum['population'] = round((us_cum['population']/1000000),2)
us_cum = us_cum.rename(columns = {'population': 'Population (million)'})
us_cum['Cases per Million'] = round((us_cum['cases']/us_cum['Population (million)']),2)
us_cum['Deaths per Million'] = round((us_cum['deaths']/us_cum['Population (million)']),2)

#remove states with missing population
us_cum = us_cum[(us_cum['Population (million)'] != 0)]


# As you can see, the state of New York has 1259 Deaths per Million. This is worse than the country with most Deaths per Million!

# In[ ]:


us_cum = us_cum.sort_values(by = "Deaths per Million", ascending = False).reset_index(drop=True)
us_cum.head(10).style.background_gradient(cmap='Reds', subset = ['Deaths per Million'])


# Below you can see the Deaths per Million on a map.

# In[ ]:


url = '../input/usa-states'
state_geo = f'{url}/usa-states.json'

bins = list(us_cum['Deaths per Million'].quantile([0, 0.5, 0.75, 0.90, 0.95, 1]))

map3 = folium.Map(location=[37, -102], zoom_start=4)

choropleth = folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=us_cum,
    columns=['state', 'Deaths per Million'],
    key_on='properties.name',
    fill_color= 'YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Deaths per Million',
    bins = bins,
    reset = True
).add_to(map3)

style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],style=style_function, labels=False)
)

map3


# ## 5.2 Relative Deaths by country when treating US states as countries too

# In[ ]:


us_cum = us_cum[['state', 'deaths', 'Population (million)', 'Deaths per Million']]
us_cum = us_cum.rename(columns = {'state': 'Country or US State', 'deaths': 'Deaths'})

countries = countries.drop(columns = ['ISO', 'Confirmed', 'Cases per Million'])
countries = countries.rename(columns = {'Country': 'Country or US State'})

countries = pd.concat([us_cum, countries], ignore_index = True)
countries = countries.sort_values(by = 'Deaths per Million', ascending = False).reset_index(drop=True)
countries.head(10).style.background_gradient(cmap='Reds', subset = ['Deaths per Million'])


# ## 5.3 New York City figures
# 
# The next thing that I wanted to do is to dive deeper into the figures of New York City, which consists of 5 boroughs: Manhattan, Queens, the Bronx, Brooklyn and Staten Island. However, the county names of the counties of those boroughs are in some cases different:
# 
# * Manhattan: New York County
# * Brooklyn: Kings County
# * Queens: Queens County
# * Bronx: Bronx County
# * Staten Island: Richmond County
# 
# Unfortunately, as you can see, all cases and deaths have been assigned to county New York, which actually is Manhattan!
# 

# In[ ]:


nyc_counties = ['New York', 'Kings', 'Queens', 'Bronx', 'Richmond']
new_york = counties_us[((counties_us.state == "New York") & (counties_us.county.isin(nyc_counties)))].sort_values(by="fips")
new_york


# Therefore, unfortunately no insights yet on the figures by borough. All I can do for now is what the total numbers for NYC look like. As you can see, the Deaths per Million is 2242, which is again way worse than the state of New York average.

# In[ ]:


nyc = new_york.groupby(['state', 'date'])['cases', 'deaths', 'population'].sum()
nyc.index.names = ['city', 'date']

nyc['population'] = nyc['population'].astype(int)

#rounding population to millions with 2 digits, and creating two new columns
nyc['population'] = round((nyc['population']/1000000),2)
nyc = nyc.rename(columns = {'population': 'Population (million)'})
nyc['Cases per Million'] = round((nyc['cases']/nyc['Population (million)']),2).astype(int)
nyc['Deaths per Million'] = round((nyc['deaths']/nyc['Population (million)']),2).astype(int)

nyc

