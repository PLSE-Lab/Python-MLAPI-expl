#!/usr/bin/env python
# coding: utf-8

# ## Goals
# 
# * This analysis is based on the data released by [__Protezione Civile__](http://opendatadpc.maps.arcgis.com/apps/opsdashboard/index.html#/b0c68bce2cce478eaac82fe38d4138b1) on COVID-19 cases in Italy. It visualizes data on cumulative cases, new positive cases, deaths and measures such as the number of performed tests, hospitalized patients and recovered ones. Data are then stratified for the 20 administrative regions composing Italy and for the 12 provinces of Lombardy, the worst affected region. 
# * To report cases on 10,000 inhabitants, information on region and province population has been retrieved from [__ISTAT website__](https://www.istat.it/en/population-and-households?data-and-indicators).
# * Geojson files have been retrieved from [this repo](https://gist.github.com/datajournalism-it/)
# 
# This analysis tries to offer insight about these questions: 
#  * How is the shape of Italian curves for total positive cases and new cases?
#  * How are they related to measures such as the number of performed tests, hospitalized patients, ICU patients and deaths?
#  * From a temporal perspective, which are the regions that have been most affected by the outbreak?
#  * How do Lombardy numbers and % compare to national ones?
#  * In Lombardy, where did the outbreak start, and how did it evolve?
# 
# #### Note
# Source data currently updated frequently; the comments of the analysis refer to the situation as of mid-April. _All analysis conclusions are subjective._
# Information for number of tsted subjects is available only from mid-April.

# *****

# ## Index
# 
# 
# [Set Up](#SetUp)
# 
# [Reading and Reshaping Data](#Reading-and-reshaping-data)
# 
# 
# [1. Parameters at the national level](#1.-Parameters-at-the-national-level)
# * [1.1 Total and New Positive Cases: trends in Italy](#1.1-Total-and-New-Positive-Cases:-trends-in-Italy)
# * [1.2 Total Positive Cases and related measures](#1.2-Total-Positive-Cases-and-related-measures)
# 
# [2. Which Regions have been affected more harshly?](#2.-Which-Regions-have-been-affected-more-harshly?)
# * [2.1 Looking at cumulative and new cases today, 4 weeks ago and 8 weeks ago](#2.1-Looking-at-cumulative-and-new-cases-today,-4-weeks-ago-and-8-weeks-ago)
# * [2.2 Stratification for regions: current situation](#2.2-Stratification-for-regions:-current-situation)
# * [2.3 The time dimension](#2.3-The-time-dimension)
# 
# 
# [3. Lombardy Contribution to National Trends](#3.-Lombardy-Contribution-to-National-Trends)
# * [3.1 National and Regional -Lombardy- breakdown of positive cases](#3.1-National-and-Regional--Lombardy--breakdown-of-positive-cases)
# * [3.2 National and Regional -Lombardy- breakdown of current cases](#3.2-National-and-Regional--Lombardy--breakdown-of-current-cases)
# * [3.3 Comparing Italy and Lombardy curves](#3.3-Comparing-Italy-and-Lombardy-curves)
# 
# 
# [4. Zooming in: Lombardy provinces](#4.-Zooming-in:-Lombardy-provinces)
# * [4.1 Breakdown of Lombardy cumulative cases for province](#4.1-Breakdown-of-Lombardy-cumulative-cases-for-province)
# * [4.2 Contribution of each province today and 8 weeks ago](#4.2-Contribution-of-each-province-today-and-8-weeks-ago)
# * [4.3 Dynamic view through time](#4.3-Dynamic-view-through-time)

# *****

# ## SetUp

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sn

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import date, timedelta
# %matplotlib inline

import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py
import requests

import json
#from urllib.request import urlopen

# COLOR PALETTE
tot, hos, icu, dea, rec, new, test, cur = '#4d4dff', '#fe9801', '#ff4d4d',  '#393e46', '#21bf73', '#9801fe', '#00b3b3', '#ff3300'


# *****

# ## Reading and reshaping data
# 
# ### Italy Covid-19 data at the regional and province level
# 
# **Main Available information at the regional level:**
#    * Date
#    * RegionName
#    * HospitalizedPatients; IntensiveCarePatients; HomeConfinement
#    * TotalPositiveCases; NewPositiveCases; CurrentPositiveCases
#    * Recovered; Deaths; TestPerformed (available only from mi-april onward)
#  
# I also calculate: 
#    * % of deaths on total cases _(Deaths / TotalPositiveCases)_
#    * % of positives on total performed tests _(TotalPositiveCases / TestPerformed)_
#    * for each region: TotalPositiveCases, NewPositiveCases and Deaths every 10,000 inhabitants
# 
# **At the local (province) level only the __TotalPositiveCases__ are reported.**
# I calculate for each province TotalPositiveCases every 10,000 inhabitants.

# In[ ]:


# RegData: data at the regional level
RegData = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
RegData.head()
#RegData.shape
#RegData.columns.values
#RegData.describe()
RegData['Day'] = pd.to_datetime(RegData['Date']).dt.date # Set variable Day
RegData['DeathOnTot'] = round(RegData['Deaths'] / RegData['TotalPositiveCases']*100, 2) # Calculate % deaths on total cases
RegData['PosOnTests'] = round(RegData['TotalPositiveCases'] / RegData['TestsPerformed']*100, 2) 

# Add information on population 
# From: dati.istat.it/Index.aspx?QueryId=18548
PopRegDict = {'Piemonte':4356406, "Valle d'Aosta":125666, "Liguria":1550640, 'Lombardia':10060574 , 'P.A. Bolzano':531178, 'P.A. Trento':541098, 
          'Veneto':4905854, 'Friuli Venezia Giulia':1215220, 'Emilia-Romagna':4459477, 'Toscana':3729641, 'Umbria':882015, 'Marche':1525271, 
          'Lazio':5879082, 'Abruzzo':1311580, 'Molise':305617, 'Campania':5801692,'Puglia':4029053, 'Basilicata':562869, 'Calabria':1947131, 
           'Sicilia':4999891, 'Sardegna':1639591}

RegData['Population'] = RegData.apply(lambda x: PopRegDict[x.RegionName], axis=1)

# Calculate cases for 10000 inhabitants
RegData['TotalOnPop'] = round(RegData.TotalPositiveCases/RegData.Population * 10000, 2)
RegData['NewOnPop'] = round(RegData.NewPositiveCases/RegData.Population * 10000, 2)
RegData['DeaOnPop'] = round(RegData.Deaths/RegData.Population * 10000, 2)

RegData.iloc[:,[17,4,11,12,13,14,15,16,18,19,21,22]].tail(10)


# In[ ]:


# ProvData: data at the province level. I select only data for Lombardia
ProvData = pd.read_csv("../input/covid19-in-italy/covid19_italy_province.csv")
ProvData = ProvData.loc[(ProvData.RegionName=='Lombardia') & (ProvData.ProvinceName != 'In fase di definizione/aggiornamento')]
ProvData['Day'] = pd.to_datetime(ProvData['Date']).dt.date

# Add information on population 
# From: dati.istat.it/Index.aspx?QueryId=18548
PopDict = {'Bergamo':1114590, 'Brescia':1265954, 'Como':599204, 'Cremona':358955 , 'Lecco':337380, 'Lodi':230198, 
          'Mantova':412292, 'Milano':3250315, 'Monza e della Brianza':873935, 'Pavia':545888, 'Sondrio':181095, 'Varese':890768}

ProvData['Population'] = ProvData.apply(lambda x: PopDict[x.ProvinceName], axis=1)

# Calculate cases for 10000 inhabitants
ProvData['TotalOnPop'] = round(ProvData.TotalPositiveCases/ProvData.Population * 10000, 2)
#ProvData.iloc[:,[11,6,10,13]].tail(8)


# ### Aggregation at the national level and selection of Lombardy data
# The aggregated data at the national level is retrieved for the following variables: __HospitalizedPatients; IntensiveCarePatients; TotalPositiveCases; NewPositiveCases; CurrentPositiveCases; Recovered; Deaths; TestPerformed.__

# In[ ]:


#RegData.columns.values
National = RegData.groupby(['Day', 'Country'])['TotalPositiveCases', 'TotalHospitalizedPatients', 'HomeConfinement', 'IntensiveCarePatients', 
                                                'Deaths', 'Recovered', 'TestsPerformed', 'NewPositiveCases', 'CurrentPositiveCases'].sum().reset_index()
National['DeathOnTot'] = round(National['Deaths'] / National['TotalPositiveCases']*100, 2)
National['PosOnTests'] = round(National['TotalPositiveCases'] / National['TestsPerformed']*100, 2)
#National.head()

Lombardia = RegData.loc[RegData.RegionName=='Lombardia', :] #Lombardia.head()

#National.iloc[:, [0,2,6, 7, 8, 9, 11, 12]].tail()
#National.tail()


# ### Time: today, one month ago and a two months ago
# I select the information relative to the __latest available day__ and the one relative to __4 and 8 weeks before__. 

# In[ ]:


Today = max(RegData['Day']) #print(type(Today))
Past = (Today-timedelta(days=28)) #print(type(Past))
Past8 = (Today-timedelta(days=56)) #print(type(Past))
RegToday = RegData[RegData['Day']==Today].reset_index() #RegToday.head()
RegPast = RegData[RegData['Day']==Past].reset_index() #RegPast.head()
RegPast8 = RegData[RegData['Day']==Past8].reset_index() #RegPast.head()
ProvToday = ProvData[ProvData['Day']==Today].reset_index() #RegToday.head()
ProvPast = ProvData[ProvData['Day']==Past].reset_index() #RegPast.head()
ProvPast8 = ProvData[ProvData['Day']==Past8].reset_index() #RegPast.head()


# ## 1. Parameters at the national level
# 
# 

# In[ ]:


#https://gist.github.com/datajournalism-it/
#repo_url = 'https://gist.githubusercontent.com/datajournalism-it/48e29e7c87dca7eb1d29/raw/2636aeef92ba0770a073424853f37690064eb0ea/regioni.geojson'

with open('../input/geojsonitaly/regioni.geojson') as f:
    italy_regions_geo = json.load(f)

Map = px.choropleth(data_frame=RegToday, 
                    geojson=italy_regions_geo, 
                    locations='RegionName', # name of dataframe column
                    featureidkey='properties.NOME_REG',  # path to field in GeoJSON feature object with which to match the values passed in to locations
                    color='TotalPositiveCases',
                    color_continuous_scale="YlOrRd",
                    scope="europe", title="Total positive cases at " + str(Today)
                   )
Map.update_geos(showcountries=False, showcoastlines=False, showland=True, fitbounds="locations")
Map.update_layout(margin={"r":0,"t":28,"l":0,"b":0})
Map.show()


# ### 1.1 Total and New Positive Cases: trends in Italy
# Barplots showing for each day the number of __cumulative positive cases__ (TotalPositiveCases) and of __new cases__. The number of daily new cases starts to stabilize and then steadily decreases, although with fluctuations, from the last days of March.

# In[ ]:


BTot = px.bar(National, x='Day', y='TotalPositiveCases', color_discrete_sequence = [tot])
BNew = px.bar(National, x='Day', y='NewPositiveCases', color_discrete_sequence = [new])

Fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                   subplot_titles=('Cumulative Positive Cases', 'New Positive Cases'))

Fig.add_trace(BTot['data'][0], row=1, col=1)
Fig.add_trace(BNew['data'][0], row=2, col=1)

Fig.update_layout(height=650, width=800, title_text=("Situation updated at " + str(Today)), template='plotly_white')


# ### 1.2 Total Positive Cases and related measures
# 
# Cumulative positive cases are broken down in currently positive, recovered and deaths. The number of recovered patients overcomes the currently positives from May 7th. 

# In[ ]:


Temp = National.loc[:, ['Day', 'TotalPositiveCases', 'CurrentPositiveCases','Recovered', 'Deaths',]].melt(id_vars =['Day'], 
                                        value_vars =['TotalPositiveCases', 'CurrentPositiveCases', 'Recovered', 'Deaths'])

TotBreak = px.scatter(Temp, x="Day", y="value", color='variable', height=650, width=800, template='plotly_white',
                  color_discrete_sequence = [tot, cur, rec, dea], 
                  title='Breakdown of total cases at the national level')
TotBreak.update_traces(mode="markers+lines", marker=dict(size=8))
TotBreak.update_layout(xaxis_rangeslider_visible=True)


# In[ ]:


# print(National.columns.values)
BTot = px.bar(National, x='Day', y='TotalPositiveCases', color_discrete_sequence = [tot])
BHos = px.bar(National, x='Day', y='TotalHospitalizedPatients', color_discrete_sequence = [hos])
BIcu = px.bar(National, x='Day', y='IntensiveCarePatients', color_discrete_sequence = [icu])
BRec = px.bar(National, x='Day', y='Recovered', color_discrete_sequence = [rec])
BDea = px.bar(National, x='Day', y='Deaths', color_discrete_sequence = [dea])
BTest = px.bar(National, x='Day', y='TestsPerformed', color_discrete_sequence = [test])
BPosOnTests = px.bar(National, x='Day', y='PosOnTests', color_discrete_sequence = [tot])
BDeaOnTot = px.bar(National, x='Day', y='DeathOnTot', color_discrete_sequence = [dea])

Fig = make_subplots(rows=4, cols=2, shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.05,
                   subplot_titles=('Total Positive Cases', 'Tests Performed', 'Hospitalized Patients', 
                                   'Recovered', 'Intensive Care Patients', 'Deaths', '% Cases / Performed Tests', 
                                  '% Deaths / Cases'))

Fig.add_trace(BTot['data'][0], row=1, col=1)
Fig.add_trace(BTest['data'][0], row=1, col=2)
Fig.add_trace(BHos['data'][0], row=2, col=1)
Fig.add_trace(BRec['data'][0], row=2, col=2)
Fig.add_trace(BIcu['data'][0], row=3, col=1)
Fig.add_trace(BDea['data'][0], row=3, col=2)
Fig.add_trace(BPosOnTests['data'][0], row=4, col=1)
Fig.add_trace(BDeaOnTot['data'][0], row=4, col=2)

Fig.update_layout(height=1200, width=800, title_text=("Situation updated at " + str(Today)), template='plotly_white')


# 
#  * Comparing the cumulative number of positive cases and of performed tests: > 220K cases on a total of > 1.5 Million tests at May 12th
#  * The distribution of hospitalized patients reaches a plateau around the end of March and then starts to decrease
#  * Patients in Intensive Care in slight decrease from early April
#  * The % of positive cases on total performed tests is in a decreasing trend, but still higher around 13%
#  * On mid-May, the % of deaths on positive cases still does not show a tendency to decrease
# 
# 

# *****

# ## 2. Which Regions have been affected more harshly?
# 
# ### 2.1 Looking at cumulative and new cases today, 4 weeks ago and 8 weeks ago
# 
# Let's check the number of __total positive cases__ and the __new cases__ as of the latest reported day and 4 and 8 weeks before. Each parameter is visualized for the 8 Italian regions having the highest values. 
# __Lombardy is constantly the top-ranking region. __
# 

# In[ ]:


TodayTotal = px.bar(RegToday.sort_values('TotalPositiveCases').tail(8), x="TotalPositiveCases", y="RegionName", 
                orientation='h', text='TotalPositiveCases', color_discrete_sequence = [tot])
PastTotal = px.bar(RegPast.sort_values('TotalPositiveCases').tail(8), x="TotalPositiveCases", y="RegionName",
                   orientation='h', text='TotalPositiveCases', color_discrete_sequence = [tot])
PastTotal8 = px.bar(RegPast8.sort_values('TotalPositiveCases').tail(8), x="TotalPositiveCases", y="RegionName",
                   orientation='h', text='TotalPositiveCases', color_discrete_sequence = [tot])
TodayNew = px.bar(RegToday.sort_values('NewPositiveCases').tail(8), x="NewPositiveCases", y="RegionName", 
                orientation='h', text='NewPositiveCases', color_discrete_sequence = [new])
PastNew = px.bar(RegPast.sort_values('NewPositiveCases').tail(8), x="NewPositiveCases", y="RegionName", 
                orientation='h',text='NewPositiveCases', color_discrete_sequence = [new])
PastNew8 = px.bar(RegPast8.sort_values('NewPositiveCases').tail(8), x="NewPositiveCases", y="RegionName", 
                orientation='h',text='NewPositiveCases', color_discrete_sequence = [new])

BarFig = make_subplots(rows=3, cols=2, shared_xaxes=True, horizontal_spacing=0.15, vertical_spacing=0.075,
                       subplot_titles=(
                                      'Cumulative Cases at ' + str(Past8), 'New Cases on ' + str(Past8), 
                                      'Cumulative Cases at ' + str(Past), 'New Cases on ' + str(Past),
                                       'Cumulative Cases at ' + str(Today), 'New Cases on ' + str(Today)))

BarFig.add_trace(PastTotal8['data'][0], row=1, col=1)
BarFig.add_trace(PastNew8['data'][0], row=1, col=2)
BarFig.add_trace(PastTotal['data'][0], row=2, col=1)
BarFig.add_trace(PastNew['data'][0], row=2, col=2)
BarFig.add_trace(TodayTotal['data'][0], row=3, col=1)
BarFig.add_trace(TodayNew['data'][0], row=3, col=2)

BarFig.update_layout(height=800, template='plotly_white')


# ### 2.2 Stratification for regions: current situation
# 
# Representing the data as treeplots allows to visually illustrate the contribution of each region to the national values. 
# * In the first plot, the area represents the __Cumulative Positive Cases__, with the color code according to the __number of cases adjusted for the population__ of each region (_TotalPositiveCases/10,000 inhabitants_).  
# * In the second treeplot, the area is related to __Cumulative Deaths__, that is then __adjusted for the population__ (_Deaths/10,000 inhabitants_) for the color code.
# 
# _Commenting the situation at May 12th:_ while Lombardy accounts for around 1/3 of the total national cases, about half of the deaths have occurred in Lombardy. When considering the population, Valle d'Aosta (the smallest Italian region) shows the highest fraction of positive cases, while the highest fraction of deaths remains in Lombardy.
# 

# In[ ]:


TotToday = px.treemap(RegToday, 
                 path=["RegionName"], values="TotalPositiveCases", height=700,
                 title='Total Cases updated at ' + str(Today), color='TotalOnPop',
                 color_continuous_scale='viridis')
TotToday.data[0].textinfo = 'label+text+value'

TotToday.update_layout(height=450, width=800, template='plotly_white')
TotToday.show()

DeathToday = px.treemap(RegToday, 
                 path=["RegionName"], values="Deaths",
                 title='Total Deaths updated at ' + str(Today), color='DeaOnPop',
                 color_continuous_scale='viridis')
DeathToday.data[0].textinfo = 'label+text+value'

DeathToday.update_layout(height=450, width=800, template='plotly_white')


# ### 2.3 The time dimension
# 
# * The first visualization highlights the __number of new cases__ reported daily for each region (dot size and color)
# * The dynamic baplots depict the increase in the number of cases normalized for population (positive cases/10,000 inhabitants)
# * The dynamic scatterplot shows the increase of cumulative cases (x-axis) and cumulative deaths(y-axis) throguh time; the size each dot is related to the fraction of deaths/10,000 inhabitants
# 
# 

# In[ ]:


temp = RegData[RegData['NewPositiveCases']>0].sort_values('RegionName', ascending=False)
temp.head()
fig = px.scatter(temp, x='Day', y='RegionName', size='NewPositiveCases', color='NewPositiveCases', height=600, width=850, 
           color_continuous_scale=px.colors.sequential.Viridis, title='Daily New Cases stratified by region')

fig.update_layout(yaxis = dict(dtick = 1), template='plotly_white')


# In[ ]:


pal_hls = sn.hls_palette(20, l=.6, s=.9).as_hex() 

fig = px.bar(RegData, y="RegionName", x="TotalOnPop", animation_frame="Date", animation_group="RegionName",
            color="RegionName", hover_name="RegionName", color_discrete_sequence=pal_hls, orientation='h', height=650)
fig.update_yaxes(categoryorder='total descending')

fig.update_xaxes(range=[0,100])
fig.update_layout(template='plotly_white')
fig.update_layout(title='Cumulative cases every 10,000 inhabitants')


# In[ ]:


#help(sn.hls_palette)
#print(pal_hls)

fig = px.scatter(RegData, x="TotalPositiveCases", y="Deaths", animation_frame="Date", animation_group="RegionName",
           size="DeaOnPop", color="RegionName", hover_name="RegionName", color_discrete_sequence=pal_hls,
           size_max=40, range_x=[-1000,100000], range_y=[-1000,18000], )
fig.update_layout(template='plotly_white', height=800, width=850)
#https://plotly.com/python/builtin-colorscales/


# *****

# ## 3. Lombardy Contribution to National Trends
# 
# From the analyses above, __Lombardy stands out as the Italian region most affected by the pandemic__. The following steps are therefore focused on Lombardy, comparing its data and trends to the national ones. 
# 

# ### 3.1 National and Regional -Lombardy- breakdown of positive cases

# In[ ]:


# Extract data for Italy
TempN = National.loc[:, ['Day', 'TotalPositiveCases', 'CurrentPositiveCases','Recovered', 'Deaths',]].melt(id_vars =['Day'], 
                                        value_vars =['TotalPositiveCases', 'CurrentPositiveCases', 'Recovered', 'Deaths'])

# Extract data for Lombardy
TempL = Lombardia.loc[:, ['Day', 'TotalPositiveCases', 'CurrentPositiveCases','Recovered', 'Deaths',]].melt(id_vars =['Day'], 
                                        value_vars =['TotalPositiveCases', 'CurrentPositiveCases', 'Recovered', 'Deaths'])

# National Plot
TotBreakN = px.scatter(TempN, x="Day", y="value", color='variable', template='plotly_white',
                  color_discrete_sequence = [tot, cur, rec, dea], 
                  title='Breakdown of total cases at the national level')
TotBreakN.update_traces(mode="markers+lines", marker=dict(size=6))

# Lombardy Plot
TotBreakL = px.scatter(TempL, x="Day", y="value", color='variable', template='plotly_white',
                  color_discrete_sequence = [tot, cur, rec, dea], 
                  title='Breakdown of total cases in Lombardy')
TotBreakL.update_traces(mode="markers+lines", marker=dict(size=6))

# Subplots
TotFig = make_subplots(rows=1, cols=2, horizontal_spacing=0.025, shared_yaxes=True,
                       subplot_titles=('National', 'Lombardy'))


TotFig.add_trace(TotBreakN['data'][0], row=1, col=1)
TotFig.add_trace(TotBreakN['data'][1], row=1, col=1)
TotFig.add_trace(TotBreakN['data'][2], row=1, col=1)
TotFig.add_trace(TotBreakN['data'][3], row=1, col=1)
TotFig.add_trace(TotBreakL['data'][0], row=1, col=2)
TotFig.add_trace(TotBreakL['data'][1], row=1, col=2)
TotFig.add_trace(TotBreakL['data'][2], row=1, col=2)
TotFig.add_trace(TotBreakL['data'][3], row=1, col=2)

TotFig.update_layout(height=650, template='plotly_white', width=800, legend_orientation="h", 
                     legend=dict(font=dict(size=10)))


# ### 3.2 National and Regional -Lombardy- breakdown of current cases

# In[ ]:


# Extract data for Italy
TempN = National.loc[:, ['Day', 'CurrentPositiveCases', 'HomeConfinement','TotalHospitalizedPatients', 'IntensiveCarePatients']].melt(id_vars =['Day'], 
                                        value_vars =['CurrentPositiveCases','HomeConfinement', 'TotalHospitalizedPatients', 'IntensiveCarePatients'])

# Extract data for Lombardy
TempL = Lombardia.loc[:, ['Day', 'CurrentPositiveCases', 'HomeConfinement', 'TotalHospitalizedPatients', 'IntensiveCarePatients', ]].melt(id_vars =['Day'], 
                                        value_vars =['CurrentPositiveCases', 'HomeConfinement', 'TotalHospitalizedPatients', 'IntensiveCarePatients'])

# Italy plot
CaseBreakN = px.scatter(TempN, x="Day", y="value", color='variable', height=650, width=800, template='plotly_white',
                  color_discrete_sequence = px.colors.qualitative.Prism[5:9], 
                  title='Breakdown of total cases at the national level')
CaseBreakN.update_traces(mode="markers+lines", marker=dict(size=6))
#yaxis_type="log"

# Lombardy plot
CaseBreakL = px.scatter(TempL, x="Day", y="value", color='variable', height=650, width=800, template='plotly_white',
                  color_discrete_sequence = px.colors.qualitative.Prism[5:9], 
                  title='Breakdown of total cases in Lombardy')
CaseBreakL.update_traces(mode="markers+lines", marker=dict(size=6))

# Subplots
CaseFig = make_subplots(rows=1, cols=2, horizontal_spacing=0.025, shared_yaxes=True,
                       subplot_titles=('National', 'Lombardy'))


CaseFig.add_trace(CaseBreakN['data'][0], row=1, col=1)
CaseFig.add_trace(CaseBreakN['data'][1], row=1, col=1)
CaseFig.add_trace(CaseBreakN['data'][2], row=1, col=1)
CaseFig.add_trace(CaseBreakN['data'][3], row=1, col=1)
CaseFig.add_trace(CaseBreakL['data'][0], row=1, col=2)
CaseFig.add_trace(CaseBreakL['data'][1], row=1, col=2)
CaseFig.add_trace(CaseBreakL['data'][2], row=1, col=2)
CaseFig.add_trace(CaseBreakL['data'][3], row=1, col=2)

CaseFig.update_layout(height=650, template='plotly_white', width=800, legend_orientation="h", 
                     legend=dict(font=dict(size=10)))


# ### 3.3 Comparing Italy and Lombardy curves

# In[ ]:


tempL = Lombardia.loc[:, ['Day', 'TotalPositiveCases', 'IntensiveCarePatients', 'Deaths', 'Recovered', 'TestsPerformed', 'NewPositiveCases', 
                          'DeathOnTot', 'PosOnTests', 'TotalHospitalizedPatients']]
tempL['Type'] ='Lombardy'
tempN = National.loc[:, ['Day', 'TotalPositiveCases', 'IntensiveCarePatients', 'Deaths', 'Recovered', 'TestsPerformed', 'NewPositiveCases', 
                          'DeathOnTot', 'PosOnTests', 'TotalHospitalizedPatients']]
tempN['Type'] ='National'
temp = tempL.append(tempN)

temp.head()

ATot = px.scatter(temp, x="Day", y="TotalPositiveCases", color='Type', height=800, template='plotly_white',
                  color_discrete_sequence = [tot, '#b8b8f9'])
ATot.update_traces(mode="markers+lines", marker=dict(size=6))

ANew = px.scatter(temp, x="Day", y="NewPositiveCases", color='Type', height=800, template='plotly_white',
             color_discrete_sequence = [new, '#ddacfe'])
ANew.update_traces(mode="markers+lines", marker=dict(size=6))

ADea = px.scatter(temp, x="Day", y="Deaths", color='Type', template='plotly_white',
              color_discrete_sequence = [dea, '#c1c1c1'])
ADea.update_traces(mode="markers+lines", marker=dict(size=6))

ATest = px.scatter(temp, x="Day", y="TestsPerformed", color='Type', template='plotly_white',
            color_discrete_sequence = ['#006767', test])
ATest.update_traces(mode="markers+lines", marker=dict(size=6))

APercDea = px.scatter(temp, x="Day", y="DeathOnTot", color='Type', template='plotly_white',
             color_discrete_sequence = [dea, '#c1c1c1'])
APercDea.update_traces(mode="markers+lines", marker=dict(size=6))

APercPos = px.scatter(temp, x="Day", y="PosOnTests", color='Type', template='plotly_white',
             color_discrete_sequence = ['#006767', test])
APercPos.update_traces(mode="markers+lines", marker=dict(size=6))

AHos = px.scatter(temp, x="Day", y="TotalHospitalizedPatients", color='Type', template='plotly_white',
             color_discrete_sequence = [px.colors.qualitative.Prism[7], px.colors.qualitative.Prism[5]])
AHos.update_traces(mode="markers+lines", marker=dict(size=6))

AICU = px.scatter(temp, x="Day", y="IntensiveCarePatients", color='Type', template='plotly_white',
             color_discrete_sequence = [px.colors.qualitative.Prism[9], px.colors.qualitative.Prism[6]])
AICU.update_traces(mode="markers+lines", marker=dict(size=6))


LineFig = make_subplots(rows=4, cols=2, shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.05, 
                       subplot_titles=('Cumulative Positive Cases', 'New Positive Cases', 'Cumulative Deaths', 
                                      'Cumulative Tests','TotalHospitalizedPatients', 'IntensiveCarePatients', '% Deaths on Total Cases', '% Cases on Performed Tests'))

LineFig.add_trace(ATot['data'][0], row=1, col=1)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].TotalPositiveCases), text='Lombardy', row=1, col=1)
LineFig.add_trace(ATot['data'][1], row=1, col=1)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].TotalPositiveCases), text='National', row=1, col=1)

LineFig.add_trace(ANew['data'][0], row=1, col=2)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].NewPositiveCases), text='Lombardy', row=1, col=2)
LineFig.add_trace(ANew['data'][1], row=1, col=2)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].NewPositiveCases), text='National', row=1, col=2)

LineFig.add_trace(ADea['data'][0], row=2, col=1)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].Deaths), text='Lombardy', row=2, col=1)
LineFig.add_trace(ADea['data'][1], row=2, col=1)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].Deaths), text='National', row=2, col=1)

LineFig.add_trace(ATest['data'][0], row=2, col=2)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].TestsPerformed), text='Lombardy', row=2, col=2)
LineFig.add_trace(ATest['data'][1], row=2, col=2)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].TestsPerformed), text='National', row=2, col=2)

LineFig.add_trace(AHos['data'][0], row=3, col=1)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].TotalHospitalizedPatients), text='Lombardy', row=3, col=1)
LineFig.add_trace(AHos['data'][1], row=3, col=1)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].TotalHospitalizedPatients), text='National', row=3, col=1)

LineFig.add_trace(AICU['data'][0], row=3, col=2)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].IntensiveCarePatients), text='Lombardy', row=3, col=2)
LineFig.add_trace(AICU['data'][1], row=3, col=2)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].IntensiveCarePatients), text='National', row=3, col=2)

LineFig.add_trace(APercDea['data'][0], row=4, col=1)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].DeathOnTot), text='Lombardy', row=4, col=1)
LineFig.add_trace(APercDea['data'][1], row=4, col=1)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].DeathOnTot), text='National', row=4, col=1)

LineFig.add_trace(APercPos['data'][0], row=4, col=2)
LineFig.add_annotation(x=Today, y=float(tempL.loc[tempL.Day==Today,:].PosOnTests), text='Lombardy', row=4, col=2)
LineFig.add_trace(APercPos['data'][1], row=4, col=2)
LineFig.add_annotation(x=Today, y=float(tempN.loc[tempN.Day==Today,:].PosOnTests), text='National', row=4, col=2)

LineFig.update_layout(height=1400, template='plotly_white', showlegend=False)


# __Observing the behaviour of the curves till mid-April:__
# 
# * For absolute values, Lombardy curve is closest to the national one for the number of deathts while is furthest apart for the performed tests
# * Looking at the % curves, the fraction of positive cases/performed tests and deaths/total cases are both higher for Lombardy compared to the national levels.

# ******

# ## 4. Zooming in: Lombardy provinces
# 
# 
# 
# 

# In[ ]:


#https://gist.github.com/datajournalism-it/
#repo_url = 'https://gist.githubusercontent.com/datajournalism-it/212e7134625fbee6f9f7/raw/dabd071fe607f5210921f138ad3c7276e3841166/province.geojson'

with open('../input/geojsonitaly/province.geojson') as f:
    italy_province_geo = json.load(f)

Map = px.choropleth(data_frame=ProvToday, 
                    geojson=italy_province_geo, 
                    locations='ProvinceName', # name of dataframe column
                    featureidkey='properties.NOME_PRO',  # path to field in GeoJSON feature object with which to match the values passed in to locations
                    color='TotalPositiveCases',
                    color_continuous_scale="YlOrRd",
                    scope="europe", title="Total positive cases at " + str(Today)
                   )
Map.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")
Map.update_layout(margin={"r":0,"t":28,"l":0,"b":0})
Map.show()


# ### 4.1 Breakdown of Lombardy cumulative cases for province

# In[ ]:


ProPal = sn.hls_palette(12, l=.5, s=.9).as_hex() 

ProvTot = px.scatter(ProvData, x="Day", y="TotalPositiveCases", color='ProvinceName', height=750, width=850, template='plotly_white',
                  title='Breakdown of total cases in Lombardy', color_discrete_sequence=ProPal)

ProvTot.update_traces(mode="markers+lines", marker=dict(size=8))
#ProvTot.update_layout(xaxis_rangeslider_visible=True, yaxis_type="log")
ProvTot.update_layout(xaxis_rangeslider_visible=True)


# In[ ]:


ProPal = sn.hls_palette(12, l=.5, s=.9).as_hex() 

ProvTotAdj = px.scatter(ProvData, x="Day", y="TotalOnPop", color='ProvinceName', height=750, width=850, template='plotly_white',
                  title='Cases/10000 inhabitant in each Province', color_discrete_sequence=ProPal)

ProvTotAdj.update_traces(mode="markers+lines", marker=dict(size=8))
#ProvTot.update_layout(xaxis_rangeslider_visible=True, yaxis_type="log")
ProvTotAdj.update_layout(xaxis_rangeslider_visible=True)


# * Observing the total numbers: after an early increase in Lodi, Bergamo, Brescia and Milano are the major contributors
# * When adjusting for the number of inhabitants, the strong increase in the first half of March in Lodi is much more visible. Through time, Cremona  stands out, followed by Bergamo and Brescia.

# ### 4.2 Contribution of each province today and 8 weeks ago
# 
# The treeplots report the __contribution of each province to Lombardy total cases__ as of the most current date and 8 weeks before (area of each square). The color code represents to the __number of cases adjusted for the population__ (_TotalPositiveCases/10,000 inhabitants_)

# In[ ]:


TotPast8 = px.treemap(ProvPast8, 
                 path=["ProvinceName"], values="TotalPositiveCases",
                 title='Total Cases updated at ' + str(Past8), color='TotalOnPop',
                 color_continuous_scale='viridis', height=450, width=800)
TotPast8.data[0].textinfo = 'label+text+value'
#TotToday.update_layout()
TotPast8.show()


# In[ ]:


TotToday = px.treemap(ProvToday, 
                 path=["ProvinceName"], values="TotalPositiveCases",
                 title='Total Cases updated at ' + str(Today), color='TotalOnPop',
                 color_continuous_scale='viridis', height=450, width=800)
TotToday.data[0].textinfo = 'label+text+value'
#TotToday.update_layout()
TotToday.show()


# ### 4.3 Dynamic view through time

# In[ ]:


fig = px.bar(ProvData, y="ProvinceName", x="TotalPositiveCases", animation_frame="Date", animation_group="ProvinceName",
            color="ProvinceName", hover_name="ProvinceName", color_discrete_sequence=ProPal, orientation='h', height=650)
fig.update_yaxes(categoryorder='total descending')

fig.update_xaxes(range=[0,28000])
fig.update_layout(template='plotly_white')
fig.update_layout(title='Total cases')


# In[ ]:


ProvData['DaySum'] = ProvData.apply(lambda x: ProvData.loc[(ProvData.Day == x.Day), 'TotalPositiveCases'].sum(), axis=1)
ProvData['Population'] = ProvData.apply(lambda x: PopDict[x.ProvinceName], axis=1)
# Cases for 10000 inhabitants
#ProvData['TotalOnPop'] = round(ProvData.TotalPositiveCases/ProvData.Population * 10000, 2)

fig = px.scatter(ProvData, x="TotalPositiveCases", y="DaySum", animation_frame="Date", animation_group="ProvinceName",
           size="TotalOnPop", color="ProvinceName", hover_name="ProvinceName",
           size_max=45, range_x=[-500,28000], range_y=[-2000,110000], color_discrete_sequence=ProPal)
fig.update_layout(template='plotly_white')

