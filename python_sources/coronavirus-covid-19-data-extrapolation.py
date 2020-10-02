#!/usr/bin/env python
# coding: utf-8

# <font face = "Verdana" size ="5">Coronavirus is a family of viruses that are named after their spiky crown. The novel coronavirus, also known as SARS-CoV-2, is a contagious respiratory virus that first reported in Wuhan, China. On 2/11/2020, the World Health Organization designated the name COVID-19 for the disease caused by the novel coronavirus. This notebook aims at exploring COVID-19 through data analysis and projections. 
#     
#    <br> This notebook is forked from Xingyu Bian's <a href='https://www.kaggle.com/therealcyberlord/coronavirus-covid-19-visualization-prediction'>notebook</a>
#  
#    <br>Data is provided by the <a href='https://github.com/nytimes/covid-19-data'>New York Times</a> and the <a href='https://covidtracking.com/'>COVID Tracking Project</a>
#    <br>Learn more from the <a href='https://www.who.int/emergencies/diseases/novel-coronavirus-2019'>WHO</a>
#    <br>Learn more from the <a href='https://www.cdc.gov/coronavirus/2019-ncov'>CDC</a>
#    
#    <font face = "Verdana" size ="4">
#     <br> Last update: 8/19/2020 08:05 EST
#     <br><i> New Updates: Update for 8/19 data </i>
#    </font>
# 
#    <font face = "Verdana" size ="1">
#     <center><img src='https://www.statnews.com/wp-content/uploads/2020/02/Coronavirus-CDC-645x645.jpg'>
#      Source: https://www.statnews.com/wp-content/uploads/2020/02/Coronavirus-CDC-645x645.jpg </center> 
#     </font>
# 
# <br>
# <font face = "Verdana" size ="6"> Sections </font>
# * <a href='#features'>List of Features</a>
# * <a href='#load_us_data'>Load latest Data</a>
# * <a href='#world_metrics'>World Metrics</a>    
# * <a href='#us_summary'>US Summary</a>
# * <a href='#us_features'>Visualizing US Data</a>
# * <a href='#build_train_ML'>Building and Training ML models for data extrapolation</a>

# <font size="7"><b>Features to be examined</b></font>
#  <a id='features'></a>

# * (1-3) Stability indices for cases, hospitalizations and deaths
# * (4-6) Doubling times for cases, hospitalizations and deaths
# * (7-10) % increase in cases, hospitalizations, deaths and recovered (if available) over the last week
# * (11-14) % positive cases, Hospitalizations/Capacity, ICU beds/Capacity, Intubations/Ventilator Capacity

# <font size="5">Import Packages</font>

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt

import pandas as pd 
import random
import math
import time

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from scipy.signal import dlsim, TransferFunction, argrelextrema, find_peaks
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Embedding

import datetime
from datetime import date
import operator 

import collections

# Libraries for creating interactive plots
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs
init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)

from urllib.request import urlopen
import json

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#!/opt/conda/bin/python3.7 -m pip install --upgrade pip

get_ipython().system('pip install us')
import us


# <font size="6"><b>Loading latest data (along with normalization)</b></font>
# <a id='load_us_data'></a>

# In[ ]:


# Load latest GitHub data for US
usData = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv')
usStateData = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
usCountyData = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

# Load latest data from COVID Tracking Project (has hospitalization data)
covidTrackingDaily = pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
covidTrackingDaily = covidTrackingDaily.fillna(0)

# Load racial data for COVID Tracking Project - NOTE: This is only updated every 3-4 days
racialDataUS = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR_xmYt4ACPDZCDJcY12kCiMiH0ODyx3E1ZvgOHB8ae1tRcjXbs_yWBOA4j4uoCEADVfC1PS2jYO68B/pub?gid=43720681&single=true&output=csv')
racialDataUS = racialDataUS.fillna(0)

# Load in WORLD data from John Hopkins, along with FIPS/ISO lookup table for world geo plots
worldCases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
worldDeaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
worldRecoveries = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
isoLookupTable = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv')

# Don't display decimals
pd.set_option('precision', 0)


# In[ ]:


usData.loc[usData.shape[0] - 14:].style.background_gradient(cmap='RdYlGn_r')


# In[ ]:


usStateData.loc[usStateData.shape[0] - 14:].style.background_gradient(cmap='RdYlGn_r', subset=['cases','deaths'])


# In[ ]:


usCountyData.loc[usCountyData.shape[0] - 14:].style.background_gradient(cmap='RdYlGn_r', subset=['cases','deaths'])


# In[ ]:


dates = covidTrackingDaily.date
dates_fixed = []
for i in range(dates.shape[0]):
    date_str = str(dates[i])
    fixed_date = datetime.datetime(year=int(date_str[0:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))
    dates_fixed.append(fixed_date.strftime("%Y-%m-%d"))
covidTrackingDaily.date = dates_fixed

covidTrackingDaily.loc[0:14].style.background_gradient(cmap='RdYlGn_r')


# In[ ]:


dates = racialDataUS.Date
dates_fixed = []
for i in range(dates.shape[0]):
    date_str = str(dates[i])
    fixed_date = datetime.datetime(year=int(date_str[0:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))
    dates_fixed.append(fixed_date.strftime("%Y-%m-%d"))
racialDataUS.Date = dates_fixed

racialDataUS.loc[0:14].style.background_gradient(cmap='RdYlGn_r')


# <font size="4"><b>Only last 7 days of world data are shown (to save space)</b></font>

# In[ ]:


# Get next7Days as string
last7Days_str = []
date = datetime.datetime(2020,8,19)              # Latest date available in data
for i in range(7): 
    date -= datetime.timedelta(days=1)
    last7Days_str.append(date.strftime('%-m/%-d/%Y'))
    
numCols = worldRecoveries.shape[1]
cols = [0,1]
for i in range(7,0,-1):
    cols.append(numCols - i)


# <font size="4"><b>World Cases</b></font>

# In[ ]:


worldCases.iloc[0:14,cols].style.background_gradient(cmap='RdYlGn_r', axis=1)


# <font size="4"><b>World Deaths</b></font>

# In[ ]:


worldDeaths.iloc[0:14,cols].style.background_gradient(cmap='RdYlGn_r', axis=1)


# <font size="4"><b>World Recoveries</b></font>

# In[ ]:


worldRecoveries.iloc[0:14,cols].style.background_gradient(cmap='RdYlGn', axis=1)


# <font size="4"><b>Data Normalization</b></font>

# In[ ]:


usCases = (usData.cases).to_numpy()
usDeaths = (usData.deaths).to_numpy()
dates =  (usData.date).to_numpy()

min_max_scaler = MinMaxScaler()
usCases_norm = min_max_scaler.fit_transform(usCases.reshape(-1,1))


# <font size="7"><b>World Metrics</b></font>
# <a id='world_metrics'></a>

# In[ ]:


def fullStateNameToAbbrev(fullStateNames):
    abbrevs = []
    
    for i in range(fullStateNames.shape[0]):
        abbrevs.append(eval("us.states.lookup('" + str(fullStateNames[i]) + "').abbr"))
    
    return abbrevs


# In[ ]:


# Create n-day average for an array
def nDayAverage(data, n):
    nDayAvgData = np.zeros(data.shape)
    dataLen = nDayAvgData.shape[0]
    
    for i in range(dataLen):
        idxs = np.arange(-n + 1 + i, i + 1)
        idxs = np.clip(idxs, a_min=0, a_max=np.inf)
        idxs = idxs.astype(int)
        
        nDayAvgData[i] = np.mean(data[idxs])
        
    return nDayAvgData


# In[ ]:


def allDoublingTimes(data):
    allDoublingTimes = np.zeros(data.shape[0],)
    
    for i in range(-1,-allDoublingTimes.shape[0],-1):
        currentValue = data[i]
        halfOfCurrentValue = currentValue / 2.0
        
        dataMinusHalfOfCurrentCases = abs(data - halfOfCurrentValue)
        
        idxApproxHalfOfValue = np.where(dataMinusHalfOfCurrentCases == np.min(dataMinusHalfOfCurrentCases))[0][0]
        
        allDoublingTimes[i] = (i + data.shape[0]) - idxApproxHalfOfValue
        
    return allDoublingTimes


# In[ ]:


# -1: State has already recovered
# 365: State is getting worse, no recovery in progress

def findTimeForRecovery(DTData):
    # Find global max (right before things went bad)
    if ((np.diff(DTData[-45:]) > -0.01).all()):  # This is required for good states where DT is continuously rising (at least for the last 45 days)
        maximas = DTData.shape[0] - 1
    else:
        maximas = find_peaks(DTData)[0]
    dataAtMaximas = DTData[maximas]
    globalMaxIdx = np.where(DTData == np.amax(dataAtMaximas))[0][0]
    globalMax = np.mean(DTData[globalMaxIdx])                        # To account for two of the same maxs found
    
    # Then find minima on the data that comes after the maxima
    clippedData = DTData[globalMaxIdx:]
    minimas = argrelextrema(clippedData, np.less_equal)[0]
    latestMinIdxClipped = np.where(clippedData == np.amin(clippedData[minimas]))[0][0]
    latestMinIdx = globalMaxIdx + latestMinIdxClipped
    latestMin = np.mean(clippedData[latestMinIdxClipped])

    # Find avg slope for increase
    avgIncPerDay = 0
    if (latestMinIdx != DTData.shape[0] and DTData[-1] - DTData[latestMinIdx] != 0):
        avgIncPerDay = (DTData[-1] - DTData[latestMinIdx]) / (DTData.shape[0] - latestMinIdx)
        incReqForRecovery = 30 - DTData[-1]                                              # Make sure to take into account of today's DT (find time to reach 30 days, NOT PEAK)
        timeForRecovery = incReqForRecovery / avgIncPerDay                                         
    if (latestMinIdx == DTData.shape[0] or avgIncPerDay < 0.1):
        timeForRecovery = 365
    if (DTData[-1] - DTData[latestMinIdx] == 0):
        timeForRecovery = 365
    if (globalMax == latestMin):
        timeForRecovery = -1

    return timeForRecovery


# In[ ]:


# Functions for pandas DataFrame cell conditioning of DTs
def redColorscale(s):
    is_red = s < 21
    return ['background-color: red' if v else '' for v in is_red]

def yellowColorscale(s):
    is_yellow = [1 if x >= 21 and x <= 30 else 0 for x in s]
    return ['background-color: yellow' if v else '' for v in is_yellow]

def greenColorscale(s):
    is_green = s > 30
    return ['background-color: green' if v else '' for v in is_green]


# In[ ]:


# Choropleth for world with animation
allCountries = np.unique(worldCases['Country/Region'])
numCountries = allCountries.shape[0]
numDates = 211                           # 1-22-2020 to 8-19-2020
numEntries = numCountries * numDates
blankTableSpace = np.zeros(numEntries,).tolist()

# Get all dates as strings
allDates = []
date = datetime.datetime(2020,1,21)              # Latest date available in data
for i in range(numDates): 
    date += datetime.timedelta(days=1)
    allDates.append(date.strftime('%-m/%-d/%Y'))

# Duplicate country names, ISOs, and Pop for each day
allCountriesRep = np.zeros(numEntries,).astype(str)
allCountryISOsRep = np.zeros(numEntries,).astype(str)
allCountryPopRep = np.zeros(numEntries,)
for i in range(numCountries):
    idxs = np.arange(i*numDates,(i+1)*numDates)
    allCountriesRep[idxs] = allCountries[i]  
    allCountryISOsRep[idxs] = (isoLookupTable.loc[isoLookupTable['Country_Region'] == allCountries[i]].iso3).to_numpy()[0]
    allCountryPopRep[idxs] = (isoLookupTable.loc[isoLookupTable['Country_Region'] == allCountries[i]].Population).to_numpy()[0]

# Create data table for choropleth plot
dummyData = np.zeros(numEntries,).tolist()
d = {'Country': allCountriesRep.tolist(), 'Date': np.ones(numEntries,).astype(str).tolist(), 'ISO': allCountryISOsRep, 'Population': allCountryPopRep, 
     'Cases': dummyData, 'CasesPer100k': dummyData, 'NewCasesPer100k': dummyData, 'DT_Cases': dummyData, 
     'Deaths': dummyData,'DeathsPer100k': dummyData, 'DT_Deaths': dummyData,
     'Recoveries': dummyData, 'RecoveriesPer100k': dummyData, 'MortalityRate(%)': dummyData}
worldGeoPlotData = pd.DataFrame(data=d)
    
# Find/compute world data for each day
for i in range(numCountries):
#     if (i % 25 == 0):
#         print(i)
        
    # Find rows for current country
    countryIdxs = worldGeoPlotData.loc[worldGeoPlotData.Country == allCountries[i]].index.to_numpy()
    countryPopulation = worldGeoPlotData.Population.to_numpy()[countryIdxs[0]]
    
    # Set dates properly
    worldGeoPlotData.at[countryIdxs, 'Date'] = allDates
    
    # Get cases for country, but first concanete it if necessary
    countryDFCases = worldCases.loc[worldCases['Country/Region'] == allCountries[i]]
    countryCasesIdxs = countryDFCases.index.to_numpy()
    concatCountryDataCases = countryDFCases.groupby('Country/Region').sum().reset_index()
    
    countryCases = concatCountryDataCases.iloc[0,3:].to_numpy().astype(int)
    countryCasesPer100k = countryCases / (countryPopulation / 100000)
    allCountryDT_Cases = allDoublingTimes(countryCases)
    countryNewCasesPer100k = np.concatenate(([0],np.diff(countryCasesPer100k)), axis=0)
    worldGeoPlotData.at[countryIdxs, 'Cases'] = countryCases
    worldGeoPlotData.at[countryIdxs, 'CasesPer100k'] = np.round(countryCasesPer100k, 3)
    worldGeoPlotData.at[countryIdxs, 'DT_Cases'] = allCountryDT_Cases
    worldGeoPlotData.at[countryIdxs, 'NewCasesPer100k'] = countryNewCasesPer100k
    
    # Get deaths for country, but first concanete it if necessary
    countryDFDeaths = worldDeaths.loc[worldDeaths['Country/Region'] == allCountries[i]]
    countryDeathsIdxs = countryDFDeaths.index.to_numpy()
    concatCountryDataDeaths = countryDFDeaths.groupby('Country/Region').sum().reset_index()
    
    countryDeaths = concatCountryDataDeaths.iloc[0,3:].to_numpy().astype(int)
    countryDeathsPer100k = countryDeaths / (countryPopulation / 100000)
    allCountryDT_Deaths = allDoublingTimes(countryDeaths)
    worldGeoPlotData.at[countryIdxs, 'Deaths'] = countryDeaths
    worldGeoPlotData.at[countryIdxs, 'DeathsPer100k'] = np.round(countryDeathsPer100k, 3)
    worldGeoPlotData.at[countryIdxs, 'DT_Deaths'] = allCountryDT_Deaths
    
    # Get recoveries for country, but first concanete it if necessary
    countryDFRec = worldRecoveries.loc[worldRecoveries['Country/Region'] == allCountries[i]]
    countryRecIdxs = countryDFRec.index.to_numpy()
    concatCountryDataRec = countryDFRec.groupby('Country/Region').sum().reset_index()
    
    countryRec = concatCountryDataRec.iloc[0,3:].to_numpy().astype(int)
    countryRecPer100k = countryRec / (countryPopulation / 100000)
    worldGeoPlotData.at[countryIdxs, 'Recoveries'] = countryRec
    worldGeoPlotData.at[countryIdxs, 'RecoveriesPer100k'] = np.round(countryRecPer100k, 3)
    
    # Get mortality rate for country (deaths/cases)
    if (not countryCases.any()):                                       # If any value is not zero
        countryMortalityRate = countryDeaths / countryCases * 100
    else:
        countryMortalityRate = countryDeaths / (countryCases + 0.1) * 100
    worldGeoPlotData.at[countryIdxs, 'MortalityRate(%)'] = np.round(countryMortalityRate, 3)

    
# Sort dates (and convert to date_time to ensure proper date sorting)
worldGeoPlotData.Date = pd.to_datetime(worldGeoPlotData.Date)
worldGeoPlotData = worldGeoPlotData.sort_values(by=['Date'], ascending=False)
worldGeoPlotData.Date = (worldGeoPlotData.Date).astype(str)

    
# Now create the choropleth data
fig = px.choropleth(worldGeoPlotData, locations="ISO",
                    color="DT_Cases",
                    hover_name="Country",
                    hover_data=['Population','Cases','CasesPer100k','DT_Cases','Deaths','DeathsPer100k','DT_Deaths','Recoveries','RecoveriesPer100k', 'MortalityRate(%)'],
                    animation_frame='Date',
                    color_continuous_scale="RdYlGn")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(title='World Metrics')
fig.show()


# <font size="5"><b>World Summary Table (for 8/19/20)</b></font>

# In[ ]:


# First get world clustering data
todayDate = '2020-08-19'
allCountries = np.unique(worldGeoPlotData.Country)
numCountries = allCountries.shape[0]

worldDataToday = worldGeoPlotData.loc[worldGeoPlotData.Date == todayDate]
preExistingData = worldDataToday.sort_values(by=['DT_Cases','DT_Deaths'], ascending=True)

d = {'Country': preExistingData.Country, 'DT_Cases': preExistingData.DT_Cases, 'DT_Deaths': preExistingData.DT_Deaths, 'SeverityIndex': np.zeros(numCountries,).tolist()}
worldClusteringData = pd.DataFrame(data=d)

# Calculate 5 day avg for new cases per 100k for all countries
for i in range(numCountries):
    countryIdx = worldClusteringData.loc[worldClusteringData.Country == allCountries[i]].index[0]
    countryData = worldGeoPlotData.loc[worldGeoPlotData.Country == allCountries[i]]
    countryNewCasesPer100k_7DayAvg = np.mean(countryData.NewCasesPer100k.to_numpy()[0:8])
    worldClusteringData.at[countryIdx, 'SeverityIndex'] = countryNewCasesPer100k_7DayAvg
    
# # Normalize the 5 day avg data
# min_max_scaler = MinMaxScaler()
# worldClusteringData.SeverityIndex = min_max_scaler.fit_transform(worldClusteringData.SeverityIndex.to_numpy().reshape(-1,1))
# worldClusteringData.SeverityIndex = np.round(worldClusteringData.SeverityIndex / np.amax(worldClusteringData.SeverityIndex), 3)
worldClusteringData.SeverityIndex = worldClusteringData.SeverityIndex.fillna(0) 
median = np.median(worldClusteringData.SeverityIndex.to_numpy())
worldClusteringData.SeverityIndex = np.divide(worldClusteringData.SeverityIndex, median)
    

# Now display the table    
worldClusteringData['DT_Cases'].round(0)
worldClusteringData['DT_Deaths'].round(0)
worldClusteringData['SeverityIndex'].round(2)

worldClusteringData.style.    apply(redColorscale, subset=['DT_Cases','DT_Deaths']).    apply(yellowColorscale, subset=['DT_Cases','DT_Deaths']).    apply(greenColorscale, subset=['DT_Cases','DT_Deaths']).    format({
        'DT_Cases': '{:,.0f}'.format,
        'DT_Deaths': '{:,.0f}'.format,
        'SeverityIndex': '{:,.2f}'.format,
    }).\
    background_gradient(cmap='RdYlGn_r', subset='SeverityIndex')


# <font size="5"><b>World Clustering</b></font>

# In[ ]:


# Now create the 3D scatter plot
fig = px.scatter_3d(worldClusteringData, x='DT_Cases', y='DT_Deaths', z='SeverityIndex', hover_name='Country')
fig.update_layout(title='World Doubling Time Clustering by Country', height=800)
fig.show()


# <font size="7"><b>US Summary</b></font>
# <a id='us_summary'></a>

# <font size="4.5">DT is an abbreviation for Doubling Time</font>

# <font size="4.5">Severity Index is the 5 Day Average of New Cases per 100k for all US States divided by the median of that data </font>

# In[ ]:


allStateNames = np.unique(usStateData.state)

#################################################################################################
# DT_Cases/DT_Deaths
numStates = allStateNames.shape[0]
stateCaseData = (usStateData.cases).to_numpy()
stateDeathData = (usStateData.deaths).to_numpy()

doublingTimesCases = np.zeros(numStates,)
doublingTimesDeaths = np.zeros(numStates,)

for i in range(numStates):
    caseData = stateCaseData[np.where(usStateData.state == allStateNames[i])]
    deathData = stateDeathData[np.where(usStateData.state == allStateNames[i])]
    
    doublingTimesCases[i] = allDoublingTimes(caseData)[-1]
    doublingTimesDeaths[i] = allDoublingTimes(deathData)[-1]
#################################################################################################
# Severity Index
severityIdxs = np.zeros(numStates,)

populationData = pd.read_csv('../input/population-data/Population_v2.csv')
populationData = populationData.to_numpy()

for i in range(numStates):
    stateCases = stateCaseData[np.where(usStateData.state == allStateNames[i])]
     
    if (populationData[np.where(populationData == allStateNames[i])[0]].shape[0] != 0):       
        # Check if population data is available
        stateRow = populationData[np.where(populationData == allStateNames[i])[0]]
        statePop = stateRow[0,1]
    
        newCasesData = np.diff(stateCases)
        newCasesPer100k = (newCasesData / (statePop / 100000)).astype(int)
        newCasesPer100kFiltered = nDayAverage(newCasesPer100k, 7)
        severityIdxs[i] = np.mean(newCasesPer100k[-7:])
#################################################################################################    
# Put all the computed data into a table
d = {'State': allStateNames.tolist(), 'DT_Cases': doublingTimesCases.tolist(), 'DT_Deaths': doublingTimesDeaths.tolist(), 'SeverityIndex': severityIdxs.tolist()}
usSummary = pd.DataFrame(data=d)

# Sort the data
usSummary_Sorted = usSummary.sort_values(by=['DT_Cases','DT_Deaths'], ascending=True)

median = np.median(usSummary_Sorted.SeverityIndex.to_numpy())
usSummary_Sorted.SeverityIndex = np.divide(usSummary_Sorted.SeverityIndex, median)


# # Don't display decimals
# pd.set_option('precision', 2)

# Display the table with bakground gradient
#usSummary_Sorted.style.background_gradient(cmap='RdYlGn', subset=['DT_Cases','DT_Deaths'])
        
usSummary_Sorted.style.    apply(redColorscale, subset=['DT_Cases','DT_Deaths']).    apply(yellowColorscale, subset=['DT_Cases','DT_Deaths']).    apply(greenColorscale, subset=['DT_Cases','DT_Deaths']).    background_gradient(cmap='RdYlGn_r', subset='SeverityIndex').    format({
        'DT_Cases': '{:,.0f}'.format,
        'DT_Deaths': '{:,.0f}'.format,
        'SeverityIndex': '{:,.2f}'.format,
    })


# <font size="5"><b>Clustering Test</b></font>

# In[ ]:


fig = px.scatter_3d(usSummary_Sorted, x='DT_Cases', y='DT_Deaths', z='SeverityIndex', hover_name='State')
fig.update_layout(title='US Doubling Time Clustering by State')
fig.show()


# <font size="7"><b>US Data Visualization</b></font>
# <a id='us_features'></a>

# <font size="5"><b>Doubling Times per US State (Geographical Plot followed by Bar Plot)</b></font>

# <font size="4.5">Cases</font>

# In[ ]:


## DT for Cases/Deaths
# Obtain case/death data for ALL days for ALL 50 states
allStateNames = (usStateData.state).to_numpy()
allStateNames = np.unique(allStateNames)

stateCaseData = (usStateData.cases).to_numpy()
stateDeathData = (usStateData.deaths).to_numpy()

numStates = allStateNames.shape[0]
doublingTimesCases = np.zeros(numStates,)
doublingTimesDeaths = np.zeros(numStates,)
timeSincePeak = np.zeros(numStates,)
recoveryTime = np.zeros(numStates,)

for i in range(numStates):
    caseData = stateCaseData[np.where(usStateData.state == allStateNames[i])]
    deathData = stateDeathData[np.where(usStateData.state == allStateNames[i])]
    
    # Get data to calculate days since peak
    allDTsCases = allDoublingTimes(caseData)
    allDTsDeaths = allDoublingTimes(deathData)
    
    DTPeaks = np.array(find_peaks(nDayAverage(allDTsCases, 5)))[0]
    valuesAtPeaks = allDTsCases[DTPeaks]
    if (np.amax(valuesAtPeaks) > 30):
        latestPeak = DTPeaks[np.where(valuesAtPeaks == np.amax(valuesAtPeaks[np.where(valuesAtPeaks >= 30)[0]]))[0][0]]        # Ensures that latest peak started at a DT above 30
    else:
        latestPeak = DTPeaks[-1]
    timeSincePeak[i] = caseData.shape[0] - latestPeak
    
    # Get recovery time data
    filteredDTsCases = nDayAverage(allDTsCases, 5)
    recoveryTime[i] = np.round(findTimeForRecovery(filteredDTsCases), 0)
    
    doublingTimesCases[i] = allDTsCases[-1]
    doublingTimesDeaths[i] = allDTsDeaths[-1]

# Concatenate the data
d = {'State': allStateNames.tolist(), 'DT_Cases': doublingTimesCases.tolist(), 'DT_Deaths': doublingTimesDeaths.tolist()}
doublingTimeData = pd.DataFrame(data=d)

d4 = {'State': allStateNames.tolist(), 'TimeSincePeak': timeSincePeak.tolist()}
timeSincePeakData = pd.DataFrame(data=d4)

d5 = {'State': fullStateNameToAbbrev(allStateNames), 'RecoveryTime': recoveryTime.tolist()}
recoveryTimeData = pd.DataFrame(data=d5)

maxDT_Cases = np.amax(doublingTimeData.DT_Cases)
maxDT_Deaths = np.amax(doublingTimeData.DT_Deaths)

maxTimeSincePeak = np.amax(timeSincePeak)
########################################################################################################################
# Get stability index data
stabIdxsCases = np.zeros(doublingTimeData.shape[0],)
for i in range(stabIdxsCases.shape[0]):
    stabIdxsCases[i] = math.exp(0.693 / doublingTimeData.DT_Cases[i])
    
d3 = {'State': doublingTimeData.State.tolist(), 'StabIdxCases': stabIdxsCases.tolist()}
stabIdxsCasesData = pd.DataFrame(data=d3)

maxStabIdx = np.amax(stabIdxsCasesData.StabIdxCases.to_numpy())
########################################################################################################################
# Create choropleth plots for doubling time of CASES
choroplethDTCases = go.Choropleth(
    locations=fullStateNameToAbbrev(doublingTimeData.State),
    locationmode='USA-states',
    z = doublingTimeData.DT_Cases,
    zmin = 0,
    zmax = maxDT_Cases,
    #colorscale=["red","yellow","green"],
    colorscale=[[0,'red'], [20/maxDT_Cases,'red'], [21/maxDT_Cases,'yellow'], [22/maxDT_Cases,'yellow'], [29/maxDT_Cases,'yellow'], [30/maxDT_Cases,'green'], [1,'green']],
    autocolorscale=False,
    text='Doubling Time (Days)', 
    marker_line_color='grey',
    colorbar_title="Doubling Time"
)
DTCasesPlot = go.Figure(data=choroplethDTCases)

DTCasesPlot.update_layout(
    title_text='Doubling Time for Cases per US State',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

DTCasesPlot.show()

# Create SORTED doubling time for cases plot with colors
dt_data = doublingTimeData.sort_values(by=['DT_Cases'], ascending=False)
DT_Cases_Int = (dt_data.DT_Cases).astype(int)

colors = np.full((dt_data.DT_Cases.shape[0],), 'blue')
for i in range (0,21):
    colors = np.where(DT_Cases_Int == i, 'red', colors)
for i in range (21,31):
    colors = np.where(DT_Cases_Int == i, 'yellow', colors)
colors = np.where(DT_Cases_Int >= 31, 'green', colors) 


doublingTimeCasesDataSORTED = go.Bar(x=dt_data.DT_Cases, y=dt_data.State, orientation='h', marker=dict(color=colors.tolist()))
doublingTimeCasesLayoutSORTED = go.Layout(title='Average Doubling Time for Cases in each US State/Territory - SORTED', 
                                    xaxis_title='Doubling Time for Cases', yaxis_title='State Name',
                                    width=800, height=1100)
doublingTimeCasesPlotSORTED = go.Figure(data=doublingTimeCasesDataSORTED, layout=doublingTimeCasesLayoutSORTED)
doublingTimeCasesPlotSORTED.add_shape(
    dict(
        type="line",
        x0=20,
        y0=-1,
        x1=20,
        y1=56,
        line=dict(
            color="Black",
            width=2
        )
))
doublingTimeCasesPlotSORTED.add_shape(
    dict(
        type="line",
        x0=30,
        y0=-1,
        x1=30,
        y1=56,
        line=dict(
            color="Black",
            width=2
        )
))
doublingTimeCasesPlotSORTED.show()
########################################################################################################################
# Create choropleth for Stability Index data
choroplethStabIdx = go.Choropleth(
    locations=fullStateNameToAbbrev(stabIdxsCasesData.State),
    locationmode='USA-states',
    z = stabIdxsCasesData.StabIdxCases,
    zmin = 0,
    zmax = maxStabIdx,
    #colorscale=["green","yellow","red"],
    colorscale=[[0,'green'],[math.exp(0.693/30)/maxStabIdx, 'green'],[(math.exp(0.693/30) + 0.001)/maxStabIdx,'yellow'],[math.exp(0.693/21)/maxStabIdx,'yellow'],[(math.exp(0.693/21) + 0.001)/maxStabIdx,'red'],[1,'red']],
    autocolorscale=False,
    text='Stability Index (Cases)', 
    marker_line_color='grey',
    colorbar_title="Stability Index (Cases)"
)
stabIdxPlot = go.Figure(data=choroplethStabIdx)

stabIdxPlot.update_layout(
    title_text='Stability Index for Cases per US State',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

#stabIdxPlot.show()

# Create bar plot for stability index data
stabIdx_SORTED = stabIdxsCasesData.sort_values(by=['StabIdxCases'], ascending=True)
stabIdxs = np.round(stabIdx_SORTED.StabIdxCases, 3)

colors = np.full((stabIdx_SORTED.StabIdxCases.shape[0],), 'green')
i = math.exp(0.693/30)
while (i < math.exp(0.693/21)):
    i = np.round(i, 3)           # Fix round off error
    colors = np.where(stabIdxs == i, 'yellow', colors)
    i += 0.001
colors = np.where(stabIdxs > math.exp(0.693/21), 'red', colors) 


stabIdxData_SORTED = go.Bar(x=stabIdx_SORTED.StabIdxCases, y=stabIdx_SORTED.State, orientation='h', marker=dict(color=colors))
stabIdxLayout_SORTED = go.Layout(title='Stability Index for Cases per US State', 
                                    xaxis_title='Stability Index', yaxis_title='State Name',
                                    width=800, height=1100)
stabIdxPlotSORTED = go.Figure(data=stabIdxData_SORTED, layout=stabIdxLayout_SORTED)
stabIdxPlotSORTED.add_shape(
    dict(
        type="line",
        x0=math.exp(0.693/21),
        y0=-1,
        x1=math.exp(0.693/21),
        y1=56,
        line=dict(
            color="Black",
            width=2
        )
))
stabIdxPlotSORTED.add_shape(
    dict(
        type="line",
        x0=math.exp(0.693/30),
        y0=-1,
        x1=math.exp(0.693/30),
        y1=56,
        line=dict(
            color="Black",
            width=2
        )
))

########################################################################################################################
# Recovery time plots
# Remove -1 and 365 values from data (just don't color these states)
a = recoveryTimeData[recoveryTimeData.RecoveryTime != -1]
b = a[a.RecoveryTime != 365]
c = b[b.RecoveryTime > 0]
maxRecoveryTime = np.amax(c.RecoveryTime)

# Create choropleth for recovery time
choroplethRecoveryTime = go.Choropleth(
    locations=c.State,
    locationmode='USA-states',
    z = c.RecoveryTime,
    zmin = 0,
    zmax = maxRecoveryTime,
    #colorscale=["green","yellow","red"],
    colorscale=[[0,'green'], [20/maxRecoveryTime,'green'], [21/maxRecoveryTime,'yellow'], [min(30/maxRecoveryTime,1),'yellow'], [min(31/maxRecoveryTime,1),'red'], [1,'red']],
    autocolorscale=False,
    text='Recovery Time (Days)', 
    marker_line_color='grey',
    colorbar_title="Recovery Time (Days)"
)
recoveryTimePlot = go.Figure(data=choroplethRecoveryTime)

recoveryTimePlot.update_layout(
    title_text='Estimated Recovery Times for Recovering US States',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

recoveryTimePlot.show()

# print('Recovery time of -1: State has already recovered')
# print('365: State is getting worse, no recovery in progress')


# <font size="4.5">Deaths</font>

# In[ ]:


# Get stability index data for DEATHS
stabIdxsDeaths = np.zeros(doublingTimeData.shape[0],)
for i in range(stabIdxsDeaths.shape[0]):
    stabIdxsDeaths[i] = math.exp(0.693 / doublingTimeData.DT_Deaths[i])
    
d3 = {'State': doublingTimeData.State.tolist(), 'StabIdxDeaths': stabIdxsDeaths.tolist()}
stabIdxsDeathsData = pd.DataFrame(data=d3)

maxStabIdx = np.amax(stabIdxsDeathsData.StabIdxDeaths.to_numpy()) + 0.005
##############################################################################################################
# Create choropleth plots for doubling time of DEATHS
choroplethDTDeaths = go.Choropleth(
    locations=fullStateNameToAbbrev(doublingTimeData.State),
    locationmode='USA-states',
    z = doublingTimeData.DT_Deaths,
    zmin = 0,
    zmax = maxDT_Deaths,
    colorscale=[[0,'red'], [20/maxDT_Deaths,'red'], [21/maxDT_Deaths,'yellow'], [30/maxDT_Deaths,'yellow'], [31/maxDT_Deaths,'green'], [1,'green']],
    autocolorscale=False,
    text='Doubling Time (Days)', 
    marker_line_color='grey',
    colorbar_title="Doubling Time"
)
DTDeathsPlot = go.Figure(data=choroplethDTDeaths)

DTDeathsPlot.update_layout(
    title_text='Doubling Time for Deaths per US State',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

DTDeathsPlot.show()

# Create bar plots for doubling times of DEATHS
dt_data = doublingTimeData.sort_values(by=['DT_Deaths'], ascending=False)
colors = np.full((dt_data.DT_Deaths.shape[0],), 'blue')
DT_Deaths_Int = (dt_data.DT_Deaths).astype(int)

for i in range (0,21):
    colors = np.where(DT_Deaths_Int == i, 'red', colors)
for i in range (21,31):
    colors = np.where(DT_Deaths_Int == i, 'yellow', colors)
colors = np.where(DT_Deaths_Int >= 31, 'green', colors) 

doublingTimeDeathsDataSORTED = go.Bar(x=dt_data.DT_Deaths, y=dt_data.State, orientation='h', marker=dict(color=colors.tolist()))
doublingTimeDeathsLayoutSORTED = go.Layout(title='Average Doubling Time for Deaths in each US State/Territory - SORTED', 
                                    xaxis_title='Doubling Time for Deaths', yaxis_title='State Name',
                                    width=800, height=1100)
doublingTimeDeathsPlotSORTED = go.Figure(data=doublingTimeDeathsDataSORTED, layout=doublingTimeDeathsLayoutSORTED)
doublingTimeDeathsPlotSORTED.add_shape(
    dict(
        type="line",
        x0=20,
        y0=-1,
        x1=20,
        y1=56,
        line=dict(
            color="Black",
            width=2
        )
))
doublingTimeDeathsPlotSORTED.add_shape(
    dict(
        type="line",
        x0=30,
        y0=-1,
        x1=30,
        y1=56,
        line=dict(
            color="Black",
            width=2
        )
))
doublingTimeDeathsPlotSORTED.show()


# <font size="4.5">Hospitalizations</font>

# In[ ]:


#########################################################################################################################
## DT for Hospitalizations
# Get ALL data, state, and hospitalizedCumulative data and make new DataFrame
covidTrackingDates = (covidTrackingDaily.date).to_numpy()
covidTrackingState = (covidTrackingDaily.state).to_numpy()
covidTrackingHosCumul = (covidTrackingDaily.hospitalizedCumulative).to_numpy()
dataLen = covidTrackingState.shape[0]

# Get full state names
fullStateNames = []
for i in range(dataLen):
    fullName = eval('us.states.' + covidTrackingState[i] + '.name')
    fullStateNames.append(fullName)
    
covidTrackingState = fullStateNames
uniqueStateNames = np.unique(fullStateNames)
numStates = uniqueStateNames.shape[0]

d = {'Date': covidTrackingDates.tolist(), 'State': covidTrackingState, 'HosCumul': covidTrackingHosCumul.tolist()}
hosCumulData = pd.DataFrame(data=d)

doublingTimesHos = np.zeros(numStates,)

# Get hosCumul for each state for all days and compute the doubling times
for i in range(numStates):
    hosCumulAllDays = (hosCumulData.HosCumul[np.where(hosCumulData.State == uniqueStateNames[i])[0]]).to_numpy()
    currentHosCumul = hosCumulAllDays[0]
    halfOfCurrentHosCumul = currentHosCumul / 2.0
    
    hosCumul_MinusHalfOfCurrentHosCumul = abs(hosCumulAllDays - halfOfCurrentHosCumul)
    
    idxApproxHalfOfHos = np.where(hosCumul_MinusHalfOfCurrentHosCumul == np.min(hosCumul_MinusHalfOfCurrentHosCumul))[0][0]

    doublingTimesHos[i] = idxApproxHalfOfHos
    
    
# Concatenate the data
d2 = {'State': uniqueStateNames.tolist(), 'DT_Hos': doublingTimesHos.tolist()}
doublingTimeHos = pd.DataFrame(data=d2)

maxDT_Cases = np.amax(doublingTimeData.DT_Cases)
maxDT_Deaths = np.amax(doublingTimeData.DT_Deaths)
maxDT_Hos = np.amax(doublingTimeHos.DT_Hos)

b = doublingTimeHos[doublingTimeHos.DT_Hos > 0]
######################################################################################################################### 
######################################################################################################################### 
# Create choropleth plots for doubling time of HOSPITALIZATIONS
choroplethDTHos = go.Choropleth(
    locations=fullStateNameToAbbrev(b.State.to_numpy()),
    locationmode='USA-states',
    z = b.DT_Hos,
    zmin = 0,
    zmax = maxDT_Hos,
    colorscale=[[0,'red'], [20/maxDT_Hos,'red'], [21/maxDT_Hos,'yellow'], [22/maxDT_Hos,'yellow'], [29/maxDT_Hos,'yellow'], [30/maxDT_Hos,'green'], [1,'green']],
    autocolorscale=False,
    text='Doubling Time (Days)', 
    marker_line_color='grey',
    colorbar_title="Doubling Time"
)
DTHosPlot = go.Figure(data=choroplethDTHos)

DTHosPlot.update_layout(
    title_text='Doubling Time for Hospitalizations per US State',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

DTHosPlot.show()

# Create SORTED doubling time for hospitalizations plot
dt_data = doublingTimeHos.sort_values(by=['DT_Hos'], ascending=False)
colors = np.full((dt_data.DT_Hos.shape[0],), 'blue')
DT_Hos_Int = (dt_data.DT_Hos).astype(int)

for i in range (0,21):
    colors = np.where(DT_Hos_Int == i, 'red', colors)
for i in range (21,31):
    colors = np.where(DT_Hos_Int == i, 'yellow', colors)
colors = np.where(DT_Hos_Int >= 31, 'green', colors) 

doublingTimeHosDataSORTED = go.Bar(x=dt_data.DT_Hos, y=dt_data.State, orientation='h', marker=dict(color=colors.tolist()))
doublingTimeHosLayoutSORTED = go.Layout(title='Average Doubling Time for Hospitalizations in each US State/Territory - SORTED', 
                                    xaxis_title='Doubling Time for Hospitalizations', yaxis_title='State Name',
                                    width=800, height=1000)
doublingTimeHosPlotSORTED = go.Figure(data=doublingTimeHosDataSORTED, layout=doublingTimeHosLayoutSORTED)
doublingTimeHosPlotSORTED.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= 30, x1= 30,
    )
])

doublingTimeHosPlotSORTED.show()


# <font size="5"><b>ALL Doubling Times for all US States (with 7 Day Predictions)</b></font>

# In[ ]:


def kalmanStep(x_k_k, v_k_k, z_kp1, P_k_k, twoState, a_k_k=0, q=1, r=1):
    if (twoState):
        I = np.identity(2)
        F = np.array([[1,1],[0,1]])
        H = np.array([1,0]).reshape(1,2)
        Q = np.array([[0,0],[0,q]])

        temp = np.array([x_k_k,v_k_k]).reshape(2,1)
        temp1 = np.matmul(F, temp)

        P_kp1_k = np.matmul(np.matmul(F, P_k_k), np.transpose(F)) + Q
        sigmaX = math.sqrt(P_kp1_k[0,0])
        sigmaV = math.sqrt(P_kp1_k[1,1])

        S_kp1 = np.matmul(np.matmul(H, P_kp1_k), np.transpose(H)) + r

        Ws = np.matmul(P_kp1_k, np.transpose(H)) / S_kp1
        Ws = Ws.reshape(2,1)         # For proper matrix multiplication
        
        temp = temp1 + np.matmul(Ws, z_kp1 - np.matmul(H, temp1))
        
        x_kp1_kp1 = temp[0]
        v_kp1_kp1 = temp[1]

        I_minus_WH = I - np.matmul(Ws,H)
        P_kp1_kp1 = np.matmul(np.matmul(I_minus_WH, P_kp1_k), np.transpose(I_minus_WH)) + np.matmul(Ws, r*np.transpose(Ws))
        
        return x_kp1_kp1, v_kp1_kp1, P_kp1_kp1, sigmaX, sigmaV
    else:
        # Three state model
        I = np.identity(3)
        
        F = np.array([[1,1,0.5],[0,1,1],[0,0,1]])
        gamma = np.array([0.1667,0.5,1]).reshape(3,1)
        Q = q * np.matmul(gamma, np.transpose(gamma))
        H = np.array([1,0,0]).reshape(1,3)
        
        temp = np.array([x_k_k,v_k_k,a_k_k]).reshape(3,1)
        temp1 = np.matmul(F, temp)

        P_kp1_k = np.matmul(np.matmul(F, P_k_k), np.transpose(F)) + Q
           
        sigmaX = math.sqrt(P_kp1_k[0,0])
        sigmaV = math.sqrt(P_kp1_k[1,1])
        sigmaA = math.sqrt(P_kp1_k[2,2])

        S_kp1 = np.matmul(np.matmul(H, P_kp1_k), np.transpose(H)) + r

        Ws = np.matmul(np.matmul(P_kp1_k, np.transpose(H)), np.linalg.inv(S_kp1))
        Ws = Ws.reshape(3,1)         # For proper matrix multiplication
        
        temp = temp1 + np.matmul(Ws, z_kp1 - np.matmul(H, temp1))
        
        x_kp1_kp1 = temp[0]
        v_kp1_kp1 = temp[1]
        a_kp1_kp1 = temp[2]

        I_minus_WH = I - np.matmul(Ws,H)
        P_kp1_kp1 = np.matmul(np.matmul(I_minus_WH, P_kp1_k), np.transpose(I_minus_WH)) + np.matmul(Ws, r*np.transpose(Ws))
        
        return x_kp1_kp1, v_kp1_kp1, a_kp1_kp1, P_kp1_kp1, sigmaX, sigmaV, sigmaA
    
    return None


# In[ ]:


def kalmanPrediction(x_final, v_final, p_final, L, F, Q, a_final=0):
    #Q = np.array([[0,0],[0,q]])
    #F = np.array([[1,1],[0,1]])
    numStates = F.shape[0]
    
    pPreds = np.zeros((L,numStates,numStates))
    pPreds[0] = p_final
    
    if (numStates == 2):
        xPreds = np.zeros(L,)
        vPreds = np.zeros(L,)
        sigmaPredXs = np.zeros(L,)
        sigmaPredVs = np.zeros(L,)
        xPreds[0] = x_final 
        vPreds[0] = v_final
        
        for i in range(1,L):
            xPreds[i] = xPreds[i-1] + vPreds[i-1]
            vPreds[i] = vPreds[i-1]
            pPreds[i] = np.matmul(np.matmul(F, pPreds[i-1]), np.transpose(F)) + Q
            sigmaPredXs[i] = math.sqrt(pPreds[i,0,0])
            sigmaPredVs[i] = math.sqrt(pPreds[i,1,1])
        
        return xPreds, vPreds, pPreds, sigmaPredXs, sigmaPredVs
    else:
        xPreds = np.zeros(L,)
        vPreds = np.zeros(L,)
        aPreds = np.zeros(L,)
        sigmaPredXs = np.zeros(L,)
        sigmaPredVs = np.zeros(L,)
        sigmaPredAs = np.zeros(L,)
        xPreds[0] = x_final 
        vPreds[0] = v_final
        aPreds[0] = a_final
        
        for i in range(1,L):
            xPreds[i] = xPreds[i-1] + vPreds[i-1] + 0.5*aPreds[i-1]
            vPreds[i] = vPreds[i-1] + aPreds[i-1]
            aPreds[i] = aPreds[i-1]
            pPreds[i] = np.matmul(np.matmul(F, pPreds[i-1]), np.transpose(F)) + Q
            sigmaPredXs[i] = math.sqrt(pPreds[i,0,0])
            sigmaPredVs[i] = math.sqrt(pPreds[i,1,1])
            sigmaPredAs[i] = math.sqrt(pPreds[i,2,2])
        
        return xPreds, vPreds, aPreds, pPreds, sigmaPredXs, sigmaPredVs, sigmaPredAs
    
    return None


# In[ ]:


def kalmanFilter(measuredData, numSteps, numPreds, twoState, a_0_0=1, q=1, r=1):  
    x_0_0 = measuredData[0]
    v_0_0 = np.diff(measuredData)[0]
    
    if (twoState):
        F = np.array([[1,1],[0,1]])
        Q = np.array([[0,0],[0,q]])
        
        xs = np.zeros(numSteps,)
        vs = np.zeros(numSteps,)
        Ps = np.zeros((numSteps,2,2))
        sigmaXs = np.zeros(numSteps + numPreds,)
        sigmaVs = np.zeros(numSteps + numPreds,)
        
        xs[0] = x_0_0
        vs[0] = v_0_0
        Ps[0] = 1e6 * np.identity(2)
        
        # Call step function to filter on measured data
        for i in range(1,numSteps):
            xs[i], vs[i], Ps[i], sigmaXs[i], sigmaVs[i] = kalmanStep(xs[i-1],vs[i-1],measuredData[i],Ps[i-1], q, r)
            
        # Then compute numPreds day predictions
        xPred, vPred, pPred, sigmaPredXs, sigmaPredVs = kalmanPrediction(xs[-1], vs[-1], Ps[-1], numPreds, F, Q)
        
        # Add sigmas from predictions to main sigma arrays
        sigmaXs[numSteps:] = sigmaPredXs
        sigmaVs[numSteps:] = sigmaPredVs
        
        # Return necessary variables
        return xs, vs, Ps, xPred, vPred, pPred, sigmaXs, sigmaVs
    else:
        gamma = np.array([0.1667,0.5,1]).reshape(3,1)
        Q = q * np.matmul(gamma, np.transpose(gamma))
        F = np.array([[1,1,0.5],[0,1,1],[0,0,1]])
        
        xs = np.zeros(numSteps,)
        vs = np.zeros(numSteps,)
        As = np.zeros(numSteps,)
        Ps = np.zeros((numSteps,3,3))
        sigmaXs = np.zeros(numSteps + numPreds,)
        sigmaVs = np.zeros(numSteps + numPreds,)
        sigmaAs = np.zeros(numSteps + numPreds,)
        
        xs[0] = x_0_0
        vs[0] = v_0_0
        As[0] = a_0_0
        Ps[0] = 1e6 * np.identity(3)
        
        # Call step function to filter on measured data
        for i in range(1,numSteps):
            xs[i], vs[i], As[i], Ps[i], sigmaXs[i], sigmaVs[i], sigmaAs[i] = kalmanStep(xs[i-1],vs[i-1],usCases[i],Ps[i-1], False, q, r)
        
        # Then compute numPreds day predictions
        xPred, vPred, aPred, pPred, sigmaPredXs, sigmaPredVs, sigmaPredAs = kalmanPrediction(xs[-1], vs[-1], Ps[-1], numPreds, F, Q, a_final=As[-1])
    
        # Add sigmas from predictions to main sigma arrays
        sigmaXs[numSteps:] = sigmaPredXs
        sigmaVs[numSteps:] = sigmaPredVs
        sigmaAs[numSteps:] = sigmaPredAs
        
        # Return necessary variables
        return xs, vs, As, Ps, xPred, vPred, aPred, pPred, sigmaXs, sigmaVs, sigmaAs
    
    return None


# In[ ]:


def confidence(value, error, desired):
    maxValue = value + error
    errorRange = 2 * error
    
    confidence = min(max((maxValue - desired) / errorRange * 100, 0), 100)
    
    return confidence


# In[ ]:


# Obtain case/death data for ALL days for ALL 50 states
allStateNames = (usStateData.state).to_numpy()
allStateNames = np.unique(allStateNames)

stateCaseData = (usStateData.cases).to_numpy()
stateDeathData = (usStateData.deaths).to_numpy()

numStates = allStateNames.shape[0]
doublingTimesCases = np.zeros(numStates,)
doublingTimesDeaths = np.zeros(numStates,)

for i in range(numStates):
    caseData = stateCaseData[np.where(usStateData.state == allStateNames[i])]
    deathData = stateDeathData[np.where(usStateData.state == allStateNames[i])]
    
    doublingTimesCases[i] = allDoublingTimes(caseData)[-1]
    doublingTimesDeaths[i] = allDoublingTimes(deathData)[-1]

# Concatenate the data
d = {'State': allStateNames.tolist(), 'DT_Cases': doublingTimesCases.tolist(), 'DT_Deaths': doublingTimesDeaths.tolist()}
doublingTimeData = pd.DataFrame(data=d)


# Create SORTED doubling time for cases plot with colors
dt_data = doublingTimeData.sort_values(by=['DT_Cases'], ascending=False)
DT_Cases_Int = (dt_data.DT_Cases).astype(int)


# Plot ALL doubling times for states with DTs under 30 days
a = (dt_data.DT_Cases).to_numpy()
b = np.flip(a)
length = np.split(b, np.where(b >= 22)[0])[0].shape[0]
redStates = np.flip(dt_data.State)[0:length]
redStates = redStates.to_numpy()

allStates = np.unique(dt_data.State)
numStates = allStates.shape[0]

twoState = True
numPreds = 7

# Get next7Days as string
next7Days_str = []
date = datetime.datetime(2020,8,19)              # Latest date available in data
for i in range(7): 
    next7Days_str.append(date.strftime('%Y-%m-%d'))
    date += datetime.timedelta(days=1)

for i in range(numStates):
    stateData = usStateData.loc[usStateData.state == allStates[i]]
    
    caseData = stateData.cases.to_numpy()
    deathData = stateData.deaths.to_numpy()
    dates = stateData.date
    allDT_Cases = allDoublingTimes(caseData)
    allDT_Deaths = allDoublingTimes(deathData)
    
    numDays = allDT_Cases.shape[0]
    
    filteredDT_Cases = nDayAverage(allDT_Cases, 5)
    filteredDT_Deaths = nDayAverage(allDT_Deaths, 5)
    
    # Run Kalman Filter for predictions
    kalFilt_DTCases, kalFilt_DeltaDTCases, kalFiltPs, DT_CasesPreds, vPreds, pPreds, sigmaXs, sigmaVs = kalmanFilter(filteredDT_Cases, numDays, numPreds, twoState, q=0.1, r=0.1)
    kalFilt_DTDeaths, kalFilt_DeltaDTDeaths, kalFiltPs, DT_DeathsPreds, vPreds, pPreds, sigmaXs, sigmaVs = kalmanFilter(filteredDT_Deaths, numDays, numPreds, twoState, q=0.1, r=0.1)
    DT_CasesPreds = np.round(DT_CasesPreds, 3)
    DT_DeathsPreds = np.round(DT_DeathsPreds, 3)
    
    
    allDT_CasesData = go.Scatter(x=dates, y=allDT_Cases, name='Doubling Times for Cases')
    allDT_CasesFilteredData = go.Scatter(x=dates, y=filteredDT_Cases, name='Doubling Times for Cases (5 Day Average)')
    DT_CasesPredsData = go.Scatter(x=next7Days_str, y=DT_CasesPreds, name='Doubling Times for Cases (Kalman Filter Pred)')
    
    allDT_DeathsData = go.Scatter(x=dates, y=allDT_Deaths, name='Doubling Times for Deaths')
    allDT_DeathsFilteredData = go.Scatter(x=dates, y=filteredDT_Deaths, name='Doubling Times for Deaths (5 Day Average)')
    DT_DeathsPredsData = go.Scatter(x=next7Days_str, y=DT_DeathsPreds, name='Doubling Times for Deaths (Kalman Filter Pred)')
    
    thirtyDaysLine = go.Scatter(x=np.concatenate((dates,next7Days_str)), y=np.ones(dates.shape[0] + numPreds,)*30, name='30 Days Line')
    #DTPeaksData = go.Scatter(x=DTPeaks, y=allDTs[DTPeaks], name='Peaks')
    
    allDTsLayout = go.Layout(title='ALL Doubling Times in ' + allStates[i], 
                             xaxis_title='Date', yaxis_title='Doubling Time for Cases',
                             width=800, height=600)
    allDTsPlot = go.Figure(data=[allDT_CasesData,allDT_CasesFilteredData,DT_CasesPredsData,allDT_DeathsData,allDT_DeathsFilteredData,DT_DeathsPredsData,thirtyDaysLine], layout=allDTsLayout)
    
    allDTsPlot.show()


# In[ ]:


#data = filteredDTs
stateName = 'Alaska'
caseData = stateCaseData[np.where(usStateData.state == stateName)]
allDTs = allDoublingTimes(caseData)
    
filteredDTs = nDayAverage(allDTs, 5)
data = filteredDTs

# Find global max (right before things went bad)
if ((np.diff(data[-45:]) > -0.01).all()):  # This is required for good states where DT is continuously rising (at least for the last 30 days)
    maximas = data.shape[0] - 1
else:
    maximas = find_peaks(data)[0]
    #maximas = argrelextrema(data, np.greater_equal)
dataAtMaximas = data[maximas]
globalMaxIdx = np.where(data == np.amax(dataAtMaximas))[0][0]
globalMax = np.mean(data[globalMaxIdx])                        # To account for two of the same maxs found

# Then find minima on the data that comes after the maxima
clippedData = data[globalMaxIdx:]
minimas = argrelextrema(clippedData, np.less_equal)[0]
latestMinIdxClipped = np.where(clippedData == np.amin(clippedData[minimas]))[0][0]
latestMinIdx = globalMaxIdx + latestMinIdxClipped
latestMin = np.mean(clippedData[latestMinIdxClipped])

# Find avg slope for increase
avgIncPerDay = 0
if (latestMinIdx != data.shape[0] and data[-1] - data[latestMinIdx] != 0):
    avgIncPerDay = (data[-1] - data[latestMinIdx]) / (data.shape[0] - latestMinIdx)
    incReqForRecovery = globalMax - data[-1]
    timeForRecovery = incReqForRecovery / avgIncPerDay                                         # Make sure to take into account of today's DT
if (latestMinIdx == data.shape[0] or avgIncPerDay < 0.1):
    timeForRecovery = 365
if (data[-1] - data[latestMinIdx] == 0):
    timeForRecovery = 365
if (globalMax == latestMin):
    timeForRecovery = -1
    
plt.plot(data)
plt.plot(globalMaxIdx,globalMax,'ko')
plt.plot(latestMinIdx,latestMin,'ro')
plt.title('Doubling Times of Cases in ' + stateName)
plt.xlabel('Days since Data Collection')
plt.ylabel ('Doubling Times')
plt.legend(['Data','Doubling Time Peak','Worst Doubling Time Since Peak'])

print(globalMax,latestMin,avgIncPerDay,timeForRecovery)


# <font size="5"><b>Find the Percentage of Positive Tests for each state (7 Day Average)</b></font>

# In[ ]:


def positivityRates(positive, negative, pending, latestFirst=True):  
    if (latestFirst):
        newPositiveCases = -np.diff(positive)
        newNegativeCases = -np.diff(negative)
        newPendingCases = -np.diff(pending)
    else:
        newPositiveCases = np.diff(positive)
        newNegativeCases = np.diff(negative)
        newPendingCases = np.diff(pending)
    
    # Remove any negative numbers
    newPositiveCases = np.clip(newPositiveCases, a_min=0, a_max=np.inf)
    newNegativeCases = np.clip(newNegativeCases, a_min=0, a_max=np.inf)
    newPendingCases = np.clip(newPendingCases, a_min=0, a_max=np.inf)
    
    positivityRates = np.zeros(newPositiveCases.shape[0],)
    
    totalNewCases = newPositiveCases + newNegativeCases + newPendingCases
    
    positivityRates = np.divide(newPositiveCases, totalNewCases) * 100
    
    #print(newPositiveCases,totalNewCases)
    
#     if (not totalNewCases.any()):
#         positivityRates = np.divide(newPositiveCases, totalNewCases) * 100
#     else:
#         positivityRates = np.divide(newPositiveCases, totalNewCases + 1) * 100
    
    # if any div by 0 errors
    positivityRates[np.isnan(positivityRates)] = 0
    positivityRates[positivityRates == np.inf] = 0
    
    return positivityRates


# In[ ]:


from plotly.subplots import make_subplots

state = 'TX'
dates = covidTrackingDaily.loc[covidTrackingDaily['state'] == state].date

posCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == state].positive
negCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == state].negative
pendingCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == state].pending

# newPositiveCases = -np.diff(posCasesForState)
# newNegativeCases = -np.diff(negCasesForState)
# newPendingCases = -np.diff(pendingCasesForState)

# newPositiveCases = np.clip(newPositiveCases, a_min=0, a_max=np.inf)
# newNegativeCases = np.clip(newNegativeCases, a_min=0, a_max=np.inf)
# newPendingCases = np.clip(newPendingCases, a_min=0, a_max=np.inf)

# totalNewCases = newPositiveCases + newNegativeCases + newPendingCases

# posRates = np.divide(newPositiveCases, totalNewCases) * 100
# posRates[np.isnan(posRates)] = 0
# posRates[posRates == np.inf] = 0
# posRates[posRates == 100] = 0            # remove 100 % pos

posRates = positivityRates(posCasesForState, negCasesForState, np.zeros(posCasesForState.shape[0]))
posRates[posRates == 100] = 0            # remove 100 % pos

i = 0
window_size = 7
moving_averages = []
while i < len(posRates[::-1]) - window_size + 1:
    this_window = posRates[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=dates[1:], y=newPositiveCases, name='Pos Cases'), secondary_y=False)
fig.add_trace(go.Bar(x=dates[1:], y=totalNewCases, name='Total Cases'), secondary_y=False)
fig.add_trace(go.Scatter(x=dates[1:], y=posRates, name='Positivity Rates'), secondary_y=True)
#fig.add_trace(go.Scatter(x=dates[::-1][1:], y=nDayAverage(posRates[::-1],7), name='7 Day Avg'), secondary_y=True)
fig.add_trace(go.Scatter(x=dates[1:], y=moving_averages, name='7 Day Avg v2'), secondary_y=True)

posRatesLayout = go.Layout(title='Test Plot')

#fig = go.Figure(data=[data1,data2,avg], layout=posRatesLayout)
fig.update_layout(barmode='stack', title='State: ' + state)
fig.update_yaxes(title_text="Counts", secondary_y=False)
fig.update_yaxes(title_text="Positivity Rates", secondary_y=True)
fig.show()


# In[ ]:


# Get today's data for covidTrackingDaily data table
todayDate = '2020-08-19'
covidTrackingDates = (covidTrackingDaily.date).to_numpy()

covidTrackingDataToday = covidTrackingDaily.loc[covidTrackingDaily.date == todayDate]

# Get NEW  pos/neg data from today's data
positiveData = (covidTrackingDataToday.positive).to_numpy()
negativeData = (covidTrackingDataToday.negative).to_numpy()
totalCases = (covidTrackingDataToday.posNeg).to_numpy()

# Get state names
covidTrackingState = (covidTrackingDaily.state).to_numpy()
states = np.unique(covidTrackingState)
numStates = states.shape[0]

percentPositive = np.zeros(numStates,)

for i in range(numStates):
    posCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == states[i]].positive
    negCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == states[i]].negative
    pendingCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == states[i]].pending
    
    posRates = positivityRates(posCasesForState, negCasesForState, pendingCasesForState)
    percentPositive[i] = np.mean(posRates[0:7])

# Concatenate the data
d = {'State': states.tolist(), 'PercentPos': percentPositive.tolist()}
percentPositiveData = pd.DataFrame(data=d)

# Get full state names
allStateNames = np.unique(allStateNames)

fullStateNames = []
for i in range(numStates):
    fullName = eval('us.states.' + covidTrackingState[i] + '.name')
    fullStateNames.append(fullName)
    
percentPositiveData.State = fullStateNames
    

# Chorpleth plot for % positive cases
choroplethPerPosData = go.Choropleth(
    locations=fullStateNameToAbbrev(percentPositiveData.State),
    locationmode='USA-states',
    z = np.round(percentPositiveData.PercentPos, 2),
    zmin = 0,
    zmax = 100,
    #colorscale=["green","yellow","red"],
    colorscale=[[0,'green'], [0.05,'green'], [0.06,'yellow'], [0.1,'yellow'], [0.11,'red'], [1,'red']],
    autocolorscale=False,
    text='% Positive', 
    marker_line_color='grey',
    colorbar_title="Percent Positive"
)
choroplethPerPos = go.Figure(data=choroplethPerPosData)

choroplethPerPos.update_layout(
    title_text='Latest Percent Positive Tests per US State (7 Day Average)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

choroplethPerPos.show()

# Plot % positive cases SORTED with colors
percentPositiveData_Sorted = percentPositiveData.sort_values(by=['PercentPos'], ascending=True)
percentPositiveData_Sorted_Int = percentPositiveData_Sorted.PercentPos.astype(int)

# Bin sorting
colors = np.full((percentPositiveData_Sorted.PercentPos.shape[0],), 'blue')
for i in range (0,6):
    colors = np.where(percentPositiveData_Sorted_Int == i, 'green', colors)
for i in range (5,11):
    colors = np.where(percentPositiveData_Sorted_Int == i, 'yellow', colors)
colors = np.where(percentPositiveData_Sorted.PercentPos > 10, 'red', colors) 

perPosDataSorted = go.Bar(x=percentPositiveData_Sorted.PercentPos, y=percentPositiveData_Sorted.State, orientation='h', 
                          marker=dict(color=colors))
perPosDataSortedLayout = go.Layout(title='% Positive Tests (7 Day Average) for all States - SORTED', 
                                    xaxis_title='% Positive Tests', yaxis_title='State Name',
                                    width=800, height=1100)
perPosSortedPlot = go.Figure(data=perPosDataSorted, layout=perPosDataSortedLayout)
perPosSortedPlot.add_shape(
    dict(
        type="line",
        x0=5,
        y0=-1,
        x1=5,
        y1=56,
        line=dict(
            color="Black",
            width=3
        )
))
perPosSortedPlot.add_shape(
    dict(
        type="line",
        x0=10,
        y0=-1,
        x1=10,
        y1=56,
        line=dict(
            color="Black",
            width=3
        )
))
perPosSortedPlot.show()


# <font size="5"><b>Kalman Filter for US Cases</b></font>

# <font size="4">Two State Model</font>

# In[ ]:


numDays = usCases.shape[0]
twoState = True
numPreds = 7
kalFiltXs, kalFiltVs, kalFiltPs, xPreds, vPreds, pPreds, sigmaXs, sigmaVs = kalmanFilter(usCases, numDays, numPreds, twoState, q=100, r=10)


# In[ ]:


# # Get next7Days as string
# next7Days_str = []
# date = datetime.datetime(2020,8,18)              # Latest date available in data
# for i in range(7): 
#     next7Days_str.append(date.strftime('%Y-%m-%d'))
#     date += datetime.timedelta(days=1)

kalFiltCumulCasesData = go.Scatter(x=usData.date, y=kalFiltXs, name="Kalman Filter (Actual Data)")
actualCumulCasesData = go.Scatter(x=usData.date, y=usCases, name="Actual Data")

kalFiltCumulCasesPreds = go.Scatter(x=next7Days_str, y=xPreds, name="Kalman Filter (Predictions)", 
                                    error_y = dict(
                                    type = 'data', # value of error bar given in data coordinates
                                    array = sigmaXs[numDays:],
                                    visible = True)
                                   )

kalFiltAndActualLayout = go.Layout(title='US Cases over Time (with 7-day Predictions)', xaxis_title='Date', yaxis_title='# of Cases')
kalFiltAndActualPlot = go.Figure(data=[kalFiltCumulCasesData,actualCumulCasesData,kalFiltCumulCasesPreds], layout=kalFiltAndActualLayout)

kalFiltAndActualPlot.show()


# In[ ]:


kalFiltNewCasesData = go.Scatter(x=usData.date[1:], y=kalFiltVs, name="Kalman Filter (Actual Data)")
actualNewCasesData = go.Scatter(x=usData.date[1:], y=np.diff(usCases), name="Actual Data")
kalFiltNewCasesPreds = go.Scatter(x=next7Days_str, y=vPreds, name="Kalman Filter (Predictions)",
                                    error_y = dict(
                                    type = 'data', # value of error bar given in data coordinates
                                    array = sigmaVs[numDays:],
                                    visible = True)
                                 )

kalFiltAndActualLayout = go.Layout(title='New US Cases over Time (with 7-day Predictions)', xaxis_title='Date', yaxis_title='# of New Cases')
kalFiltAndActualPlot = go.Figure(data=[kalFiltNewCasesData,actualNewCasesData,kalFiltNewCasesPreds], layout=kalFiltAndActualLayout)

kalFiltAndActualPlot.show()


# <font size="4">Three State Model</font>

# In[ ]:


# Three state model
numDays = usCases.shape[0]
numPreds = 7
twoState = False

kalFiltXs, kalFiltVs, kalFiltAs, kalFiltPs, xPreds, vPreds, aPreds, pPreds, sigmaXs, sigmaVs, sigmaAs = kalmanFilter(usCases, numDays, numPreds, twoState, q=0.75, r=0.1)


# In[ ]:


kalFiltCumulCasesData = go.Scatter(x=usData.date, y=kalFiltXs, name="Kalman Filter (Actual Data)")
actualCumulCasesData = go.Scatter(x=usData.date, y=usCases, name="Actual Data")

kalFiltCumulCasesPreds = go.Scatter(x=next7Days_str, y=xPreds, name="Kalman Filter (Predictions)", 
                                    error_y = dict(
                                    type = 'data', # value of error bar given in data coordinates
                                    array = sigmaXs[numDays:],
                                    visible = True)
                                   )

kalFiltAndActualLayout = go.Layout(title='US Cases over Time (with 7-day Predictions)', xaxis_title='Date', yaxis_title='# of Cases')
kalFiltAndActualPlot = go.Figure(data=[kalFiltCumulCasesData,actualCumulCasesData,kalFiltCumulCasesPreds], layout=kalFiltAndActualLayout)

kalFiltAndActualPlot.show()


# In[ ]:


kalFiltNewCasesData = go.Scatter(x=usData.date[1:], y=kalFiltVs, name="Kalman Filter (Actual Data)")
actualNewCasesData = go.Scatter(x=usData.date[1:], y=np.diff(usCases), name="Actual Data")
kalFiltNewCasesPreds = go.Scatter(x=next7Days_str, y=vPreds, name="Kalman Filter (Predictions)",
                                    error_y = dict(
                                    type = 'data', # value of error bar given in data coordinates
                                    array = sigmaVs[numDays:],
                                    visible = True)
                                 )

kalFiltAndActualLayout = go.Layout(title='New US Cases over Time (with 7-day Predictions)', xaxis_title='Date', yaxis_title='# of New Cases')
kalFiltAndActualPlot = go.Figure(data=[kalFiltNewCasesData,actualNewCasesData,kalFiltNewCasesPreds], layout=kalFiltAndActualLayout)

kalFiltAndActualPlot.show()


# In[ ]:


deltaDailyCases = np.diff(np.diff(usCases))

kalFiltNewCasesData = go.Scatter(x=usData.date[2:], y=kalFiltAs, name="Kalman Filter (Actual Data)")
actualNewCasesData = go.Scatter(x=usData.date[2:], y=deltaDailyCases, name="Actual Data")
actualNewCasesFilteredData =go.Scatter(x=usData.date[2:], y=nDayAverage(deltaDailyCases, 7), name="Actual Data (7 Day Average)")
kalFiltNewCasesPreds = go.Scatter(x=next7Days_str, y=aPreds, name="Kalman Filter (Predictions)",
                                    error_y = dict(
                                    type = 'data', # value of error bar given in data coordinates
                                    array = sigmaAs[numDays:],
                                    visible = True)
                                 )

kalFiltAndActualLayout = go.Layout(title='Change in Daily US Cases over Time (with 7-day Predictions)', xaxis_title='Date', yaxis_title='Change in Daily Cases')
kalFiltAndActualPlot = go.Figure(data=[kalFiltNewCasesData,actualNewCasesData,actualNewCasesFilteredData,kalFiltNewCasesPreds], layout=kalFiltAndActualLayout)

kalFiltAndActualPlot.show()


# <font size="5"><b>New Cases/Deaths per 100k per US State (Geographical Plot)</b></font>

# In[ ]:


allStateNames = (usStateData.state).to_numpy()
allStateNames = np.unique(allStateNames)
numStates = allStateNames.shape[0]

populationData = pd.read_csv('../input/population-data/Population_v2.csv')
populationData = populationData.to_numpy()

avgNewStateCasesPer100k = np.zeros(numStates,)
avgNewStateDeathsPer100k = np.zeros(numStates,)

allStateNames_abbrev = []
for i in range(numStates):
    allStateCases = (usStateData.loc[usStateData.state == allStateNames[i]].cases).to_numpy()
    allStateDeaths = (usStateData.loc[usStateData.state == allStateNames[i]].deaths).to_numpy()
    
    # Get population data
    stateRow = populationData[np.where(populationData == allStateNames[i])[0]]
    statePop = stateRow[0,1]
    
    allNewCases = np.diff(allStateCases)
    allNewDeaths = np.diff(allStateDeaths)
    
    avgNewStateCasesPer100k[i] = np.mean(allNewCases[-7:]) / (statePop / 100000)
    avgNewStateDeathsPer100k[i] = np.mean(allNewDeaths[-7:]) / (statePop / 100000)
    
    allStateNames_abbrev.append(eval("us.states.lookup('" + str(allStateNames[i]) + "').abbr"))
        
    # Get population data
    stateRow = populationData[np.where(populationData == allStateNames[i])[0]]
    statePop = stateRow[0,1]
    
maxNewCasesPer100k = np.amax(avgNewStateCasesPer100k)
maxNewDeathsPer100k = np.amax(avgNewStateDeathsPer100k)
    
# Create choropleth plots for cases
choroplethNewCases = go.Choropleth(
    locations=allStateNames_abbrev,
    locationmode='USA-states',
    z = np.round(avgNewStateCasesPer100k, 3),
    colorscale=[[0,'green'], [4/maxNewCasesPer100k,'green'], [5/maxNewCasesPer100k,'yellow'], [10/maxNewCasesPer100k,'yellow'], [11/maxNewCasesPer100k,'red'], [1,'red']],
    autocolorscale=False,
    text='New Cases per 100k', 
    marker_line_color='grey',
    colorbar_title="New Cases Per 100k (7 Day Average)"
)
newCasesPlot = go.Figure(data=choroplethNewCases)

newCasesPlot.update_layout(
    title_text='New Cases per 100k per State (7 Day Average)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

newCasesPlot.show()


# Create choropleth plots for deaths
choroplethNewDeaths = go.Choropleth(
    locations=allStateNames_abbrev,
    locationmode='USA-states',
    z = np.round(avgNewStateDeathsPer100k, 3),
    #colorscale=[[0,'green'], [4/maxNewDeathsPer100k,'green'], [5/maxNewDeathsPer100k,'yellow'], [10/maxNewDeathsPer100k,'yellow'], [11/maxNewDeathsPer100k,'red'], [1,'red']],
    colorscale=["green","yellow","red"],
    autocolorscale=False,
    text='New Deaths per 100k', 
    marker_line_color='grey',
    colorbar_title="New Deaths per 100k (7 Day Average)"
)
newDeathsPlot = go.Figure(data=choroplethNewDeaths)

newDeathsPlot.update_layout(
    title_text='New Deaths per 100k per State (7 Day Average)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

newDeathsPlot.show()


# <font size="5"><b>Current Number of Individuals in ICU per State</b></font>

# In[ ]:


# Get ICU Data
todayDate = '2020-08-19'
covidTrackingToday = covidTrackingDaily.loc[covidTrackingDaily.date == todayDate]

icuFields = {'State': covidTrackingToday.state, 'inICU': covidTrackingToday.inIcuCurrently}
icuData = pd.DataFrame(data=icuFields)

# Remove states with 0 people (due to poor data reporting)
icuData = icuData[icuData.inICU != 0]

# Create choropleth
choroplethICU = go.Choropleth(
    locations=icuData.State,
    locationmode='USA-states',
    z = icuData.inICU,
    colorscale=["green","yellow","red"],
    autocolorscale=False,
    text='People in ICU', 
    marker_line_color='grey',
    colorbar_title="People in ICU"
)
icuPlot = go.Figure(data=choroplethICU)

icuPlot.update_layout(
    title_text='Number of Individuals in ICU per US State',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

icuPlot.show()


# <font size="5"><b>Current Metrics per US County (Geographical Plot)</b></font>

# In[ ]:


# Obtain unique FIPS codes
uniqueFIPS = np.unique(usCountyData['fips'])
uniqueFIPS = uniqueFIPS[~np.isnan(uniqueFIPS)]
length = uniqueFIPS.shape[0]

countyCases = np.zeros(length,)
countyDeaths = np.zeros(length,)

allStates = []
allCounties = []
fullNames = []

# Load in US County Population Data
usCountyPopData = pd.read_csv('../input/populationdatauscounties/usCountyPopulationData.csv')

for i in range(length):
    allCasesForCounty = (usCountyData.loc[usCountyData['fips'] == uniqueFIPS[i]].cases).to_numpy()
    allDeathsForCounty = (usCountyData.loc[usCountyData['fips'] == uniqueFIPS[i]].deaths).to_numpy()
    
    countyCases[i] = allCasesForCounty[-1]
    countyDeaths[i] = allDeathsForCounty[-1]
    
    stateName = (usCountyData.loc[usCountyData['fips'] == uniqueFIPS[i]].state).to_numpy()[0]
    countyName = (usCountyData.loc[usCountyData['fips'] == uniqueFIPS[i]].county).to_numpy()[0]
    fullName = countyName + ", " + stateName
    
    allStates.append(stateName)
    allCounties.append(countyName)
    fullNames.append(fullName)

# Fix FIPS codes to length 5 strings
int_fips = [int(uniqueFIPS) for uniqueFIPS in uniqueFIPS]
form_fips = ["%05d" % uniqueFIPS for uniqueFIPS in uniqueFIPS]   
numEntries = len(fullNames)
blankTableSpace = np.zeros(numEntries,).tolist()

d = {'State': allStates, 'County': allCounties, 'FullName': fullNames, 'FIPS': form_fips, 'Cases': countyCases.tolist(),
     'Population': blankTableSpace, 'CasesPer100k': blankTableSpace, 'DeathsPer100k': blankTableSpace,
     'Deaths': countyDeaths.tolist(), 'DT_Cases': blankTableSpace, 'DT_Deaths': blankTableSpace, 'SeverityIndex': blankTableSpace}
choroplethData = pd.DataFrame(data=d)    
    
maxDT_Cases = np.amax(choroplethData.DT_Cases)
        
# Compute doubling times for cases and deaths per county
stateNames = np.unique(usCountyData.state)
numStates = stateNames.shape[0]

for i in range(numStates):
    stateData = usCountyData.loc[usCountyData.state == stateNames[i]]
    stateCounties = np.unique(stateData.county)
    stateCounties = np.delete(stateCounties, np.where(stateCounties == 'Unknown'))           # Don't add unknown county to county list
    
    for j in range(stateCounties.shape[0]):
        # Get ALL county data
        countyData = stateData.loc[stateData.county == stateCounties[j]]
        
        # Get cases and deaths for the county
        countyCases = (countyData.cases).to_numpy()
        countyDeaths = (countyData.deaths).to_numpy()
        
        fullName = stateCounties[j] + ', ' + stateNames[i]
        
        # Normalize cases/deaths per 100k and obtain severity index
        if (usCountyPopData.loc[usCountyPopData.FullName == fullName].size != 0):
            countyPop = (usCountyPopData.loc[usCountyPopData.FullName == fullName].Population).to_numpy()[0]
            countyCasesPer100k = countyCases[-1] / (countyPop / 100000)
            countyDeathsPer100k = countyDeaths[-1] / (countyPop / 100000)
            
            newCasesData = np.diff(countyCases)
            newCasesPer100k = (newCasesData / (countyPop / 100000)).astype(int)
            newCasesPer100kFiltered = nDayAverage(newCasesPer100k, 7)
            severity_index = np.mean(newCasesPer100kFiltered[-7:])
        
        county_Latest_DT_Cases = allDoublingTimes(countyCases)[-1]
        county_Latest_DT_Deaths = allDoublingTimes(countyDeaths)[-1]
        
        whereToAddData = choroplethData.loc[choroplethData.FullName == fullName]
        
        if (whereToAddData.size != 0):                        # Only write data if choroplethData has an entry for that county
            whereToAddDataIdx = whereToAddData.index[0]

            choroplethData.at[whereToAddDataIdx, 'Population'] = countyPop.astype(int)
            
            choroplethData.at[whereToAddDataIdx, 'DT_Cases'] = county_Latest_DT_Cases
            choroplethData.at[whereToAddDataIdx, 'DT_Deaths'] = county_Latest_DT_Deaths
            
            choroplethData.at[whereToAddDataIdx, 'CasesPer100k'] = np.round(countyCasesPer100k, 3)
            choroplethData.at[whereToAddDataIdx, 'DeathsPer100k'] = np.round(countyDeathsPer100k, 3)
            
            choroplethData.at[whereToAddDataIdx, 'SeverityIndex'] = severity_index

# Normalize severity index data before plotting
severityIdxs = choroplethData.SeverityIndex.to_numpy()
min_max_scaler = MinMaxScaler()
severityIdxsNorm = min_max_scaler.fit_transform(severityIdxs.reshape(-1,1))
choroplethData.at[0:len(blankTableSpace)+1, 'SeverityIndex'] = np.round(severityIdxsNorm, 2)
            
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

maxDT_Cases = np.amax(choroplethData.DT_Cases.to_numpy())

# Get colors for DT colors
DT_Cases_Int = (choroplethData.DT_Cases).astype(int)

colors = np.full((choroplethData.DT_Cases.shape[0],), 'blue')
for i in range (0,21):
    colors = np.where(DT_Cases_Int == i, 'red', colors)
for i in range (21,31):
    colors = np.where(DT_Cases_Int == i, 'yellow', colors)
colors = np.where(DT_Cases_Int >= 31, 'green', colors) 
    
# Create Choropleth Plot
# fig = px.choropleth(choroplethData, geojson=counties, locations='FIPS', color='DT_Cases',
#                     color_continuous_scale="RdYlGn",
#                     scope="usa",
#                     hover_name='FullName', hover_data=['Population','Cases','CasesPer100k','Deaths','DeathsPer100k','DT_Cases','DT_Deaths','SeverityIndex'],
#                     )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()

c = choroplethData.astype(str)

text = '<b>' + c['FullName'] + '</b>' + '<br>' +     'Population: ' + c['Population'] + '<br>' +     'Cases: ' + c['Cases'] + '<br>' +     'CasesPer100k: ' + c['CasesPer100k'] + '<br>' +     'DT_Cases: ' + c['DT_Cases'] + '<br>' +     'Deaths: ' + c['Deaths'] + '<br>' +     'DeathsPer100k: ' + c['DeathsPer100k'] + '<br>' +     'DT_Deaths: ' + c['DT_Deaths'] + '<br>' +     'Severity Index: ' + c['SeverityIndex']

choroplethCountyMetrics = go.Choropleth(
    locations=choroplethData.FIPS,
    geojson=counties,
    z = np.round(choroplethData.DT_Cases, 3),
    zmin = 0,
    zmax = maxDT_Cases,
    colorscale=[[0,'red'], [20/maxDT_Cases,'red'], [21/maxDT_Cases,'yellow'], [30/maxDT_Cases,'yellow'], [31/maxDT_Cases,'green'], [1,'green']],
    autocolorscale=False,
    text=text, 
    marker_line_color='black',
    colorbar_title="Doubling Time for Cases",
)
countyMetricsPlot = go.Figure(data=choroplethCountyMetrics)

countyMetricsPlot.update_layout(
    title_text='US County Metrics',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'),
)

countyMetricsPlot.show()


# <font size="5"><b>New Cases per 100k and Positivity Rates for Each State</b></font>

# In[ ]:


def stabilityIndices(filteredNewDataPer100k):
    length = filteredNewDataPer100k.shape[0]
    
    stabilityIndices = np.zeros(length,)
    for i in range(length - 1,1,-1):
        if (filteredNewDataPer100k[i - 1] > 0):
            stabilityIndices[i] = filteredNewDataPer100k[i] / filteredNewDataPer100k[i - 1]
    
    return stabilityIndices


# In[ ]:


# Obtain case/death data for ALL days for ALL 50 states
allStateNames = (usStateData.state).to_numpy()
allStateNames = np.unique(allStateNames)

allDates = (usStateData.date).to_numpy()

stateCaseData = (usStateData.cases).to_numpy()
stateDeathData = (usStateData.deaths).to_numpy()

numStates = allStateNames.shape[0]

# Get population data
populationData = pd.read_csv('../input/population-data/Population_v2.csv')
populationData = populationData.to_numpy()

# Get next14Days as string
numPreds = 14
next14Days_str = []
date = datetime.datetime(2020,8,19)              # Latest date available in data
for i in range(numPreds): 
    next14Days_str.append(date.strftime('%Y-%m-%d'))
    date += datetime.timedelta(days=1)

# Create starter table for predicted CT Travel Ban states
blankTableSpace = np.zeros(numStates,).tolist()
d = {'State': allStateNames, 'NewCasesPer100k': blankTableSpace, 'StdDev_Cases': blankTableSpace, 'PositivityRate': blankTableSpace}
CT_TravelBan_8_25 = pd.DataFrame(data=d)
CT_TravelBan_9_01 = pd.DataFrame(data=d)
    
severity_index = np.zeros(numStates,)     # For cases
plots = []
for i in range(numStates):   
    if (populationData[np.where(populationData == allStateNames[i])[0]].shape[0] != 0):
        
        # Check if population data is available
        stateRow = populationData[np.where(populationData == allStateNames[i])[0]]
        statePop = stateRow[0,1]
    
        stateData = usStateData.loc[usStateData.state == allStateNames[i]]
        
        caseData = stateData.cases.to_numpy()
        dates = stateData.date
        
        newCasesData = np.diff(caseData)
        newCasesPer100k = (newCasesData / (statePop / 100000)).astype(int)
        newCasesPer100kFiltered = nDayAverage(newCasesPer100k, 7)
        severity_index[i] = np.mean(newCasesPer100kFiltered[-7:])
        
        deathData = stateDeathData[np.where(usStateData.state == allStateNames[i])]
        newDeathsData = np.diff(deathData)
        newDeathsPer100k = (newDeathsData / (statePop / 100000)).astype(int)
        newDeathsPer100kFiltered = nDayAverage(newDeathsPer100k, 7)
    
        dates = allDates[np.where(usStateData.state == allStateNames[i])]
        #datesNum = np.arange(1,dates.shape[0])
        
        # Calculate positivity rate
        dates_covidtracking = covidTrackingDaily.loc[covidTrackingDaily['state'] == fullStateNameToAbbrev(np.array([allStateNames[i]]))[0]].date
        posCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == fullStateNameToAbbrev(np.array([allStateNames[i]]))[0]].positive
        negCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == fullStateNameToAbbrev(np.array([allStateNames[i]]))[0]].negative
        pendingCasesForState = covidTrackingDaily.loc[covidTrackingDaily['state'] == fullStateNameToAbbrev(np.array([allStateNames[i]]))[0]].pending
        posRates = positivityRates(posCasesForState, negCasesForState, pendingCasesForState)

        
        # Run 14 day Kalman Predictions
        #dates = dates[:np.where(dates == '2020-06-24')[0][0] + 1]                  
        numDays = dates.shape[0] - 1
        twoState = True
        kalFilt_NewCPer100k, kalFilt_DeltaCPer100k, kalFiltPs, NewCPer100kPreds, vPreds, pPreds, sigmaNewCases, sigmaVs = kalmanFilter(newCasesPer100kFiltered, numDays, numPreds, twoState, q=0.05, r=0.1)
        newCasesPer100kPreds = go.Scatter(x=next14Days_str, y=NewCPer100kPreds, name="Predicted New Cases Per 100k (14 Days)",
                                        error_y = dict(
                                        type = 'data', # value of error bar given in data coordinates
                                        array = sigmaXs[numDays:],
                                        visible = False))
        kalFilt_PosRate, kalFilt_DeltaPosRates, kalFiltPs, PosRatePreds, vPreds, pPreds, sigmaPosRates, sigmaVs = kalmanFilter(posRates[::-1], posRates.shape[0], numPreds, twoState, q=1, r=1)
        posRatePreds = np.clip(PosRatePreds, a_min=0, a_max=100)
        posRatePredsData = go.Scatter(x=next14Days_str, y=posRatePreds, name="Predicted Positivity Rates (14 Days)")
        
        # Add new cases per 100k on 8/25 and 9/01 to CT travel ban data table
        idx8_25 = np.where(np.array(next14Days_str) == '2020-08-25')[0][0]
        idx9_01 = np.where(np.array(next14Days_str) == '2020-09-01')[0][0]
        
        newCasesPer100kPred_8_25 = NewCPer100kPreds[idx8_25]
        newCasesPer100kPred_9_01 = NewCPer100kPreds[idx9_01]
        
        predPosRate_8_25 = posRatePreds[idx8_25]
        predPosRate_9_01 = posRatePreds[idx9_01]
        
        CT_TravelBan_8_25.at[i, 'NewCasesPer100k'] = newCasesPer100kPred_8_25
        CT_TravelBan_8_25.at[i, 'StdDev_Cases'] = sigmaNewCases[np.where(sigmaNewCases == 0)[0][-1] + idx8_25]
        CT_TravelBan_8_25.at[i, 'PositivityRate'] = predPosRate_8_25
        
        CT_TravelBan_9_01.at[i, 'NewCasesPer100k'] = newCasesPer100kPred_9_01
        CT_TravelBan_9_01.at[i, 'StdDev_Cases'] = sigmaNewCases[np.where(sigmaNewCases == 0)[0][-1] + idx9_01]
        CT_TravelBan_9_01.at[i, 'PositivityRate'] = predPosRate_9_01
        
        
        # Now create the interactive plots
        newCasesPer100kData = go.Scatter(x=dates, y=newCasesPer100k, name="New Cases Per 100k")
        newDeathsPer100kData = go.Scatter(x=dates, y=newDeathsPer100k, name="New Deaths Per 100k")
        newCasesPer100kData_Filtered = go.Scatter(x=dates, y=newCasesPer100kFiltered, name="New Cases Per 100k (7 Day Average)")
        newDeathsPer100kData_Filtered = go.Scatter(x=dates, y=newDeathsPer100kFiltered, name="New Deaths Per 100k (7 Day Average)")
        stabilityIndicesCases = go.Scatter(x=dates, y=stabilityIndices(newCasesPer100kFiltered), name="Stability Indices")
        positivityRateData = go.Scatter(x=dates_covidtracking[1:], y=nDayAverage(posRates, 7), name="Positivity Rates (7 Day Average)")
        tenNewCDPer100k = go.Scatter(x=np.concatenate((dates,next14Days_str)), y=np.ones(dates.shape[0] + numPreds)*10, name="10 New Cases/Deaths Per 100k Line")
        
        per100kLayout = go.Layout(title=allStateNames[i], xaxis_title='Date', yaxis_title='Value', width=700, height=500)
        per100kPlot = go.Figure(data=[newCasesPer100kData,newDeathsPer100kData,newCasesPer100kData_Filtered,tenNewCDPer100k,newCasesPer100kPreds,positivityRateData,posRatePredsData], layout=per100kLayout)
        
        # Add plots to plot list
        plots.append(per100kPlot)
        
        
# Normalize severity index
min_max_scaler = MinMaxScaler()
severity_index_norm = min_max_scaler.fit_transform(severity_index.reshape(-1,1))

# Plot all figures
for i in range(len(plots)):
    plots[i].update_layout(title=allStateNames[i] + '      ' + 'Severity Index: ' + str(np.around(severity_index_norm[i][0], decimals=2)))
    plots[i].show()


# <font size="5"><b>CT Travel Ban List for 8/25</b></font>

# In[ ]:


pd.set_option('precision', 2)
CT_TravelBan_8_25[CT_TravelBan_8_25.NewCasesPer100k > 10]


# <font size="5"><b>CT Travel Ban List for 9/01</b></font>

# In[ ]:


pd.set_option('precision', 2)
CT_TravelBan_9_01[CT_TravelBan_9_01.NewCasesPer100k > 10]


# <font size="5"><b>US New Cases (all days)</b></font>

# In[ ]:


newCases = np.diff(usCases)
allDates = (usData.date).to_numpy()[1:]
#allDates = np.arange(allDates.shape[0])
dates_len = allDates.shape[0]

# Set up low pass filter for newCases
a = 0.3
Ts = 1
num = [1-a]
den = [1,-a]
fLP = TransferFunction(num,den,dt=Ts)

#newCasesLPF = dlsim(fLP, newCases.tolist(), np.arange(0,13))[1]

# Set up n-day average for newCases
newCases7DayAvg = nDayAverage(newCases, 7)


allNewCaseData = go.Bar(x=allDates, y=newCases, name="Actual Data")
allNewCaseData7DayAvg = go.Scatter(x=allDates, y=newCases7DayAvg, name="7 Day Average")

allNewCasesLayout = go.Layout(title='New US Cases over Time', xaxis_title='Date', yaxis_title='# of New Cases')
allNewCasesPlot = go.Figure(data=[allNewCaseData,allNewCaseData7DayAvg], layout=allNewCasesLayout)

allNewCasesPlot.show()


# <font size="5"><b>US New Cases over the last 14 days</b></font>

# In[ ]:


days_in_past = 14

days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)])
adjusted_dates = future_forcast[:-10]

dates_len = dates.shape[0]
dates_zoom = dates[dates_len - days_in_past - 1:]
adjusted_dates_zoom = adjusted_dates[dates_len - days_in_past:]

usCases_zoom = usCases[-15:]
newCases = np.diff(usCases_zoom)
dates = (usData.date).to_numpy()[-14:]

# # Set up low pass filter for newCases
# a = 0.3
# Ts = 1
# num = [1-a]
# den = [1,-a]
# fLP = TransferFunction(num,den,dt=Ts)

# newCasesLPF = dlsim(fLP, newCases.tolist(), np.arange(0,14))[1]

# Set up n-day average for newCases
newCases7DayAvg = nDayAverage(newCases, 7)

# Create interactive plot
newCaseData = go.Bar(x=dates, y=newCases, name="Actual Data")
newCaseData7DayAvg = go.Scatter(x=dates, y=newCases7DayAvg, name="7 Day Average")

newCasesLayout = go.Layout(title='New US Cases over Time (last 14 days)', xaxis_title='Date', yaxis_title='# of New Cases')
newCasesPlot = go.Figure(data=[newCaseData,newCaseData7DayAvg], layout=newCasesLayout)

newCasesPlot.show()


# <font size="5"><b>Simple linear regression on US cases (last 14 days)</b></font>

# In[ ]:


dates_len = (usData.date).to_numpy().shape[0]
adjusted_dates_zoom = np.arange(dates_len - 15, dates_len - 1)
adjusted_dates_zoom_fix = adjusted_dates_zoom.reshape(-1,1)
usCases_zoom = usCases[-14:]
dates_zoom = (usData.date).to_numpy()[-14:]

# Create linear regression object
regr = LinearRegression()

# Train linear regression model on case data
regr.fit(adjusted_dates_zoom_fix, usCases_zoom)

# Get case predictions from linear regression model
usCases_zoom_pred = regr.predict(adjusted_dates_zoom_fix)

# Get R^2 score
r_score = regr.score(adjusted_dates_zoom_fix, usCases_zoom)

# Print out equation
print('# of cases = ' + str(regr.coef_) + '*days_since_1_22_2020' + ' + ' + str(regr.intercept_))

usCasesZoomData = go.Scatter(x=dates_zoom, y=usCases_zoom, name="Actual Data")
usCasesZoomDataLinReg = go.Scatter(x=dates_zoom, y=usCases_zoom_pred, name="Linear Regression")

usCasesLinRegLayout = go.Layout(title='US Cases over Time (last 14 days) - R^2 = ' + str(np.round(r_score,3)), xaxis_title='Date', yaxis_title='# of Cases')
usCasesLinReg = go.Figure(data=[usCasesZoomData,usCasesZoomDataLinReg], layout=usCasesLinRegLayout)

usCasesLinReg.show()


# <font size="5"><b>US Case Predictions for the next 7 Days</b></font>

# In[ ]:


start = adjusted_dates_zoom[-1] + 1
end = adjusted_dates_zoom[-1] + 8
next7Days = np.arange(start, end).reshape(-1,1)

predictions = regr.predict(next7Days)

# Get next7Days as string
next7Days_str = []
date = datetime.datetime(2020,8,19)              # Latest date available in data
for i in range(7): 
    date += datetime.timedelta(days=1)
    next7Days_str.append(date.strftime('%Y-%m-%d'))
    
# Now plot
next7DaysData = go.Scatter(x=next7Days_str, y=predictions, name="Predictions")

next7DaysLayout = go.Layout(title='US Cases over Time (last 14 days) with 7 Day Prediction', xaxis_title='Date', yaxis_title='# of Cases')
next7DaysPlot = go.Figure(data=[usCasesZoomData,usCasesZoomDataLinReg,next7DaysData], layout=next7DaysLayout)

next7DaysPlot.show()


# Also print out the predictions
print('Date:' + '\t\t' + 'Cases:')
for i in range(7):
    print(str(next7Days_str[i]) + '\t' + str(int(predictions[i])))


# <font size="5"><b>Demographics Data for US</b></font>

# In[ ]:


# Obtain latest US racial data
latestDate = '2020-08-19'
racialDataUSToday = racialDataUS.loc[racialDataUS.Date == latestDate]

# Get racial data for all states (combined)
racialDataUSToday_np = racialDataUSToday.to_numpy()
allRaces = ['White','Black','LatinX','Asian','AIAN','NHPI','Multi','Other','Unknown','Hisp','NonHisp','EthUnknown']

totalRaceCases = []
for i in range(3,15):
    totalRaceCases.append(np.sum(racialDataUSToday_np[:,i].astype(float)))
allRacialDataCases = pd.DataFrame([totalRaceCases], columns=allRaces)

totalRaceDeaths = []
for i in range(16,28):
     totalRaceDeaths.append(np.sum(racialDataUSToday_np[:,i].astype(float)))    
allRacialDataDeaths = pd.DataFrame([totalRaceDeaths], columns=allRaces)
        
    
    
# Create pie charts
colorsRace ={'White':'lightcyan',
             'Black':'cyan',
             'LatinX':'royalblue',
             'Asian':'blueviolet',
             'AIAN':'darkblue',
             'NHPI':'crimson',
             'Multi':'darkolivegreen',
             'Other':'deeppink',
             'Unknown':'goldenrod',
             'Hisp':'lightcoral',
             'NonHisp':'darkred',
             'EthUnknown':'violet',
            }

# Racial data for Cases
agePlotCases = px.pie(allRacialDataCases, values=totalRaceCases, names=allRaces, color=allRaces,
             color_discrete_map=colorsRace)
agePlotCases.update_traces(title='Racial Data for US Cases')
agePlotCases.show()

# Racial data for Deaths
agePlotDeaths = px.pie(allRacialDataDeaths, values=totalRaceDeaths, names=allRaces, color=allRaces,
             color_discrete_map=colorsRace)
agePlotDeaths.update_traces(title='Racial Data for US Deaths')
agePlotDeaths.show()


# <font size="5"><b>Demographics information for South Korea (data as of 6/30/20)</b></font>

# In[ ]:


# Obtain timeAge and timeGender data
timeAgeData = pd.read_csv('../input/coronavirusdataset/TimeAge.csv')
timeGenderData = pd.read_csv('../input/coronavirusdataset/TimeGender.csv')

latestTimeAgeData = timeAgeData[np.where(timeAgeData.date == '2020-06-30')[0][0]:np.where(timeAgeData.date == '2020-06-30')[0][-1] + 1]
latestTimeGenderData = timeGenderData[-2:]


# Create pie chart for age groups
colorsAgeGroup ={'0s':'lightcyan',
         '10s':'cyan',
         '20s':'royalblue',
         '30s':'blueviolet',
         '40s':'darkblue',
         '50s':'crimson',
         '60s':'fuchsia',
         '70s':'hotpink',
         '80s':'lime'}

colorsGender = {'male':'blue',
                'female':'hotpink'}

agePlotCases = px.pie(latestTimeAgeData, values='confirmed', names='age', color='age',
             color_discrete_map=colorsAgeGroup)
agePlotCases.update_traces(title='Cases per Age Group')
agePlotCases.show()

agePlotDeaths = px.pie(latestTimeAgeData, values='deceased', names='age', color='age',
             color_discrete_map=colorsAgeGroup)
agePlotDeaths.update_traces(title='Deaths per Age Group')
agePlotDeaths.show()


# Create pie chart for gender data (cases and deaths)
genderPlotCases = px.pie(latestTimeGenderData, values='confirmed', names='sex', color='sex',
             color_discrete_map=colorsGender)
genderPlotCases.update_traces(title='Cases per Gender')
genderPlotCases.show()

genderPlotDeaths = px.pie(latestTimeGenderData, values='deceased', names='sex', color='sex',
             color_discrete_map=colorsGender)
genderPlotDeaths.update_traces(title='Deaths per Gender')
genderPlotDeaths.show()


# Contact Tracing Data (for Seoul Province): Where did the virus infect people (data from 1-22-20 to 6-25-20)

# In[ ]:


# Load in the data
patientRouteData = pd.read_csv('../input/coronavirusdataset/PatientRoute.csv')
allTypes = (patientRouteData.type).to_numpy()
types = np.unique(allTypes)
numTypes = types.shape[0]

# Find the number of patients for each type of contact
countsOfEachType = collections.Counter(allTypes)

countsOfEachType_list = []
for k,v in countsOfEachType.items():
    countsOfEachType_list.append(v)
    
# Now create pie chart
contactTypeData = go.Pie(values=countsOfEachType_list, labels=types)
contactTypeLayout = go.Layout(title='Contact Type')
contactTypePlot = go.Figure(data=contactTypeData, layout=contactTypeLayout)
contactTypePlot.show()


# <font size="7"><b>Building and Training ML Models</b></font>
# <a id='build_train_ML'></a>

# <font size="5"><b>Define functions needed for RNN/LSTM model visualization</b></font>

# In[ ]:


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
    end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    
    return np.array(data), np.array(labels)


# In[ ]:


univariate_past_history = days_in_past
univariate_future_target = 0
TRAIN_SPLIT = days_in_past

x_train_uni, y_train_uni = univariate_data(usCases_norm, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(usCases_norm, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)


# <font size="5"><b>Building a Simple LSTM model</b></font>

# In[ ]:


# # Create the model
# cases_lstm = Sequential()
# cases_lstm.name = "LSTM Model for US Case Prediction"

# cases_lstm.add(LSTM(8, input_shape=(5,1)))
# cases_lstm.add(Dense(1))

# cases_lstm.compile(optimizer='adam', loss='mae')

# # Display the layers
# cases_lstm.summary()


# <font size="5"><b>Train the simple LSTM model</b></font>
