#!/usr/bin/env python
# coding: utf-8

# # **Population Factors that Influence COVID-19** <br>
# TeYang<br>
# Created: 3/4/2020<br>
# Last update: 3/4/2020<br>
# 
# <img src="https://www.tvw.org/wp-content/uploads/2020/03/COVID-19_image.jpg" width="1000" height="300" align="center"/>
# 
# 
# ## What's in this kernel:
# 1. [Data Loading and Cleaning](#Data_loading_structure)
# 2. [Confirmed Cases, Deaths and Recoveries Across Countries](#Frequencies)
# 3. [Factors that Influence COVID-19](#Factors)
# 
# 
# Other Datasets:
# * [COVID-19 Complete Dataset (Updated every 24hrs)](https://www.kaggle.com/imdevskp/corona-virus-report#covid_19_clean_complete.csv) by Devakumar kp
# 
# * [Cleaned Population of Countries and States](https://www.kaggle.com/dgrechka/covid19-global-forecasting-locations-population) by DmitryA. Grechka

# <a id='Data_loading_structure'></a>
# ## **1. Data Loading and Cleaning** ##

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


indicators = pd.read_csv('/kaggle/input/uncover/HDE/inform-covid-indicators.csv')
cases = pd.read_csv('/kaggle/input/uncover/johns_hopkins/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')
dailycases = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
country_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')

# indicators = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\Uncover\inform-covid-indicators.csv')
# cases = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\Uncover\johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')
# dailycases = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\covid_19_clean_complete.csv')
# country_data = pd.read_csv(r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\Data\locations_population.csv')

dailycases = dailycases.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region', 'ConfirmedCases':'Confirmed', 'Fatalities':'Deaths'}).sort_values(['Country_Region','Province_State']).reset_index().drop('index',axis=1)
country_data = country_data.rename(columns={'Province.State': 'Province_State', 'Country.Region': 'Country_Region'}).drop('Provenance',axis=1)


# In[ ]:


indicators.head()


# In[ ]:


indicators = indicators.replace('No data', np.nan)


# In[ ]:


dailycases.head()


# In[ ]:


country_data.head()


# ### Get the rate of change of confirmed cases

# In[ ]:


dailycases['Province_State'] = dailycases['Province_State'].fillna(dailycases['Country_Region']) # replace NaN States with country name
dailycases['Date'] = pd.to_datetime(dailycases['Date']) 

# sum up the states to country level
dailycases= dailycases[['Country_Region','Date','Confirmed','Deaths', 'Recovered']].groupby(['Country_Region','Date'],as_index=False).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).sort_values(by=['Country_Region','Date'])

# make all countries start with their first confirmed case
dailycases = dailycases[dailycases['Confirmed']!=0]


# In[ ]:


## calculate rate of change of confirmed cases

days=dailycases.groupby('Country_Region').Country_Region.agg('count') # get number of days since first case
maxCounts = dailycases.groupby('Country_Region').Confirmed.agg('max') # get current cumulative cases
rateofchange = ((maxCounts-1)/days).to_frame().reset_index().rename(columns={0:'confirmedROC'})


# In[ ]:


rateofchange


# In[ ]:


country_data = country_data.groupby('Country_Region').agg({'Population':'sum'}).reset_index() # sum up population of states to country level


# In[ ]:


## Create ISO3 for datasets
import pycountry

import geopandas as gpd # for reading vector-based spatial data format
shapefile = '/kaggle/input/natural-earth-maps/ne_110m_admin_0_countries.shp'
#shapefile = r'C:\Users\TeYan\OneDrive\Work\Kaggle\COVID19\110m_cultural\ne_110m_admin_0_countries.shp'

# Read shapefile using Geopandas
#gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf = gpd.read_file(shapefile)

# Drop row corresponding to 'Antarctica'
gdf = gdf.drop(gdf.index[159])
## Get the ISO 3166-1 alpha-3 Country Codes

# function for getting the iso code through fuzzy search
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_3

# manually change name of some countries
rateofchange.loc[rateofchange['Country_Region'] == 'South Korea', 'Country_Region'] = 'Korea, Republic of'
rateofchange.loc[rateofchange['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
rateofchange.loc[rateofchange['Country_Region'] == 'Burma', 'Country_Region'] = 'Myanmar'
rateofchange.loc[rateofchange['Country_Region'] == 'Congo (Kinshasa)', 'Country_Region'] = 'Congo, The Democratic Republic of the'
rateofchange.loc[rateofchange['Country_Region'] == 'Congo (Brazzaville)', 'Country_Region'] = 'Congo'
rateofchange.loc[rateofchange['Country_Region'] == 'Laos', 'Country_Region'] = "Lao People's Democratic Republic"

cases.loc[cases['country_region'] == 'Korea, South', 'country_region'] = 'Korea, Republic of'
cases.loc[cases['country_region'] == 'Taiwan*', 'country_region'] = 'Taiwan'
cases.loc[cases['country_region'] == 'Burma', 'country_region'] = 'Myanmar'
cases.loc[cases['country_region'] == 'Congo (Kinshasa)', 'country_region'] = 'Congo, The Democratic Republic of the'
cases.loc[cases['country_region'] == 'Congo (Brazzaville)', 'country_region'] = 'Congo'
cases.loc[cases['country_region'] == 'Laos', 'country_region'] = "Lao People's Democratic Republic"

country_data.loc[country_data['Country_Region'] == 'Korea, South', 'Country_Region'] = 'Korea, Republic of'
country_data.loc[country_data['Country_Region'] == 'Taiwan*', 'Country_Region'] = 'Taiwan'
country_data.loc[country_data['Country_Region'] == 'Congo (Kinshasa)', 'Country_Region'] = 'Congo, The Democratic Republic of the'
country_data.loc[country_data['Country_Region'] == 'Congo (Brazzaville)', 'Country_Region'] = 'Congo'

# create iso mapping for countries in df
iso_map = {country: do_fuzzy_search(country) for country in cases['country_region'].unique()}
# apply the mapping to df
cases['iso3'] = cases['country_region'].map(iso_map)
rateofchange['iso3'] = rateofchange['Country_Region'].map(iso_map)
rateofchange = rateofchange.drop('Country_Region',axis=1)
country_data['iso3'] = country_data['Country_Region'].map(iso_map)
country_data = country_data.drop('Country_Region',axis=1)


# In[ ]:


noiso = cases[cases['iso3'].isna()]['country_region'].unique()
noiso


# In[ ]:


# merge dataframes
df = cases.merge(rateofchange,on='iso3',how='outer').merge(indicators, on='iso3', how='outer').merge(country_data, on='iso3', how='left')


# <a id='Frequencies'></a>
# ## **2. Confirmed Cases, Deaths and Recoveries Across Countries** ##

# In[ ]:


# Standard plotly imports
#import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode, plot
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[ ]:


df['country_region'] = df['country_region'].replace(np.nan,'None')


# In[ ]:


fig = px.treemap(df.sort_values(by='confirmed', ascending=False).reset_index(drop=True), 
                 path=["country_region"], values="confirmed", height=700, width=800,
                 title='Number of Confirmed Cases',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# In[ ]:


fig = px.treemap(df.sort_values(by='deaths', ascending=False).reset_index(drop=True), 
                 path=["country_region"], values="deaths", height=700, width=800,
                 title='Number of Deaths',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# In[ ]:


fig = px.treemap(df.sort_values(by='recovered', ascending=False).reset_index(drop=True), 
                 path=["country_region"], values="recovered", height=700, width=800,
                 title='Number of Recovered',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# <a id='Factors'></a>
# ## **3. Factors that Influence COVID-19** ##

# ### INFORM Epidemic Risk Index

# In[ ]:


fig = px.choropleth(df, locations='iso3', color='inform_risk', hover_name='country', color_continuous_scale='reds', labels={'inform_risk':'Epidemic Risk'}, title='Epidemic Risk Index by Country')
fig.show()


# In[ ]:


fig = px.choropleth(df, locations='iso3', color='inform_p2p_hazard_and_exposure_dimension', hover_name='country', color_continuous_scale='reds', labels={'inform_p2p_hazard_and_exposure_dimension':'P2P Risk'}, title='P2P Risk Index by Country')

fig.show()


# The INFORM Epidemic Risk Index assesses the risk of countries to epidemic outbreak, which would exceed the national capacity to respond to the crisis. The P2P is one of the subcomponents of the mode of transmission consisting of 1) Zoonoses, (2) Vector borne, (3) Person-to-person (P2P), (4) Foodborne, and Waterborne. Since COVID-19 is mainly dominated by human to human transmission, the P2P component is used. Both inform_risk and inform_p2p_dimension range from 0-10, where 10 is the highest risk.
# 
# We can see that in general, developing countries, especially Africa, have the highest risk of an epidemic outbreak. However, the current highest outbreak is in the regions of Europe and North America. Therefore, other factors are contributing to the spread of the virus, such as urbanity and mobility of the country/states/cities. However, the confirmed cases reported can also be biased due to testing and lags. 
# 
# Evenso, given that Europe and North America are quite low on the P2P risk index but now have a staggering numbers of confirmed COVID-19 cases, it is frightening to imagine what will happen if the virus is not contained in Africa. 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# removed_nocases = df[df['confirmed'] != 0] # remove countries with 0 cases
# removed_nocases['confirmed_percent'] = removed_nocases['confirmed']/removed_nocases['Population']*100 # get percentage of confirmed cases in population

removed_nocases = df[(df['confirmedROC'] != 0) & (df['country_region'] != 'None')]
removed_nocases['confirmedROClog'] = np.log(removed_nocases['confirmedROC']).replace(-np.inf, 0)

# Set figsize here
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4.5))

# flatten axes for easy iterating
sns.regplot(x='inform_risk',y='confirmedROClog',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[0])
axes[0].set_ylabel(ylabel='Confirmed Cases Rate of Change (log)')
axes[0].set_xlabel(xlabel='Epidemic Risk Index')
sns.regplot(x='inform_p2p_hazard_and_exposure_dimension',y='confirmedROClog',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[1])
axes[1].set_ylabel(ylabel='Confirmed Cases Rate of Change (log)')
axes[1].set_xlabel(xlabel='P2P Risk Index')

fig.suptitle('Factors Influencing Rate of Change of Confirmed Cases', size=20)


# The plot result is not expected. Higher risk index is associated with lower confirmed cases. Again, the confirmed cases reported by countries might be influenced by testing and lags. Countries that are more prepared/at lower risk, are doing more testing and so more cases are discovered compared to countries that are not prepared and are late in implementing testing. As more data becomes available from all countries, this pattern might reverse.

# ### Population Density and Urban Population

# In[ ]:


temp = removed_nocases[removed_nocases['country_region'] != 'Singapore'] # removed singapore as density is outlier
temp['population_density'] = np.log(temp['population_density']).replace(-np.inf, 0)
temp['population_living_in_urban_areas'] = pd.to_numeric(temp['population_living_in_urban_areas'])
temp['population_living_in_urban_areas'] = np.log(temp['population_living_in_urban_areas']).replace(-np.inf, 0)

# Set figsize here
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4.5))

# flatten axes for easy iterating
sns.regplot(x='population_density',y='confirmedROClog',data=temp,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[0])
axes[0].set_ylabel(ylabel='Confirmed Cases Rate of Change (log)')
axes[0].set_xlabel(xlabel='Population Density (log)')
sns.regplot(x='population_living_in_urban_areas',y='confirmedROClog',data=temp,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[1])
axes[1].set_ylabel(ylabel='Confirmed Cases Rate of Change (log)')
axes[1].set_xlabel(xlabel='Population Living in Urban Areas (log)')

fig.suptitle('Factors Influencing Rate of Change of Confirmed Cases', size=20)


# Here we see that how dense a population is and how many are living in urban areas influence the rate of change of confirmed cases.

# ### Factors affecting Mortaliy Rate and Recovery Rate 

# In[ ]:


# compute mortality and recovery rate
removed_nocases['mortalityrate'] = removed_nocases.deaths/removed_nocases.confirmed * 100
removed_nocases['recoveryrate'] = removed_nocases.recovered/removed_nocases.confirmed * 100

removed_nocases.physicians_density = pd.to_numeric(removed_nocases.physicians_density)
removed_nocases.inform_health_conditions.iloc[328] = np.nan
removed_nocases.inform_health_conditions = pd.to_numeric(removed_nocases.inform_health_conditions)
removed_nocases.current_health_expenditure_per_capita = pd.to_numeric(removed_nocases.current_health_expenditure_per_capita)
removed_nocases.inform_access_to_healthcare.iloc[328] = np.nan
removed_nocases.inform_access_to_healthcare = pd.to_numeric(removed_nocases.inform_access_to_healthcare)


# In[ ]:


# Set figsize here
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10,9))

# flatten axes for easy iterating
sns.regplot(x='inform_epidemic_lack_of_coping_capacity',y='mortalityrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[0][0])
axes[0][0].set_ylabel(ylabel='Mortality Rate')
axes[0][0].set_xlabel(xlabel='Epidemic Lack of Coping Capacity Index')
sns.regplot(x='physicians_density',y='mortalityrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[0][1])
axes[0][1].set_ylabel(ylabel='Mortality Rate')
axes[0][1].set_xlabel(xlabel='Physicians Density')
sns.regplot(x='inform_health_conditions',y='mortalityrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[1][0])
axes[1][0].set_ylabel(ylabel='Mortality Rate')
axes[1][0].set_xlabel(xlabel='Health Conditions Index')
sns.regplot(x='people_using_at_least_basic_sanitation_services',y='mortalityrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[1][1])
axes[1][1].set_ylabel(ylabel='Mortality Rate')
axes[1][1].set_xlabel(xlabel='Percentage of People Using at Least \n Basic Sanitation Services')

fig.suptitle('Factors Influencing Mortality Rate', size=20)


# Here, higher physician density seems to be associated with lower mortality rate. The other factors are less or not associated.

# In[ ]:


# Set figsize here
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10,9))

# flatten axes for easy iterating
sns.regplot(x='inform_epidemic_lack_of_coping_capacity',y='recoveryrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[0][0])
axes[0][0].set_ylabel(ylabel='Recovery Rate')
axes[0][0].set_xlabel(xlabel='Epidemic Lack of Coping Capacity Index')
sns.regplot(x='physicians_density',y='recoveryrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[0][1])
axes[0][1].set_ylabel(ylabel='Recovery Rate')
axes[0][1].set_xlabel(xlabel='Physicians Density')
sns.regplot(x='inform_access_to_healthcare',y='recoveryrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[1][0])
axes[1][0].set_ylabel(ylabel='Recovery Rate')
axes[1][0].set_xlabel(xlabel='Acess to Healthcare Index')
sns.regplot(x='current_health_expenditure_per_capita',y='recoveryrate',data=removed_nocases,scatter_kws={'s':25},fit_reg=True, line_kws={"color": "black"}, ax=axes[1][1])
axes[1][1].set_ylabel(ylabel='Recovery Rate')
axes[1][1].set_xlabel(xlabel='Health Expenditure per Capita')

fig.suptitle('Factors Influencing Recovery Rate', size=20)


# Recovery rate does not seem to be associated with any of the variables here.

# In[ ]:




