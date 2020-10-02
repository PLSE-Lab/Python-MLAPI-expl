#!/usr/bin/env python
# coding: utf-8

# <div style='text-align:center; margin-bottom:40px'>
#     <img src='https://acclr.ccmm.ca/~/media/Images/Blog/2018-03/clanAffaires_1160x771px.jpg?h=771&la=en&w=1160'/>
# </div>
# 
# # Covid-19 - Analysis & Correlations
# 
# This Project made with Python, visualization tools, and multiple data
# resources that were gathered.
# Presenting an analysis of the Covid-19 panademic separated into 4 important subjects of analysis:
# 
# 1. World Updates
# 2. Europe
# 3. Chine
# 5. Top 10 Infected Countries Today
# 
# ### The goal: 
# 
# Explore and analyze big data resources to find insights about the following topics:
# 
# * The connection between European passenger flight traffic in February 2020 to the confirmed cases in Europe in February and March.
# * Confirmed cases in China provinces and The number of population in Each Province.
# 
# And in addition, scatter charts, pie charts, linear correlation charts and moving date charts, and more.
# 
# Let's get started.

# # Import libraries

# In[ ]:


# analyse and store data
import pandas as pd
import numpy as np
from datetime import datetime

# Vizualisaion 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
plt.style.use('dark_background') #dark mode
px.defaults.template = 'plotly_dark' #dark mode
plt.rcParams['figure.dpi'] = 100
import plotly.graph_objects as go


# hide warnings
import warnings
warnings.filterwarnings('ignore')


# # The Data 
# 
# The internal data of this project taken from:
# 1. Eurostas website [Eurostas air transport of passengers](https://ec.europa.eu/eurostat/databrowser/view/ttr00016/default/table?lang=en).
# 2. kaggle dataset [china provinces population](http://population.city/) as a reference from http://population.city/.
# 3. The data on covid-19 taken from the daily [Coronavirus disease (COVID-2019) situation reports] .(https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports) of World Health Organization. The Johns Hopkins University have transform the reports into a dataset available on their [GitHub repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series).

# **unorganized start table of confirmed cases for example.**

# In[ ]:


filename_pattern = '../input/novel-corona-virus-2019-dataset/time_series_covid_19_{}.csv'
confirmed = pd.read_csv(filename_pattern.format('confirmed'))
confirmed


# **Combining all structured tables into one, based on forign keys and load and clean the data :**

# In[ ]:


# convert the three table sources on covid_19 into one 
covid= pd.concat([pd.read_csv(filename_pattern.format(data)).                melt(id_vars=['Province/State','Country/Region','Lat','Long'],var_name = 'Date'                    ,value_name=data).set_index(['Province/State','Country/Region', 'Lat', 'Long', 'Date'])                    for data in ['confirmed','deaths','recovered']],axis = 1).reset_index()    .assign(Infected=lambda df: df['confirmed']-df['recovered']-df['deaths']).rename(columns={"confirmed": "Confirmed", "recovered": "Recovered","deaths": "Deaths" }).fillna(value = 0.00).astype({'Confirmed': 'int32','Deaths': 'int32','Recovered': 'int32','Infected': 'int32',})
covid


# # Variables
# 
# The table above represents in each row, a day data on cases in different countries or regions from 22-01-2020 (the first case of infected cases in china) until the current day below.
# 
# The variabels are:
# 
# * **Confirmed** - number of verified cases of new infected in each day.
# * **Deaths** - number of deaths from the virus in each day.
# * **Recovered** - number of recoverers from the virus in each day.
# * **Infected** - number of verified infected in each day.
# 

# In[ ]:


latest_date = covid.iloc[-1]['Date']
print("Coronavirus in the world, updated to: " + str(latest_date))
current = covid[['Date','Deaths','Recovered','Infected','Confirmed']].groupby('Date').sum().reset_index()
current[current['Date'] == latest_date].reset_index(drop = True).style.background_gradient(cmap='Dark2')


# # Part one - World Updates
# 

# **Updated case table order by confirmed cases:**

# In[ ]:


latest_update_table = covid.groupby(['Date','Country/Region']).sum().xs([latest_date])[['Confirmed','Deaths','Recovered','Infected']].sort_values(by = 'Confirmed', ascending = False).style.background_gradient(cmap='YlOrRd')
latest_update_table


# **Infected cases across the world**

# In[ ]:


world_latest_update_table = covid[covid['Date']==latest_date].sort_values(by = 'Confirmed',ascending = False).reset_index(drop = True)
fig = px.scatter_geo(world_latest_update_table                     ,lat = world_latest_update_table['Lat'],lon = world_latest_update_table['Long']                     ,size = 'Confirmed', text = world_latest_update_table['Country/Region']                             ,color ='Confirmed',projection = 'natural earth' ,size_max =50,  color_continuous_scale = 'Reds')
fig.update_layout(
        title = 'Confirmed COVID-19 Cases',
        geo_scope='world',
    )
fig.update_geos(
    visible=False, resolution=50,
    showcountries=True, countrycolor="RebeccaPurple"
)


# In[ ]:


covid_by_country = covid.groupby(['Country/Region','Date']).sum().reset_index()
fig = px.choropleth(covid_by_country, locations=covid_by_country['Country/Region'],
                    color=covid_by_country['Infected'],locationmode='country names', 
                    hover_name=covid_by_country['Country/Region'], 
                    color_continuous_scale=px.colors.sequential.speed,
                    animation_frame="Date")
fig.update_layout(

    title='Evolution of active cases In Each Country',
    template='plotly_dark'
)
fig.show()


# In[ ]:


covid['Date'] = pd.to_datetime(covid['Date'])
p = covid[['Date','Confirmed','Recovered','Infected','Deaths']].set_index('Date').groupby('Date').sum()
p.iplot(kind='scatter', filename='cufflinks/cf-simple-line',title='Rate of Cases',xTitle='Date',yTitle='Cases',theme = 'solar')


# ----------------------------------------------------------------------------------------------------------------------------------------------

# # Part two - Europe 
# 
# This chapter is divided into tow different analyzes:
# 
# * **Italy infected rate day by day**
# 
# * **Analyze flights traffic of passengers on Europe in February to prove connection to confirmed cases in European countries in February and March**  
# 
# Let's start with Italy, the most infected country all over Europe, and analyze their infected cases  and see the different rate of infected in each day till now: 
# 

# In[ ]:


italy = covid[covid['Country/Region']=='Italy'].groupby(['Country/Region','Date']).sum().reset_index().assign(Rate= lambda c: c['Infected'])
def RateUpdate(country_frame):
    for i in range(1,len(country_frame)):
         if (country_frame.iloc[i]['Infected']>0) and (country_frame.iloc[i-1]['Infected'] == 0):
                country_frame['Rate'].iloc[i] = country_frame.iloc[i]['Infected']
                continue
         country_frame['Rate'].iloc[i]=round((((country_frame.iloc[i]['Infected'] - country_frame.iloc[i-1]['Infected'])/country_frame.iloc[i-1]['Confirmed'])),4)
RateUpdate(italy)
italy[['Date','Infected','Rate']].fillna(value = 0.00).style.format('{:}').format('{:.2%}', subset='Rate').background_gradient(cmap='viridis')


# For this section of analyzis, to prove connection between confirmed cases in European countries in February and March to flight traffic in February (during this month, we still saw a lot of traffic, and no country completely closed the sky) I used several **steps**: 
# 1. Extract data from this page [Eurostas air transport of passengers](https://ec.europa.eu/eurostat/databrowser/view/ttr00016/default/table?lang=en).
# 2. Transform to a CSV structure data
# 3. Clean the file
# 4. Take from the covid table only the cases on February and March
# 5. Take into account only the flights in February 
# 
# #### Assumptions :
# 
# * Not all countries take into consideration, due to lack of information on the number of passengers in February, and because of that, We cannot be sure if this shows the best picture.
# 
# * Passenger traffic in March was not taken into account, as most countries closed the sky or reduced flights due to increased awareness of the spread of the virus, so in order to maintain equitable analysis, this data was not taken into account.
# 
# * There are other causes that can cause an increase in morbidity
# 

# In[ ]:


fp = '../input/euro-traffic/estat_ttr00016_filtered.csv'
europe_flight_data = pd.read_csv(fp)
feb_traffic_data = europe_flight_data[['Country','2020-02']]
feb_traffic_data = feb_traffic_data[feb_traffic_data['2020-02'] != ':'].reset_index(drop=True).rename(columns = {'Country':'Country/Region','2020-02':'February_passengers'})
europe_feb_list = feb_traffic_data['Country/Region'].tolist()
feb_traffic_data = feb_traffic_data.set_index('Country/Region')
feb_mar_dates = pd.date_range('2/1/20', periods=60).tolist()
temp = covid[covid['Country/Region'].isin(europe_feb_list)]
temp = temp[temp['Date'].isin(feb_mar_dates)]
europe_data_sum_on_feb_mar = temp.groupby('Country/Region').sum()
upgrade_euro = pd.merge(europe_data_sum_on_feb_mar,feb_traffic_data,how = 'inner',on='Country/Region').reset_index()
upgrade_euro['February_passengers'] = upgrade_euro['February_passengers'].apply(lambda x: int(x))
upgrade_euro[['Country/Region','Confirmed','February_passengers']].style.background_gradient(subset='Confirmed', cmap='gist_gray').background_gradient(subset='February_passengers', cmap='gist_gray')


# #### The Results:
# 
# **Conclusion : as the number of passengers increases, so does the amount of adhesion.
# There is a strong correlation between the two variables as We see below:  **
# 
# 

# In[ ]:


a = upgrade_euro[['February_passengers','Confirmed']]
sns.heatmap(a.corr(),cmap='RdYlGn_r', linewidths=0.5,annot=True,xticklabels=True, yticklabels=True)


# ## Part three - Chine
# 
# As is well known, the virus outbreak began in China.
# In this chapter on China, we will look at demographics, distribution of cases, and yes, we will explore the correlation between the Chinese population and the percentage of mortality from the virus.

# **China province population table:**

# In[ ]:


china_province_population = pd.read_csv('../input/covid19-analysis-correlations/china_provinces_population.csv')
temp = china_province_population
china_province_population.sort_values(by = 'POPULATION',ascending = False).reset_index(drop = True).style.background_gradient(cmap='Greens')


# In[ ]:


fig = px.pie(china_province_population, values=china_province_population['POPULATION'], names=china_province_population['PROVINCE NAME'],
             title='Province Population Percentage',
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    template='plotly_dark'
)
fig.show()


# **Merge between china province population table with Chine main table:**

# In[ ]:


temp['POPULATION']=temp['POPULATION'].apply(lambda t : round(t/1000,2))
temp= temp.rename(columns = {'POPULATION':'Population in thousend','PROVINCE NAME':'Province/State'})
data_with_pop = pd.merge(temp,c,how = 'inner',on='Province/State')
data_with_pop[['Province/State','Confirmed','Deaths','Recovered','Infected','Recovered_Percent','Death_Percent','Population in thousend']].style.format('{:}').format('{:.2%}', subset=['Recovered_Percent', 'Death_Percent']).background_gradient(subset='Recovered_Percent', cmap='gist_gray')     .background_gradient(subset='Death_Percent', cmap='Reds').background_gradient(subset='Population in thousend', cmap='Greens')


# #### The Results:
# 
# as  We see down below, there isn't a strong correlation between province population numbers to confirmed cases.
# 

# In[ ]:


a = data_with_pop[['Population in thousend','Confirmed']]
sns.heatmap(a.corr(),cmap='RdYlGn_r', linewidths=0.5,annot=True,xticklabels=True, yticklabels=True)


# In[ ]:


fig = px.scatter_geo(China,lat = China['Lat'],lon = China['Long'],size = 'Confirmed', text = China['Province/State']                             ,color ='Confirmed',color_continuous_scale = 'delta' ,size_max=100)
fig.update_layout(
        title = 'Speard Corona over China',
        geo_scope='asia',
    )
fig.update_geos(
    resolution=50,
    showcoastlines=True, coastlinecolor="RebeccaPurple",
    showland=True, landcolor="LightGreen",
    showocean=True, oceancolor="LightBlue",
    showlakes=True, lakecolor="Blue",
    showrivers=True, rivercolor="Blue"
)



fig.show()


# #### China infected cases compared to the rest of the world in each date, as we see, the pendulum changes over time from one side to the other.

# In[ ]:


china_infected = covid[covid['Country/Region']=='China'].groupby('Date').sum()['Infected']
world_infected = covid[covid['Country/Region']!='China'].groupby('Date').sum()['Infected']
China_VS_World = pd.DataFrame({'china_infected': china_infected,'world_infected':world_infected})
China_VS_World.style.background_gradient(cmap='Greens',subset=['world_infected','china_infected'])


# In[ ]:


China_VS_World.iplot(kind='bar', filename='cufflinks/cf-simple-line',title='Infected Rate, World Compared to Chine through Time',xTitle='Date',yTitle='Cases',theme = 'solar')


# # Part four - Top 10 Infected Countries
# 

# In[ ]:


Top_10_infected = covid[covid['Date']==latest_date].groupby('Country/Region').sum().sort_values(by = 'Infected',ascending = False).head(10).reset_index()['Country/Region']
Top_10_infected_list = Top_10_infected.tolist()
Top_10_infected_data = covid[covid['Country/Region'].isin(Top_10_infected_list)]
sum_per_day = Top_10_infected_data.groupby(['Date','Country/Region']).sum().reset_index()
fig = px.scatter(sum_per_day, 'Date', 'Infected', color='Country/Region', 
                 log_y=True, height=600)
fig.update_traces(mode='lines+markers', line=dict(width=.6))
fig.update_layout(title='Exponential Infected rate in the top ten most-affected countries')


# In[ ]:


top_10 = sum_per_day.groupby('Country/Region').sum().sort_values(by = 'Infected',ascending = False).reset_index()
fig = go.Figure(data=[go.Bar(
            x=top_10['Country/Region'], y=top_10['Infected'],
            text=top_10['Infected'],
            textposition='auto',
            marker_color='green',

        )])
fig.update_layout(
    title='Most 10 infected Countries',
    xaxis_title="Countries",
    yaxis_title="Infected Cases",
    template='plotly_dark'
)
fig.show()

