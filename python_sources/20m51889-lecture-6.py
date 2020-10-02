#!/usr/bin/env python
# coding: utf-8

# # **Exercise 6 - FAUZIA Farah, fauzia.f.ab@m.titech.ac.jp**
# 
# I decided to look at COVID-19 cases for Turkey. 
# 
# I chose Turkey because from analyzing the data compiled by [John Hopkins University of the US](http://https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases), Turkey has the highest total confirmed cases of COVID-19 at 60 days after first reported cases (followed by Italy & Spain). Turkey also [ranked 17](https://www.worldometers.info/world-population/turkey-population/) in the countries with most population in the world, with around 84 million population.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

#loading csv as data frame
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
df.head()
#print(df.columns)
#print(np.unique(df['Country/Region'].values)) #display unique values


# # TURKEY

# In[ ]:


dfT = df[df['Country/Region']=='Turkey']
dfT = dfT.groupby('ObservationDate').sum()
dfT.tail()


# Note = This 'df1' shows a cummulative information of total cases since the initial date.
# 

# # Showing Daily Cases Data
# New coloumns will be added to indicate the daily cases by calculating difference between rows

# In[ ]:


dfT['daily_confirmed'] = dfT['Confirmed'].diff() 
dfT['daily_deaths'] = dfT['Deaths'].diff()
dfT['daily_recovered'] = dfT['Recovered'].diff()
dfT.tail()

#diff substract from the row before for specific columns (obtain daily)


# Plotting simple timeline data from new additional columns

# In[ ]:


dfT['daily_confirmed'].plot()
dfT['daily_deaths'].plot()
dfT['daily_recovered'].plot()
plt.show()


# # Make an Interactive Chart
# 

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=dfT.index,y=dfT['daily_confirmed'].values,name='Daily Confirmed') #index is observed date
daily_deaths_object = go.Scatter(x=dfT.index,y=dfT['daily_deaths'].values,name='Daily Deaths')
daily_recoveries_object = go.Scatter(x=dfT.index,y=dfT['daily_recovered'].values,name='Daily Recoveries')

layout_object = go.Layout(title='Turkey Daily cases 20M51889', xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recoveries_object],layout=layout_object)
iplot(fig)
fig.write_html('Turkey_daily_cases_20M51889.html')


# From the time-series, the COVID-19 daily cases in Turkey have been decreasing. Although some spikes still happening, the daily cases number is still smaller compared to the first 60 days after first reported cases. Recoveries trend also indicating that Turkey is currently experiencing increase in recovered cases. The death cases in Turkey generally remain low amidst high confirmed cases in the country.

# # Making Informative Table
# 
# Using color gradation to correspond different values of cases in the table. 

# In[ ]:


dfT2 = dfT[['daily_confirmed']]
dfT2 = dfT.fillna(0.)

styled_object = dfT2.style.background_gradient(cmap='nipy_spectral').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)

f = open('table_20M51889.html','w')
f.write(styled_object.render())


# By observing the color gradation pattern on the total cases columns, it becomes easier to understand the daily case number at a glance.
# 
# From this interactive table, the dates with the largest daily cases are on mid April 2020, with the peak is on 16 April 2020. The daily cases number decrease from the end of April 2020 and have plateaued at around 1,000 per day since 18 May 2020. By the latest date (June 6, 2020), the daily number has decreased almost fifth times compared to the peak on 16 April 2020.
# 
# 

# # Calculate Global Ranking
# 
# Global ranking is determined by the total confirmed cases on the latest date (June 6, 2020).

# In[ ]:


latest = df[df.ObservationDate=='06/06/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Global Ranking of Turkey: ', latest[latest['Country/Region']=='Turkey'].index.values[0]+1)

##script is modified from Ryza Rynazal's shares on Slack group##


# # Discussion
# 
# News from BBC mentioned that [COVID-19 in Turkey was the one of the fastest growing outbreaks](https://www.bbc.com/news/world-europe-52831017), which worse than China or UK. Within a month, it was reported that all 81 provinces in Turkey had been affected. While there were fears that Turkey would become another Italy (which one of the hardest hit country), until this moment, [death toll is relatively low despite a high number of cases](https://www.aa.com.tr/en/latest-on-coronavirus-outbreak/turkeys-coronavirus-fight-receives-praise/1865943). 
# 
# To visually prove this, I will compare daily confirmed and death cases in Turkey with Italy. The reason is from preliminary analysis I found that Turkey & Italy are the top 2 countries with the highest total confirmed cases of COVID-19 at 60 days after first reported cases.

# **Comparing Turkey to Italy 
# 

# In[ ]:


print('Global Ranking of Italy: ', latest[latest['Country/Region']=='Italy'].index.values[0]+1)


# In[ ]:


#Obtaining daily confirmed & death cases for Italy
dfI = df[df['Country/Region']=='Italy']
dfI = dfI.groupby('ObservationDate').sum()
dfI['daily_confirmed'] = dfI['Confirmed'].diff() 
dfI['daily_deaths'] = dfI['Deaths'].diff()
dfI = dfI.fillna(0.)
dfI.tail()


# In[ ]:


#Comparing daily confirmed cases in Turkey & Italy
daily_confirmed_objectT = go.Scatter(x=dfT.index,y=dfT['daily_confirmed'].values,name='Daily Confirmed Turkey') 
daily_confirmed_objectI = go.Scatter(x=dfI.index,y=dfI['daily_confirmed'].values,name='Daily Confirmed Italy')

layout_object = go.Layout(title='Daily Confirmed Cases of COVID-19 in Turkey & Italy - 20M51889', xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_objectI,daily_confirmed_objectT],layout=layout_object)
iplot(fig)
fig.write_html('Turkey-Italy_daily_confirmed_cases_20M51889.html')

#Comparing daily death cases in Turkey & Italy
daily_deaths_objectT = go.Scatter(x=dfT.index,y=dfT['daily_deaths'].values,name='Daily Deaths Turkey')
daily_deaths_objectI = go.Scatter(x=dfI.index,y=dfI['daily_deaths'].values,name='Daily Deaths Italy')

layout_object = go.Layout(title='Daily Deaths of COVID-19 in Turkey & Italy - 20M51889', xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_deaths_objectI,daily_deaths_objectT],layout=layout_object)
iplot(fig)
fig.write_html('Turkey-Italy_daily_death_cases_20M51889.html')


# Italy ranked 7 globally in total confirmed cases of COVID-19, and one of the earliest country which hit hard by the pandemic. From the above comparison, it is clear that while Turkey showing similar fast progressing daily cases trends as Italy, the death cases is very low in comparison. Regarding this, Turkey claimed that it has [succesfully contained the pandemic ](https://www.bloomberg.com/news/articles/2020-05-20/turkey-declares-missionaccomplished-against-coronavirus)due to investments in the country's health system, science-led approach of disease management, and free treatment. 
# 
# Britain's prominent financial magazine, The Economist, notes that Turkish government ordered lock down for the young and elderly, and asked everyone else aside from the consumer-facing business to continue work, as reported in [Anadolu Agency](https://www.bloomberg.com/news/articles/2020-05-20/turkey-declares-missionaccomplished-against-coronavirus). This strategy seemed to allow Turkey to keep its factories open during pandemic and avoid the fall of economy while also ensure the necessary medical supplies in the country. While most of working-age adults infected, the most vulnerable escaped the pandemic, and the health system contributes to [high recovery rate](https://www.bloomberg.com/news/articles/2020-05-20/turkey-declares-missionaccomplished-against-coronavirus) (approximately 75%) of those who infected.  By the 1 June 2020, Turkey has [reopened its economy](https://asia.nikkei.com/Spotlight/Coronavirus/Turkey-reopens-economy-as-daily-new-COVID-19-cases-fall) and resumed domestic flights.
# 
