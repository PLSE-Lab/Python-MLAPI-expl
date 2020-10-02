#!/usr/bin/env python
# coding: utf-8

# # How to inform the people to get out a lockdown?
# ## Scroll down for a map
# *We need to know the actual cases so we can decide which places to avoid and in which areas we can be more relaxed. The RIVM posts the total amount of reported cases; the total of hospital admissions; the total amount of deceased people on a daily base starting at March 13, 2020. What do these numbers say about the recent situation? <br> I decided to make a map of the Netherlands with only the cases of the last week. So we know where the virus is still active. The next question is do we need absolute or relative numbers? I decided to use absolute numbers and people have to decide for themselves what to do with the numbers. Accuracy and open information is at this time very important in my view *

# In[ ]:


import pandas as pd
import geopandas as gpd
import requests
from datetime import datetime,timedelta


# In[ ]:


# Daily update at 16:00u. I'll run and save this script on a daily base


url = 'https://geodata.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.json'

try:
    data = requests.get(url, timeout=2).json()
except requests.exceptions.RequestException:
    raise Exception('Failed to connect to %s' % url) from None

rivm = pd.DataFrame.from_dict(data, orient='columns')


# **What is the information in the json published by the RIVM? Let's call head() to see the top 5 entries**

# In[ ]:


rivm.head()


# I have to get only Today's and last week's cases for my map. To filter I need the exact Date_of_report the RIVM uses in it's timestamp. (I use Yesterday instead of Today because the update is at 16:00hrs otherwise you get an error if you run the script before 16:00hrs)

# In[ ]:


#Yesterday's date 
today = datetime.today() - timedelta(days=1)
today = today.strftime('%Y-%m-%d')+ " 10:00:00"
print(today)


# In[ ]:


#Last week's date
weekAgo = datetime.today() - timedelta(days=7)
weekAgo = weekAgo.strftime('%Y-%m-%d')+ " 10:00:00"
print(weekAgo)


# Create two new dataframes. One of Today's cases and one of last Week. The RIVM posts the province numbers as well and I need only the cases of the municipalities so I discard the entries without a municipality name.

# In[ ]:


# maak een nieuwe df van vandaag en vorige week, de gegevens van de provincies mogen eruit
cases_Today = rivm.loc[(rivm.Date_of_report == today)& (rivm.Municipality_name.notnull())]
cases_weekAgo = rivm.loc[(rivm.Date_of_report == weekAgo)& (rivm.Municipality_name.notnull())]
cases_Today.head()


# Create a new dataframe with the two before combined and a new column with the difference between the cases of Today and last Week.

# In[ ]:


# maak een nieuwe df met het verschil van aantal gevallen van vorige week en deze week
cases_lastWeek = cases_Today.loc[:, ['Municipality_code', 'Municipality_name', 'Province','Total_reported']]
cases_lastWeek['Total_reported_LastWeek'] = cases_weekAgo['Total_reported'].values
cases_lastWeek['this_week_cases'] = cases_lastWeek['Total_reported'] - cases_lastWeek['Total_reported_LastWeek']
cases_lastWeek.head()


# Import the map with municipality-borders

# In[ ]:


# Haal de kaart met gemeentegrenzen op van PDOK
geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2019_gegeneraliseerd&outputFormat=json'
kaart_gemeente = gpd.read_file(geodata_url)


# Merge the map data with my covid data

# In[ ]:


# De covid-data kan nu gekoppeld worden aan de gemeentegrenzen met merge.
# Koppel covid-data aan geodata met gemeentecodes
kaart_gemeente = pd.merge(kaart_gemeente, cases_lastWeek,
                           left_on = "statcode", 
                           right_on = "Municipality_code")
kaart_gemeente.head()


# Plot the map

# In[ ]:


#Tot slot kan de thematische kaart gemaakt worden met de functie plot.


p = kaart_gemeente.plot(column='this_week_cases', 
                         figsize = (14,12), legend=True, cmap="Reds")
p.axis('off')
p.set_title('Amount of absolute reported Covid cases in The Netherlands only last Week')

