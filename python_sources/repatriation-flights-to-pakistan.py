#!/usr/bin/env python
# coding: utf-8

# # Scraping data from the Offical Covid Pakistan Goverment website.
# The scraper won't work since kaggle won't let request make a call for obivious reasons. But I have uploaded a data set that I scrapped on the 15-06-2020. You can get the latest version of the data till this website is live. You get a Sqlite3 database with 2 table.
# 
# 1. schedule.
# 2. completed.
# 
# ## schedule flights table
# This table has all the scheduled flights depended on the latest info on the website.
# 
# ## completed flights table
# This table has all the information of the flights complete till that. For me that is 15-06-2020. You can run this scrapper in the future to get the latest complete flights infromations.
# 
# ## If you want CSV
# I have a section at the end of of this notebook, which will provide you with the CSV if you don't want the power of Sqlite, or you don't know how to use the Sqlite.
# 
# ## Missing
# We can motify the script to get the latest information without purging the current information. At the moment when you run the scrapper it replaces that data in the database with the latest data.

# In[ ]:


import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
import numpy as np
from datetime import datetime
website = 'http://covid.gov.pk/intl_travellers/flight_info'
cxn = sqlite3.connect("flights.db")


# ## Checking the response from the website
# On successful return we will have soup object which will provide us with the data from the website.

# In[ ]:


page = requests.get(website)
if page.status_code == 200:
    soup = soup = BeautifulSoup(page.content, 'html.parser')
else:
    print("Error Page status:", page.status_code)


# ## Tables
# Bases on the struct of the website we get the 2 table that we need. They are named accordingly.

# In[ ]:


scheduled_flights = soup.find_all('table')[1]
complted_flights = soup.find_all('table')[2]


# ## Scheduled Flights table loop
# We skip the 1st row in the loop. This is the first complete in the data. It give use the following column names
# ```
# ['Sr #', 'From', 'Departure Airport', 'To', 'Arrival Date', 'Passengers', 'Airline']
# ```
# Personal if you ask me they are kind of ugly. That is why you see an order write of the column with the varabile name `cols`
# 
# ## Status column
# Since we have open and close status in the name of the from_place. I wanted to put it in it's own column. If you some need's or if we might want to fliter the status of the closed scheduled flights

# In[ ]:


data = []
for idx,tr in enumerate(scheduled_flights.find('tbody').find_all('tr')):
    if idx == 0:
        continue
    row = [td.text.replace('\n','').replace('\t','').replace('\r','').replace(' ','') for td in tr.find_all('td')]
    row.append('open' if 'open' in row[1] else 'close' if 'close' in row[1] else np.nan)
    data.append(row)

cols = ['sr_no', 'from_place', 'departure_airport', 'to_place', 'arrival_date', 'passengers', 'airline', 'status']
schedule = pd.DataFrame(data, columns=cols)
# date format setting for the database
schedule['arrival_date'] = pd.to_datetime(schedule['arrival_date']).dt.date
schedule.to_sql('schedule', cxn, if_exists='replace', index=False)


# ## Completed Flights
# We follow the same logic for completed flights table. We skip the same 1st row of the table as we did with the scheduled flights. But we have different number of columns for this table. Make sure you look at them.

# In[ ]:


data = []
for idx,tr in enumerate(complted_flights.find('tbody').find_all('tr')):
    if idx == 0:
        continue
    row = [td.text.replace('\n','').replace('\t','').replace('\r','').replace(' ','') for td in tr.find_all('td')]
    data.append(row)

cols = ['sr_no', 'from_place', 'to_place', '_date', 'passengers', 'airline',]
completed = pd.DataFrame(data, columns=cols)
# date format setting for the database
completed['_date'] = pd.to_datetime(completed['_date']).dt.date
completed.to_sql('completed', cxn, if_exists='replace', index=False)


# ## Getting the Province, cities and population info from Wikipedia for Pakistan
# I got this information because I want to see the flights based on provinces. This information is not avilabile in the orignal table.

# In[ ]:


pak_loc = 'https://en.wikipedia.org/wiki/List_of_cities_in_Pakistan'
page = requests.get(pak_loc)
if page.status_code == 200:
    soup = soup = BeautifulSoup(page.content, 'html.parser')
else:
    print("Error Page status:", page.status_code)


# ## Getting Provinces and cities of pakistan

# In[ ]:


data = []
for x in soup.find_all('table', {'class': 'wikitable'}):
    provice = ''
    _type = ''
    if 'Balochistan' in x.find('th').text:
        provice = 'Balochistan'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Khyber' in x.find('th').text:
        provice = 'Khyber'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Punjab' in x.find('th').text:
        provice = 'Punjab'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Sindh' in x.find('th').text:
        provice = 'Sindh'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Kashmir' in x.find('th').text:
        provice = 'Kashmir'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Capital' in x.find('th').text:
        provice = 'Punjab'
        _type = 'municipalities'
    elif 'Gilgit' in x.find('th').text:
        provice = 'Gilgit'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    
    population = x.find('tbody').find_all('tr')[3]
    names = x.find('tbody').find_all('tr')[2]
    for name, pop in zip(names.find_all('td'), population.find_all('td')):
        data.append({
            'provice': provice,
            '_type': _type,
            'city': name.text[:-6],
            'population': pop.text.replace('\n','').replace(',','')
        })

pak = pd.DataFrame(data)
pak.to_sql('pakistan', cxn, if_exists='replace', index=False)


# ## Get CSVs

# In[ ]:


def sql_fetch(con):

    cursorObj = con.cursor()

    cursorObj.execute('SELECT name from sqlite_master where type= "table"')

    return cursorObj.fetchall()

for x in sql_fetch(cxn):
    filename = f"{x[0]}.csv"
    df = pd.read_sql(f"select * from {x[0]}", cxn)
    df.to_csv(filename, index=False)
    del df
    print(filename, 'created')

