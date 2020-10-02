#!/usr/bin/env python
# coding: utf-8

# # King Khalid's Airport Aviation Activity

# Air transport is a vital contributer to the economy as it provides a global transportation network operating as an optimal logistical method for business and tourism. Airport aviation activity generates millions of jobs worldwide both directly and indirectly. Socially, air transport improves the quality of life in terms of leasure, cultural experiences, trade, tourism, and humanitarian aid relief as well as transporting medical supplies and organs to save lives.

# King Khalid International Airport in Riyadh, Saudi Arabia is situated on an area of 375 square kilometers, it has the second largest land area allocation for an airport worldwide. Opened in 1983, this airport consists if 5 passenger terminals, 11,600 car parking capacity, an additional royal terminal, a central control tower, and two parallel runways. King Khalid's airport welcomes a total of 51 airlines, 210,000 flights, and 25.3 million passangers annually.

# Many services in the airport depends on the arrival and departure of passengers, from resturants, gift shops, rental cars, travel agencies, and trolly services. Thus, knowing historical data to predict future demand is one essential way to enhance the performance of these service dependant businesses. Because it costs a lot of money to get this data. For example, flightaware sells 1-month data for 617 US Dollars. I decided to write a code allowing to webscrape one-week data that is worth 154 US Dollars. If this code was to run weekly, anyone can collect enough data to make time series analysis and predictions on airport aviation activity. 

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# # Arrivals Data (per week since scraped):

# In[ ]:


res = []
page = 0
r = requests.get("https://uk.flightaware.com/live/airport/OERK/arrivals?;offset={};order=actualarrivaltime;sort=DESC".format(page))
html = r.text
soup = BeautifulSoup(html, "html.parser")
table = soup.find('table', attrs={"class": "prettyTable"})
table_rows = table.find_all('tr')


while len(table_rows) > 3:
    for tr in table_rows:
        td = tr.find_all('td')
        row = [tr.text.strip() for tr in td if tr.text.strip()]
        if row:
            res.append(row)
    page += 20

df_arrivals = pd.DataFrame(res, columns=['Ident', 'Type', 'Origin', 'Departure', 'Arrival'])
df_arrivals


# ### Save df_arrivals as CSV file.

# # Departures Data (per week since scraped):

# In[ ]:


r = requests.get("https://uk.flightaware.com/live/airport/OERK/departures?;offset={};order=actualdeparturetime;sort=DESC".format(page))
html = r.text
soup = BeautifulSoup(html, "html.parser")
table = soup.find('table', attrs={"class": "prettyTable"})
table_rows = table.find_all('tr')

res = []
page = 0
while len(table_rows) > 3:   
    for tr in table_rows:
        td = tr.find_all('td')
        row = [tr.text.strip() for tr in td if tr.text.strip()]
        if row:
            res.append(row)
    page += 20

df_departures = pd.DataFrame(res, columns=['Ident', 'Type', 'Destination','Departure', 'Estimated Arrival Time', 'Arrival'])
df_departures


# ### Save df_departures as CSV.

# In[ ]:




