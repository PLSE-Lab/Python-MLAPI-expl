#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests #For Sending Request to the Webpage
from bs4 import BeautifulSoup as Soup #For Web Scraping


# Here is the link from which we are going to import the data.

# In[ ]:


link = 'https://www.worldometers.info/coronavirus/#countries'


# In[ ]:


column = ['Country','Total Cases','New Cases','Total Death','New Death','Total Recovered',
          'Active Cases','Serious Critical', 'Case Per Million', 'Death Per Million','Total Tests',
          'Test Per Million']


# This will be our final data file name. We create this file by using Pandas DataFrame Feature.

# In[ ]:


Virus = pd.DataFrame(columns = column)


# Here we send the requests to the Webpage for getting all the data. If the Response of the Data is <Response [200]> , We are good to go.

# In[ ]:


url = requests.get(link)
url_html = url.text
page = Soup(url_html, "html.parser")


# And this is the code by which we take data from the server for each column. At the end we print the data.

# In[ ]:


table = page.find('tbody')
for i in table.findAll('tr'):
    td = i.findAll('td')
    Country = td[0].text
    TCases  = td[1].text
    NCases  = td[2].text
    TDeath  = td[3].text
    NDeath  = td[4].text
    TRecover = td[5].text
    ACases = td[6].text
    SCritical = td[7].text
    CasePM = td[8].text
    DeathPM = td[9].text
    Ttests = td[10].text
    TestPM = td[11].text
    Virus_data = pd.DataFrame([[Country,TCases,NCases,TDeath,NDeath,TRecover,ACases,SCritical,
                                CasePM,DeathPM,Ttests,TestPM]])
    Virus_data.columns = column
    Virus = Virus.append(Virus_data, ignore_index = True)
    
print(Virus)


# In[ ]:


Virus = Virus.set_index('Country')
print(Virus.head(10))


# And now we will export the data into the csv or excel form.

# In[ ]:


print(Virus)


# In[ ]:


Virus.to_csv('CoronaVirus27-04.csv')  

