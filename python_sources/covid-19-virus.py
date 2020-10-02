#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
covid19 = pd.read_csv("../input/covid19-coronavirus/2019_nCoV_data.csv")


# In[ ]:


covid19.head()


# In[ ]:


covid19["Confirmed"]


# In[ ]:


covid19["Deaths"]


# In[ ]:


covid19["Country"]


# In[ ]:


countriesList = []
for country in covid19["Country"]:
    for data in country.split(","):
        if data not in countriesList:
            countriesList.append(data)
print(countriesList)


# In[ ]:


countryTotal = {}    #Total death numbers
countryCount = {}    #Data entries that has happened
countryCases = {}    #Confirmed cases
for country in countriesList:
    countryTotal[country] = 0
    countryCount[country] = 0
    countryCases[country] = 0
print(countryTotal)
print(countryCount)
print(countryCases)


# In[ ]:


for index, countries in enumerate(covid19["Country"]):
    
    for country in countries.split(","):
        countryTotal[country] += covid19["Deaths"][index] 
        countryCases[country] += covid19["Confirmed"][index]
        countryCount[country] += 1
print(countryTotal)
print("---------------------------------------------------------------------------")
print(countryCases)
print("---------------------------------------------------------------------------")
print(countryCount)


# In[ ]:


countryAverage = {}   #deaths per observed cases

for country in countryTotal.keys():
    countryAverage[country]= countryTotal[country] / countryCases[country]
print(countryAverage)


# In[ ]:


resultCountry = "temporary"
deathCountry = "temporary2"
resultDeaths = 0
resultsdeathCountry = 0
for country in countryAverage.keys():
    if countryAverage[country] > resultDeaths:
        resultCountry = country
        resultDeaths = countryAverage[country]
    if countryTotal[country] > resultsdeathCountry:
        deathCountry = country
        resultsdeathCountry = countryTotal[country]
print( resultCountry  +  " is the country that has the least cured patients infected by the deadly virus aka COVID-19 per observed cases with a ratio of %" +   str(round(resultDeaths,2)*100) + "\n" + deathCountry + " has the most deadly cases with a number of " + str(resultsdeathCountry) )

