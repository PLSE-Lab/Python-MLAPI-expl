#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ##Major cities data
# In this section, I look at the data in the major cities file. I look at the five hotest major cities, the five coldest major cities and the temperature trends in Melbourne Australia, the city where I was born. 

# In[ ]:





# In[ ]:





# In[ ]:


##five hottest major cities in 2012
dfTempByMajorCity = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv',index_col='dt',parse_dates=[0])
dfTempByMajorCity[dfTempByMajorCity.index.year == 2012]['Country'=='Colombia'][['City','Country','AverageTemperature']].groupby(['City','Country']).mean().sort_values('AverageTemperature',ascending=False)


# In[ ]:


##five coldest major cities in 2012
dfTempByMajorCity[dfTempByMajorCity.index.year == 2012].groupby(['City','Country']).mean().sort_values('AverageTemperature',ascending=True).head()


# In[ ]:





# ##Global temperature data
# In this section I look at global temperatures. I use five year smoothing to look at long-run trends (the year-to-year changes are quite noisy). I also focus on the time since 1900, which is when most of the action seems to happen. I look at both land-only temperatures as well as land-and-ocean temperatures. 

# In[ ]:


##global land temperature trends since 1900
##using 5 year rolling mean to see a smoother trend 
dfGlobalTemp = pd.read_csv('../input/GlobalTemperatures.csv',index_col='dt',parse_dates=[0])
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandAverageTemperature'],window=60).plot(x=dfGlobalTemp.index)


# In[ ]:


##using 5 year rolling mean to see a smoother trend
dfGlobalTemp = pd.read_csv('../input/GlobalTemperatures.csv',index_col='dt',parse_dates=[0])
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandAndOceanAverageTemperature'],window=60).plot(x=dfGlobalTemp.index)


# ##Temperatures by country
# I look at the five hottest countries and the five coldest countries. 

# In[ ]:


dfTempByCountry = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv',index_col='dt',parse_dates=[0])
dfTempByCountry[dfTempByCountry.index.year == 2012][['Country','AverageTemperature']].groupby(['Country']).mean().sort_values('AverageTemperature',ascending=False).head()


# In[ ]:


dfTempByCountry = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv',index_col='dt',parse_dates=[0])
dfTempByCountry[dfTempByCountry.index.year == 2012][['Country','AverageTemperature']].groupby(['Country']).mean().sort_values('AverageTemperature',ascending=True).head()

