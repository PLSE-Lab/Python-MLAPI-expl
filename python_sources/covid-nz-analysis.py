#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime


# In[ ]:


covidData = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
covidData.rename(columns={"Country/Region": "country"}, inplace=True)

nzCases = covidData.loc[covidData["country"] == "New Zealand"]
nzCases = nzCases.loc[nzCases["Confirmed"] != 0]
nzCases['Date'] = pd.to_datetime(nzCases["Date"], infer_datetime_format=True)

countryDensity = pd.read_csv("/kaggle/input/countriesbydensity/countriesByDensity.csv")
countryDensity.rename(columns={"name": "country"}, inplace=True)

covidData = pd.merge(covidData, countryDensity, how='left', on='country', )
covidData.dropna(inplace=True)


# In[ ]:


nzCases.plot(kind="line", x="Date", y="Confirmed", logy=True, title="Cases over Time")


# In[ ]:


import tensorflow as tf


# In[ ]:




