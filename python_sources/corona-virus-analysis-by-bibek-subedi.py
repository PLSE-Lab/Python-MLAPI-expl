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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from datetime import datetime


# In[ ]:


import pandas as pd
COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")


# In[ ]:


covid_19_data.head()


# Counting the number on cases on the basis of SNo and observation date

# In[ ]:


date = covid_19_data.groupby('ObservationDate').count()
date.head()


# In[ ]:


observation_date = date['SNo']


# <b>Plottting data in line graph on the basis of observation date and the number of patient count.</b>
# 

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (18,8)
plt.plot(observation_date, label = "count rise")
plt.legend(loc = "upper left")
# observation_date.plot()


# <b>Using scatter plot to see data points.</b>

# In[ ]:


plt.scatter(observation_date.index,date['SNo'])
plt.plot(observation_date, label = "count_rise")
plt.legend(loc = "upper left")


# <b>Showing data points in line graph.</b>

# In[ ]:


observation_date.plot(marker='.')
plt.grid(which='both')
plt.plot(observation_date, label = "count_rise")
plt.legend(loc = "upper left")


# <b> Showing data points with more than 5% change</b>

# In[ ]:



data_set = observation_date.index

fig, ax = plt.subplots()
ax.plot(observation_date, marker= '.', linewidth=0.5)
ax.plot(observation_date.loc[data_set], 'k*', label = 'count_rise>5')
ax.legend(loc='upper left')


# <b> Above visualization shows that the rate of change in data is very high. There is high flucation in data. The threat of corona is real.</b>

# <b> Now we will try to group data on the basis of country or Region</b>

# In[ ]:


state = covid_19_data.groupby("Country/Region").count()
state.head()


# In[ ]:


country= state['SNo']


# In[ ]:


country.plot.bar()


# <b> As we can see that there are many country affected by corona virus and it's very difficult to perform analysis on them.
#     so we will plot only those country which have more than 100 cases</b>

# In[ ]:


greater_than_100_cases = country[state["SNo"]>100]
greater_than_100_cases.head()


# <b> Above dataframe shows country with more than 100 cases which we will plot in graph below</b>

# In[ ]:


greater_than_100_cases.plot.bar()


# <b> Here I load the data again and count it directly by Observation date </b>
# 

# In[ ]:


virus = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", index_col='ObservationDate', parse_dates =["ObservationDate"])
virus.head()


# <b> Now we change the Country/Region to Country</b>

# This is how you change a table head

# In[ ]:


virus.rename(columns = {'Country/Region': 'Country'}, inplace = True)
virus['Country'] = virus.Country.str.upper()


# In[ ]:


virus.head()


# <b>Now we see all the country with unique name</b>

# In[ ]:


virus.Country.unique()


# <b> Here we sort the main four countries with more than 100 cases.</b>

# In[ ]:


china_data = virus.query('Country == "MAINLAND CHINA"')
aus_data = virus.query('Country == "AUSTRALIA"')
canada_data = virus.query('Country == "CANADA"')
us_data = virus.query('Country == "US"')


# </b> The line graph below shows the count of infected patients over time</b> 

# In[ ]:


china_data.SNo.plot()
aus_data.SNo.plot()
canada_data.SNo.plot()
us_data.SNo.plot()
plt.legend(['China','Australia', 'Canada','USA'])
plt.show()


# In[ ]:




