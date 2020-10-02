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


#importing the basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# First we will load the country wise data. Then we will convert the Date column to Date type.

# In[ ]:


#Loading the coutry wise data
country_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
country_data['Last Update'] = pd.to_datetime(country_data['Last Update'])
print(country_data.head)


# Now we will aggregate the number of confirmed cases at country level to get total number of cases reported by every country irrespective of their state/province. 

# In[ ]:


country_data_agg = country_data.groupby(['Country/Region', 'Last Update'])['Confirmed'].sum().reset_index()
print(country_data_agg.head(10))


# As you can see now we have day wise number of confirmed cases for each country. We can visuallize a trend showing the increase in number of cases reported. But what I want now is to see the top 25 countries with highest number of Covid19 cases.

# In[ ]:


latest_date = country_data_agg['Last Update'].max()
top_countries = country_data_agg[(country_data_agg['Last Update']) == latest_date]
top_countries = top_countries.sort_values('Confirmed', ascending = False).head(25)
print(top_countries)


# Now we have our data ready for top 25 countries, which I want to visuallize. For the bar chart you can follow the documentation of plotly.
# https://plotly.com/python/bar-charts/

# In[ ]:


import plotly.express  as px
import plotly.io as pio
data = px.data.gapminder()
fig = px.bar(top_countries, x='Country/Region', y='Confirmed', color='Confirmed',
             labels={'Confirmed':'Confirmed cases'}, height=500)
fig.show(renderer="kaggle")


# So till now USA is facing the most from this global pandemic followed by European countries. India started at a later phase but now India is moving rapidly in the rank index. So now we will see how India's trend got a raise within March and April.
# 
# Now we will load Covid 19 dataset for India. As India started getting affected after March,2020, we are going to load the cases after 1st March,2020. 

# In[ ]:


from datetime import datetime

india_data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
new = india_data['Date'].str.split('/', n = 2, expand = True)
india_data['temp_date'] = new[1]+'-'+new[0]+'-'+new[2]
india_data['_date'] = pd.to_datetime(india_data['temp_date'])
#india_data = india_data[(india_data['_date'])> datetime.strptime('2020-03-01', '%Y-%m-%d')]
print(india_data)


# Now we wiil transform the dataset to fit with our visuallization. We will aggregate the state level data to get the aggregated country value and later we will use state data to visuallize the comparison between all Indian states.

# In[ ]:


time_series = india_data.groupby(['_date'])['Confirmed', 'Cured', 'Deaths'].sum().reset_index()
print(time_series)


# So now we have our data ready with number of confirmed cases, number of cured cases and total number of deceased cases. We will use **seaborn lineplot** to visuallize our data. We will use multi line plot to show all the three parameters.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 6))
ax = sns.lineplot(x='_date', y="Confirmed",markers=True, data=time_series)
ax = sns.lineplot(x='_date', y="Cured",markers=True, data=time_series)
ax = sns.lineplot(x='_date', y="Deaths",markers=True, data=time_series)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))


# So as we can see, India got it's peak somewhere between 20th March to 25th March. On **24th,March** India declared a country wide lockdown. Before country wide lockdown, some states like Odisha has already declared state level lockdown. 
# 
# Now we will visuallize state wise trend for number of confirmed cases.

# In[ ]:


state_time_series = india_data.groupby(['_date', 'State/UnionTerritory'])['Confirmed'].sum().reset_index()
print(state_time_series)


# We will use same **seaborn lineplot** for visuallization.

# In[ ]:


import seaborn as statesns
import matplotlib.pyplot as stateplt

stateplt.figure(figsize=(16, 10))
ax = statesns.lineplot(x='_date', y="Confirmed",markers=True, hue='State/UnionTerritory', data=state_time_series)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))


# So from the plot we can see Maharashtra is the state with highest number of Covid19  positive cases. Now lets see how much time does it takes to double the number of confirmed cases.
# 

# In[ ]:


confirmed_case = []
double_days = []
double_dates = []

doubling_arr = []
for index,row in time_series.iterrows():
    doubling_effect = {}
    if(index==0):
        confirmed_case.append(row['Confirmed'])
        #It took 30 days to double the case from 2-5
        double_days.append(0)
        double_dates.append("{0}-{1}".format(row['_date'].strftime("%d"), row['_date'].strftime("%b")))
        
        init_date = row['_date']
        init_data = row['Confirmed']
        doubling_effect['days'] = 0
        doubling_effect['Confirmed'] = row['Confirmed']
        doubling_effect['date'] = "{0}-{1}".format(row['_date'].strftime("%d"), row['_date'].strftime("%b"))
        doubling_arr.append(doubling_effect)
        print(init_data)
    if(row['Confirmed'] >= init_data*2):
        init_data = row['Confirmed']
        days = (row['_date'] - init_date)/np.timedelta64(1,'D')
        print(days)
        print(row['Confirmed'])
        
        confirmed_case.append(row['Confirmed'])
        double_days.append(days)
        double_dates.append("{0}-{1}".format(row['_date'].strftime("%d"), row['_date'].strftime("%b")))
        
        doubling_effect['days'] = days
        doubling_effect['Confirmed'] = row['Confirmed']
        doubling_effect['date'] = "{0}-{1}".format(row['_date'].strftime("%d"), row['_date'].strftime("%b"))
        doubling_arr.append(doubling_effect)
        
        init_date = row['_date']
        
print(double_dates)
print(confirmed_case)
print(double_days)
print(doubling_arr)


# Now we will visuallize the data using the most popular matplotlib. You can follow these blogs for visuallizations.
# 
# https://medium.com/@pknerd/data-visualization-in-python-line-graph-in-matplotlib-9dfd0016d180
# https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barh.html
# 
# 

# In[ ]:



plt.figure(figsize=(20, 6))
x = confirmed_case
y = double_days
plt.plot(x, y)
plt.show()


data = px.data.gapminder()
fig = px.bar(doubling_arr, x='date', y='days', color='Confirmed', height=500)
fig.show(renderer="kaggle")


# So from above plot we can analyse with time it's taking more number of days to double the count of affected people. It started with 1 case and it took 30 days to get the second case. Then gradually doubling number of days stays between 4-5 days for a long time. And now we can see it's taking 9 days to double the number of cases. We can assume social distancing and lockdown are playing their parts.
