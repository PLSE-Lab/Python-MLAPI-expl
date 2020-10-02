#!/usr/bin/env python
# coding: utf-8

# **COVID-19 Data Downloader Function**
# 
# 
# Please bear in mind that if you are gonna use this piece of function/code you should have managed to validate your kaggle account to have intenet access, unless you have managed to run it on your personal Desktop with an internet connection.
# In case you are not familiar, two related forums of kaggle internet connection are as follows.
# 
# https://www.kaggle.com/product-feedback/63544
# 
# https://www.kaggle.com/product-feedback/113350#652385
# 
# 
# To use the function, you should simply enter an optional argument of the data subset you are looking for. This subset can be any of the 5 following: 'CONFIRMED', 'DEATH', 'RECOVERED', 'CONFIRMED_US', and 'DEATH_US'. Where first three subsets are the coronavirus related cases globally and the last two ones are the US cases, respectively. If none or a wrong value is chosen as the subset name, the globally confirmed cases is returned. The function will return a pandas DataFrame as an output. The returned dataset is the [COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data) dataset/subset from John Hopkins university
# 
# Hope you enjoy!

# In[ ]:


import requests
import pandas as pd
import io

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_covid19_confirmed_global.csv'
DEATH = 'time_series_covid19_deaths_global.csv'
RECOVERED = 'time_series_covid19_recovered_global.csv'
CONFIRMED_US = 'time_series_covid19_confirmed_US.csv'
DEATH_US = 'time_series_covid19_deaths_US.csv'

def get_covid_data(subset = 'CONFIRMED'):
    """This function returns the latest available data subset of COVID-19. 
        The returned value is in pandas DataFrame type.
    Args:
        subset (:obj:`str`, optional): Any value out of 5 subsets of 'CONFIRMED',
        'DEATH', 'RECOVERED', 'CONFIRMED_US' and 'DEATH_US' is a valid input. If the value
        is not chosen or typed wrongly, CONFIRMED subet will be returned.
    """    
    switcher =  {
                'CONFIRMED'     : BASE_URL + CONFIRMED,
                'DEATH'         : BASE_URL + DEATH,
                'RECOVERED'     : BASE_URL + RECOVERED,
                'CONFIRMED_US'  : BASE_URL + CONFIRMED_US,
                'DEATH_US'      : BASE_URL + DEATH_US,
                }

    CSV_URL = switcher.get(subset, BASE_URL + CONFIRMED)

    with requests.Session() as s:
        download        = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        data            = pd.read_csv(io.StringIO(decoded_content))

    return data


# By running the first block, the function is ready to be used at any part of your code.
# 
# As a sample, here the globally death cases of coronavirus are presented.

# In[ ]:


death = get_covid_data(subset = 'DEATH')

print(death.head(10))


# Sample Plot of the Data

# In[ ]:


from matplotlib import pyplot

countries=['Brazil', 'Italy', 'Germany']

for r in death['Country/Region']:
    if r in countries:
        pyplot.plot(range(len(death.columns)-4), death.loc[death['Country/Region']==r].iloc[0,4:], label = r) 
pyplot.legend()
pyplot.title('Total Number of COVID-19 Death Cases')
pyplot.xlabel('Day')
pyplot.ylabel('Number of Cases')


# In[ ]:




