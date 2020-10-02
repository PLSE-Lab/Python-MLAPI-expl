#!/usr/bin/env python
# coding: utf-8

# ### Abstract:
# 
# By the end of 2019, some cases of pneumonia of unknown etiology was detected in Wuhan city. Just a week after the new year, the World Health Organization reported that it was about a novel Coronavirus (2019-CoV). However, the virus takes an even larger area of spread that goes beyond China's borders due to air travels. Just by the end of January 2020, some doctors identified coronavirus transmission jut by contacting an asymptomatic infected patient in <a id='https://www.nejm.org/doi/10.1056/NEJMc2001468'>Germany</a>. However, by the end of February, COVID-19 takes an exponential speed that covers about 170 countries with more than 250.000 cases and about 11.000 reported deaths. It is important to indicate that Europe has became the new epicentre of the virus.
# 
# In this kernel, we will investigate every relevant information about 2019-CoV in order to come out with a good understanding of its transmission speed.
# 
# > Note that under lack of any effective treatment, isolation is considered the only way to at least outbreaks COVID-19 spread.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# Let's deal with missing data.

# In[ ]:


#data.drop(['Longitude', 'Latitude'], axis=1, inplace=True)
data.dropna(subset=['Confirmed', 'Recovered', 'Deaths', 'Longitude', 'Latitude'], inplace=True)


# In[ ]:


data.info()


# In[ ]:


countries = data['Country/Region'].unique().tolist()


# In the following we are creating a world map to show the most infected areas.

# In[ ]:


def mercator(data, lon="Longitude", lat="Latitude"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    data["x"] = data[lon] * (k * np.pi/180.0)
    data["y"] = np.log(np.tan((90 + data[lat]) * np.pi/360.0)) * k
    return data


# In[ ]:


data = mercator(data)
data.head()


# In[ ]:


from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import WMTSTileSource


# In[ ]:


output_notebook()


# In[ ]:






url = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
p = figure(tools='pan, wheel_zoom', x_axis_type="mercator", y_axis_type="mercator")

p.add_tile(WMTSTileSource(url=url))
p.circle(x=data['x'], y=data['y'], fill_color='orange', size=5)
show(p)


# In[ ]:


len(countries)


# We are going to cluster our data based on continents.

# In[ ]:


union_europe = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria', 'Lithuania', 'Croatia', 'Luxembourg',
                'Cyprus', 'Malta', 'Czechia', 'Netherlands', 'Denmark', 'Poland', 'Estonia', 'Portugal',
                'Finland', 'Romania', 'France', 'Slovakia', 'Germany', 'Slovenia', 'Greece', 'Spain',
                'Hungary', 'Sweden', 'Ireland', 'UK']


# In[ ]:


non_EU = ['Albania', 'Belarus', 'Bosnia', 'Herzegovina', 'Kosovo', 'Macedonia', 'Moldova',
          'Norway', 'Russia', 'Serbia', 'Switzerland', 'Ukraine', 'Turkey']


# In[ ]:


data_EU = data[data['Country/Region'].isin(union_europe)]
data_non_EU = data[data['Country/Region'].isin(non_EU)]


# In[ ]:


aisa_countries = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh' 'Bhutan',
                  'Brunei', 'Cambodia', 'Mainland China', 'Cyprus', 'Georgia','Hong Kong',
                  'India' 'Indonesia',
                  'Iran', 'Iraq', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan'
                  , 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar',
                  'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines',
                  'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka','Syria',
                  'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkmenistan',
                  'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']


# In[ ]:


data_aisa = data[data['Country/Region'].isin(aisa_countries)]


# In[ ]:


africa_countries = ['Liberia', 'Tanzania', 'Eritrea','Ethiopia', 'Cameroon', 'Ghana','South Africa', 'Kenya', 'Rwanda','Nigeria', 'Gabon', 'Tunisia','Senegal', 'Algeria', 'Ivory Coast','Uganda', 'Morocco', 'Zimbabwe','Egypt']


# In[ ]:


data_africa = data[data['Country/Region'].isin(africa_countries)]


# In[ ]:


america_countries = ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic',
                   'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico',
                   'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'US']


# In[ ]:


data_america = data[data['Country/Region'].isin(america_countries)]


# In[ ]:


data_australia = data[(data['Country/Region']=='New Zealand')|(data['Country/Region']=='Australia')]


# In[ ]:


data[data['Country/Region']=='Others'].shape


# ### Let's check deaths and confirmed cases in each continent.
# 
# In the following analysis we will separate union european countries and those which are non as we believe free movement in union europe countries supports the development speed of Covid-16.

# In[ ]:


total_confirmed = [data_africa['Confirmed'].max(), data_aisa['Confirmed'].max(), data_EU['Confirmed'].max(), data_non_EU['Confirmed'].max(), data_america['Confirmed'].max(), data_australia['Confirmed'].max()]
total_deaths = [data_africa['Deaths'].max(), data_aisa['Deaths'].max(), data_EU['Deaths'].max(), data_non_EU['Deaths'].max(), data_america['Deaths'].max(), data_australia['Deaths'].max()]


# In[ ]:


areas = ['Africa', 'Aisa', 'EU', 'NON-EU', 'America', 'Australia']
df_continents = pd.DataFrame({'Confirmed':total_confirmed, 'Deaths':total_deaths}, index=areas)


# In[ ]:


df_continents


# In[ ]:


sns.set()
plt.figure(figsize=(12, 6), dpi=300)
position = np.arange(len(areas))
width = 0.4
plt.bar(position - (width/2), (df_continents['Confirmed']/df_continents['Confirmed'].sum())*100, width=width, label='Confirmed')
plt.bar(position + (width/2), (df_continents['Deaths']/df_continents['Deaths'].sum())*100, width=width, label='Deaths')
plt.xticks(position, rotation=10)
plt.yticks(np.arange(0, 101, 10))
ax = plt.gca()
ax.set_xticklabels(areas)
ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']);
ax.set_yticks(np.arange(0, 100, 5), minor=True)
ax.yaxis.grid(which='major')
ax.yaxis.grid(which='minor', linestyle='--')
plt.title('Confirmed vs Deaths in different continents')
plt.legend();


# In[ ]:


plt.figure(figsize=(10, 6))

plt.plot('Date', 'Confirmed', data=data_aisa, label='Aisa')
plt.plot('Date', 'Confirmed', data=data_EU, label='EU')
plt.plot('Date', 'Confirmed', data=data_non_EU, label='Non-EU')
plt.plot('Date', 'Confirmed', data=data_america, label='America')
plt.plot('Date', 'Confirmed', data=data_australia, label='Australia')
plt.xticks(np.arange(0, 60, 2), rotation=70)
plt.legend()
plt.title('Confirmed cases in different continents')


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot('Date', 'Deaths', data=data_aisa, label='Aisa')
plt.plot('Date', 'Deaths', data=data_EU, label='EU')
plt.plot('Date', 'Deaths', data=data_non_EU, label='Non-EU')
plt.plot('Date', 'Deaths', data=data_america, label='America')
plt.plot('Date', 'Deaths', data=data_australia, label='Australia')
plt.xticks(np.arange(0, 60, 2), rotation=70)
plt.yticks(np.arange(0, 3001, 500))
plt.legend()
plt.title('Deaths in different continents')


# It seems that death rate has decreased exponentially in aisa by the second week of this month, while EU have exponential increase.

# In[ ]:




