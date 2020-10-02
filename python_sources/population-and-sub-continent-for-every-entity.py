#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd

from tqdm.notebook import tqdm


# # Defining API Calls

# In[ ]:


def get_restcountries(countries):
    """Retrieve all available fields from restcountries API
    https://github.com/apilayer/restcountries#response-example"""
    
    api = 'https://restcountries.eu/rest/v2'
    rdfs = []
    for country in tqdm(countries):      
        r = requests.get(f'{api}/name/{country}?fullText=true').json()
        if len(r) != 1:
            r = requests.get(f'{api}/name/{country}?fullText=false').json()
            if len(r) != 1:
                try:
                    alpha3 = {
                        'Channel Islands': None, #['GGY', 'JEY'],
                        'Congo (Brazzaville)': 'COG',
                        'Congo (Kinshasa)': 'COD',
                        'Czechia': 'CZE',
                        'Diamond Princess': None,
                        'Iran': 'IRN',
                        'Korea, South': 'PRK',
                        'North Macedonia': 'MKD',
                        'St Martin': 'MAF',
                        'Taiwan*': 'TWN',
                        'Virgin Islands': 'VIR',
                    }
                    r = requests.get(f'{api}/alpha/{alpha3[country]}')
                    r = [r.json()] if r.status_code == 200 else []
                except:
                    r = []
        rdf = pd.DataFrame(r)
        rdf['country'] = country
        rdfs.append(rdf)
    return pd.concat(rdfs, sort=False).set_index('country')


# In[ ]:


def get_datausa():
    """Retrieve population on state level from datausa.io
    https://datausa.io/about/api/"""
    
    datausa = pd.DataFrame(requests.get('https://datausa.io/api/data?drilldowns=State&measures=Population&year=latest', headers={'User-Agent': ''}).json()['data'])
    datausa = datausa[['State', 'Population']]
    datausa.columns = ['state', 'population']
    datausa['region'] = 'Americas'
    datausa['subregion'] = 'Northern America'
    return datausa.set_index('state')


# # Manual Sourcing from Wikipedia

# In[ ]:


# https://en.wikipedia.org/wiki/List_of_Canadian_provinces_and_territories_by_population
wiki_canada = {
    'Alberta': 4413146,
    'British Columbia': 5110917,
    'Manitoba': 1377517,
    'New Brunswick': 779993,
    'Newfoundland and Labrador': 521365,
    'Nova Scotia': 977457,
    'Ontario': 14711827,
    'Prince Edward Island': 158158,
    'Quebec': 8537674,
    'Saskatchewan': 1181666,
}
canada = pd.DataFrame({'population': wiki_canada, 'region': 'Americas', 'subregion': 'Northern America'})


# In[ ]:


# https://en.wikipedia.org/wiki/States_and_territories_of_Australia
wiki_australia = {
    'Australian Capital Territory': 426709,
    'New South Wales': 8089526,
    'Northern Territory': 245869,
    'Queensland': 5095100,
    'South Australia': 1751693,
    'Tasmania': 534281,
    'Victoria': 6594804,
    'Western Australia': 2621680,
}
australia = pd.DataFrame({'population': wiki_australia, 'region': 'Oceania', 'subregion': 'Australia and New Zealand'})


# In[ ]:


# https://en.wikipedia.org/wiki/List_of_Chinese_administrative_divisions_by_population
wiki_china = {
    'Anhui': 62550000,
    'Beijing': 21710000,
    'Chongqing': 30750000,
    'Fujian': 39110000,
    'Gansu': 26260000,
    'Guangdong': 111690000,
    'Guangxi': 48850000,
    'Guizhou': 35550000,
    'Hainan': 9170000,
    'Hebei': 75200000,
    'Heilongjiang': 37890000,
    'Henan': 95590000,
    'Hubei': 59020000,
    'Hunan': 68600000,
    'Inner Mongolia': 25290000,
    'Jiangsu': 80290000,
    'Jiangxi': 46220000,
    'Jilin': 27170000,
    'Liaoning': 43690000,
    'Ningxia': 6820000,
    'Qinghai': 5980000,
    'Shaanxi': 38350000,
    'Shandong': 100060000,
    'Shanghai': 24180000,
    'Shanxi': 36820000,
    'Sichuan': 83020000,
    'Tianjin': 15570000,
    'Tibet': 3370000,
    'Xinjiang': 24450000,
    'Yunnan': 48010000,
    'Zhejiang': 56570000,
}
china = pd.DataFrame({'population': wiki_china, 'region': 'Asia', 'subregion': 'Eastern Asia'})


# In[ ]:


#https://en.wikipedia.org/wiki/Channel_Islands
wiki_channel_islands = {'Channel Islands': 170499}
channel_islands = pd.DataFrame({'population': wiki_channel_islands, 'region': 'Europe', 'subregion': 'Northern Europe'})


# In[ ]:


# https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_on_cruise_ships
wiki_diamond_princess = {'Diamond Princess': 3711}
diamond_princess = pd.DataFrame({'population': wiki_diamond_princess, 'region': 'Asia', 'subregion': 'Eastern Asia'})


# # Bringing It All Together

# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', parse_dates=['Date'])
train.columns = ['id', 'province_state', 'country_region', 'date', 'confirmed', 'fatal']

# use its alternative name otherwise will overlap with US state Georgia
train['country_region'].update(train['country_region'].str.replace('Georgia', 'Sakartvelo'))

train['entity'] = train['province_state'].where(~train['province_state'].isna(), train['country_region'])

countries = train['entity'].unique()
features = get_restcountries(countries)[['region', 'subregion', 'population']]

for chunk in [get_datausa(), canada, australia, china, channel_islands, diamond_princess]:
    features = features.combine_first(chunk)
features


# In[ ]:


covid = train[['date', 'entity', 'confirmed', 'fatal']].join(features, on='entity')

# gets rid of some data anomalies where cumulative series would drop
covid['confirmed'] = covid.groupby('entity')['confirmed'].cummax()
covid['fatal'] = covid.groupby('entity')['fatal'].cummax()

covid[['confirmed', 'fatal', 'population']] = covid[['confirmed', 'fatal', 'population']].fillna(0).astype('int')
covid.sample(20)


# # Sanity Check

# In[ ]:


covid.groupby('entity').max().pivot_table(index='region', aggfunc='sum', margins=True)


# In[ ]:


covid.to_csv('covid.csv', index=False)


# In[ ]:


covid['date'].max()

