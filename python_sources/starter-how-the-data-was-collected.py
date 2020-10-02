#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ---
# Hi guys!
# <br>Please find below the code, dedicated to the data collection process

# In[ ]:


# !pip install geopy # in case kaggle image doesn't have it


# In[ ]:


import re
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from geopy.geocoders import ArcGIS


# In[ ]:


# check what data do we have
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# import data
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")

# glue datasets together, for convenience
train['is_train'] = True
test['is_train'] = False
df = pd.concat(
    [train, test], 
    sort=False, ignore_index=True
).set_index('id').sort_index()

print(train.shape, test.shape, df.shape)
df.head()


# ### 'Suspicious' locations
# ---
# Dumb regex to filter out email-like, hashtag-like, garbage-like, etc. location candidates

# In[ ]:


# define 'suspicious' locations (fictional, worldwide and/or aren't close to location pattern)

suspicious_locations = '(?:' + '|'.join([
    'worldwide',
    'world',
    'earth',
    'global',
    'narnia',
    'the universe',
    'does it really matter',
    'upstairs',
    'heaven',
    'azeroth',
    'location'  
])

suspicious_locations += '|@|\?+|^missing$|(?:any|else|some|no|every)where|^[0-9]+$|http[s]?|#)'
suspicious_locations


# In[ ]:


col = 'location'

missing_cnt = df[col].isnull().sum()
print(f'Missing values before cleaning: {missing_cnt}')

# fill in with NaNs some strange values (emails, question marks, etc.)
df[col] = df[col].fillna('missing').astype(str)
df.loc[
    (
        df[col].str.contains(suspicious_locations, flags=re.I)
        | (df[col].str.len() < 2) # strange 1-char outliers
    ),
    col
] = np.nan
print(f'Missing values after  cleaning: {df.location.isnull().sum()}')

loc_cnt = df[col].str.lower().value_counts()

min_cnt = 3

print(
    f'Unique locations: \t\t{len(loc_cnt)}'
    f'\nPopular locations (>{min_cnt}): \t{(loc_cnt > min_cnt).sum()}\n',
    f'Popular locations (>{min_cnt}) %: \t{loc_cnt[loc_cnt > min_cnt].sum()/df[col].notnull().sum() * 100 :.2f}'
)

# well, we have common ones (like USA), as well as tweet-unique locations
# as well as fake places like `Narnia` or `Azeroth` :)
# as well as unspecified like `worldwide`
pd.concat([loc_cnt.head(10), loc_cnt.tail(10)])


# While browsing `location` field I noticed some of the rows contain direct (lat,long) pairs within raw string, let's try to find them and use for reverse geocoding
# <br>We'll be using [this modified regex](https://stackoverflow.com/questions/3518504/regular-expression-for-matching-latitude-longitude-coordinates)

# In[ ]:


latlong_regex = (
    '(?P<lat>[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?)),'
    '\s*(?P<lon>[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?))$'
)

valid_coordinates_ind = df[col].fillna('missing').str.contains(latlong_regex)
# check extracted data
print(f'Coordinates found: {valid_coordinates_ind.sum()}/{df[col].notnull().sum()}')
df.loc[valid_coordinates_ind, col].head()


# In[ ]:


# extract lat, long using Pandas functionality
valid_coordinates = df[col].fillna('missing').str.extract(latlong_regex)[valid_coordinates_ind].astype(np.float32)

valid_coordinates.head()


# Please unfold cells below to see code for data downloading

# In[ ]:


from tqdm.notebook import tqdm
from time import sleep


def geocode_address(query: str, geocoder=ArcGIS(), **kwgs):
    """
    get lat/lon and other attributes 
    from query string (address)
    """
    # https://geopy.readthedocs.io/en/stable/#arcgis
    sleep(0.1)
    res = geocoder.geocode(query=query, **kwgs)
    try:
        return {
            'lat': res.point.latitude,
            'lon': res.point.longitude,
            'country': res.raw['attributes'].get('Country', np.nan),
            'city': res.raw['attributes'].get('City', np.nan),
            'match_score': res.raw['score'],
        }
    except (KeyError, AttributeError):
        return {
            'lat': np.nan,
            'lon': np.nan,
            'country': np.nan,
            'city': np.nan,
            'match_score': 0,
        }
    
def geocode_address_rev(query: str, geocoder=ArcGIS(), **kwgs):
    """
    get address attributes 
    from query string (lat/lon pair)
    """
    sleep(0.1)
    lat, lon = [float(c) for c in query.split(',')]
    # https://geopy.readthedocs.io/en/stable/#arcgis
    res = geocoder.reverse(query=query, **kwgs)
    try:
        return {
            'lat': lat,
            'lon': lon,
            'country': res.raw['CountryCode'],
            'city': res.raw.get('City', np.nan),
            'match_score': 100,
        }
    except KeyError:
        return {
            'lat': lat,
            'lon': lon,
            'country': np.nan,
            'city': np.nan,
            'match_score': 0,
        }
    

geocoder = ArcGIS(
    timeout=100,
    user_agent='kaggle_twitter_geodata',
    # or create free account at https://www.arcgis.com/index.html
    # and get up to 1kk requests per month and specify default referer
    # username='your_username', 
    # password='your_password', 
    # scheme='https', 
    # referer='https://www.example.com'
)


# ### Reverse geocoding
# ---
# Given (lat,lon) pairs, extract geoinfo

# In[ ]:


# extract info using reverse geocoding (full)
geodata_inv = {
        ind: geocode_address_rev(query=coords, geocoder=geocoder)
        for (ind, coords) in tqdm(
            valid_coordinates.astype(str)\
            .apply(lambda x: ', '.join(x), axis=1).to_dict().items()
        )
    }

# geodata
df_geo_inv = pd.DataFrame(geodata_inv).T
df_geo_inv.head()


# ### Direct geocoding
# ---
# Given address string, extract geoinfo

# In[ ]:


# DDoS geo-API a little o_0
from joblib import Parallel, delayed
from os import cpu_count

# let's try to extract popular locations (set min_cnt to zero to allow all locations parsing)
# however, you'd probably get 403 error for brute force :)
min_cnt = 15
address_to_code = loc_cnt[loc_cnt > min_cnt].index.tolist()
# comment next 3 lines, it adds least frequent locations to see imperfect matches
print(len(address_to_code))
address_to_code += loc_cnt.tail(10).index.tolist() 
print(len(address_to_code))

# extract info using direct geocoding (partial)
geodata = dict(
    zip(
        address_to_code,
        Parallel(n_jobs=cpu_count(), backend='threading')(
            delayed(geocode_address)(
                query=location, 
                geocoder=geocoder, 
                out_fields=['x', 'y', 'Country', 'Score', 'City']
            )
        for location in tqdm(address_to_code)
        )
    )
)

# geodata
df_geo = pd.DataFrame(geodata).T
df_geo.head()


# <br>**Hint!**
# <br>You're also given `match_score` field (from 0 to 100) that indicates the matching quality of a result given initial request

# In[ ]:


# take a look at imperfect matches (if there are any)...
min_match_score = 90
df_geo[df_geo.match_score < min_match_score].sort_values(by='match_score', ascending=True)
# see, what results you can obtain from imperfect matching :)


# In[ ]:


# ...and clean those outliers
print(df_geo.shape)
df_geo = df_geo[df_geo.match_score >= min_match_score]
print(df_geo.shape)


# Let's map those addresses back to tweet's dataset

# In[ ]:


# map obtained results to initial tweet data and merge with inverse geocoded
geodata = pd.concat([
    # direct geocoding
    df[[col]].astype(str).applymap(str.lower).merge(
        df_geo,
        left_on='location',
        right_index=True,
        how='inner'
    ).drop(columns=[col]),
    # inverse geocoding
    df_geo_inv
]).drop(columns=['match_score'])

# drop possible duplicates
print(df_geo.shape)
geodata = geodata[~geodata.index.duplicated(keep='first')]
print(df_geo.shape)

# save for the latter usage (this was performed on full dataset)
# geodata.reset_index().rename(columns={'index': 'id'})\
# .to_csv('geodata.csv', index=False, encoding='utf-8')

print(geodata.shape)
geodata.head()


# #### Let's see how to append this data as features
# ---

# In[ ]:


# append geodata to the initial dataframe
print(df.shape)
df = df.merge(geodata, left_index=True, right_index=True, how='left')
# drop possible duplicates
df = df[~df.index.duplicated(keep='first')]
print(df.shape)
df.sample(10, random_state=911)


# In[ ]:


# print location target (by country)
# this is only the subset, as it's based on sampled geodata
df[df.is_train].groupby(df.country.replace('', np.nan)).agg({'target': 'mean', col: 'count'}).sort_values(by=[col, 'target'], ascending=[False, False]).head(20)

