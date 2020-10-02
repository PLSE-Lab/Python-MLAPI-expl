#!/usr/bin/env python
# coding: utf-8

# Which populations are at risk of contracting COVID-19?
# ======================================================

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'state' in df.columns and 'county' in df.columns:
                print(file_path)
                print(df.columns)

# Any results you write to the current directory are saved as output.os.path.join(dirname, filename)


# In[ ]:


input_dir = '/kaggle/input'


# covid-19-county-level-data
# --------------------------

# In[ ]:


file_path = input_dir + '/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv'
covid_19_county_level_data = pd.read_csv(file_path)
mn_data = covid_19_county_level_data[covid_19_county_level_data.loc[:, 'state'] == 'Minnesota']
mn_data.loc[:, 'date'] = pd.to_datetime(mn_data['date'])


# In[ ]:


recent_date = mn_data['date'].max()
recent_date_rows = mn_data["date"] == recent_date
mn_recent_data = mn_data.loc[recent_date_rows]
mn_recent_data = mn_recent_data.dropna(subset=['fips'])
mn_recent_data.loc[:, 'fips'] = mn_recent_data['fips'].astype(int)


# 
# Date of most recent data:
# 

# In[ ]:


recent_date


# In[ ]:


from IPython.display import display, HTML
print("Minnesota data by county:")
display(mn_recent_data)


# In[ ]:


date_agg_df = mn_data.groupby(['date']).agg({'cases': 'sum','deaths': 'sum'}).reset_index()


# In[ ]:


plt.plot(date_agg_df['date'], date_agg_df['cases'], 'y', label='cases')
plt.plot(date_agg_df['date'], date_agg_df['deaths'], 'r', label='deaths')
plt.xlabel('date')
plt.ylabel('number')
plt.xticks(rotation=70)
plt.legend(loc='upper left');


# cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level
# -----------------------------------------------------------------

# Data from [CDC SVI 2016 Documentation](https://svi.cdc.gov/Documents/Data/2016_SVI_Data/SVI2016Documentation.pdf)

# In[ ]:


cdcs_file_path = input_dir + '/uncover/UNCOVER/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv'
cdcs_social_vulnerability_index = pd.read_csv(cdcs_file_path)
mn_cdcs_social_vulnerability_index = cdcs_social_vulnerability_index[cdcs_social_vulnerability_index['state'] == 'MINNESOTA']
mn_cdcs_social_vulnerability_index.head


# In[ ]:


print("CDCS social vulnerability data by county:")
display(mn_cdcs_social_vulnerability_index[['county', 'fips', 'area_sqmi', 'e_totpop']])


# In[ ]:


merged_inner = pd.merge(left=mn_recent_data, right=mn_cdcs_social_vulnerability_index, left_on='fips', right_on='fips')


# In[ ]:


merged_inner = merged_inner.drop(['county_y'], axis=1)
merged_inner = merged_inner.rename(columns={"county_x": "county"})
for col in merged_inner.columns: 
    print(col) 


# In[ ]:


print("Data by county:")
merged_inner['cases per capita'] = merged_inner['cases']/merged_inner['e_totpop']
merged_inner['deaths per ten thousand'] = 10000 * merged_inner['deaths'] / merged_inner['e_totpop']
display(merged_inner[['county', 'cases', 'cases per capita', 'deaths', 'deaths per ten thousand', 'area_sqmi', 'e_totpop']])


# In[ ]:


import geopandas as gpd


# In[ ]:


world = gpd.read_file(cdcs_file_path)


# ### From the [COVID Tracking Project Data API](https://covidtracking.com/api)

# #### MN Historical Data

# 
# 
# import requests
# 
# r = requests.get('https://covidtracking.com/api/v1/states/MN/daily.json')
# x = r.json()
# df = pd.DataFrame(x)

# In[ ]:


df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
df.head()


# In[ ]:


plt.plot(df['date'], df['positive'], 'y', label='positive')
plt.plot(df['date'], df['negative'], 'r', label='negative')
plt.xlabel('date')
plt.ylabel('number')
plt.xticks(rotation=70)
plt.legend(loc='upper left');


# In[ ]:


plt.plot(df['date'], df['hospitalizedCurrently'], 'r', label='hospitalizedCurrently')
plt.xlabel('date')
plt.ylabel('number')
plt.xticks(rotation=70)
plt.legend(loc='upper left');

