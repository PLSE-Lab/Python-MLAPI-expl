#!/usr/bin/env python
# coding: utf-8

# Update 2016-08-07: Note that I am just getting started here. Please come back for more.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ### Loading and cleaning data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


# load data into pandas DataFrame with low_memory=False to suppress warning
zika_df = pd.read_csv(os.path.join('..', 'input', 'cdc_zika.csv'),
                      low_memory=False)

keep_rows = np.logical_and(pd.notnull(zika_df['report_date']),
                           pd.notnull(zika_df['value'])) 
zika_df = zika_df[keep_rows]
print('Removed {:d} out of {:d} rows with missing '
      'report_date or missing value.'.format(len(keep_rows) - sum(keep_rows),
                                             len(keep_rows)))

# clean report_date as some dates are delimited by underscores and some by hyphens,
# then convert to DatetimeIndex and sort by report_date
zika_df['report_date'] = pd.to_datetime([d.replace('_', '-') for d in zika_df['report_date']],
                               format='%Y-%m-%d')
zika_df.sort_values(by='report_date', inplace=True, kind='mergesort')  # 'mergesort' is stable


# In[ ]:


zika_df.head(n=3)


# In[ ]:


zika_df = zika_df[zika_df['unit'] == 'cases']
sorted(set(zika_df['value']))
zika_df['value'] = pd.to_numeric(zika_df['value'], 'coerce')


# ### Initial look at the data reported on country level

# In[ ]:


country_df = zika_df[zika_df['location_type']=='country']

all_countries = set(country_df['location'])
print(os.linesep.join(all_countries))


# We see that some countries such as Brazil, Colombia, and the US do not report cases on a country-wide level. More on these countries later.

# In[ ]:


data_categories = sorted(country_df['data_field'].unique())
print(os.linesep.join(data_categories))


# First plot `total_zika_confirmed` cases per country over time.

# In[ ]:


country_total_confirmed_df = country_df.loc[country_df['data_field'] == 'total_zika_confirmed',
                                           ['report_date', 'location', 'value']]
country_total_confirmed_df['value'] = pd.to_numeric(country_total_confirmed_df['value'])


# In[ ]:


grouped = country_total_confirmed_df.groupby('location')
fig, ax = plt.subplots(1)
for key, group in grouped:
   group.plot(ax=ax, x='report_date', y='value', label=key)


# ## TODO
# 
# - figure out how other countries that do report on country-wide level but not as `total_zika_confirmed` report their data
# - analyze cases in countries not reporting on country-wide level

# In[ ]:


country_grouped = country_df.groupby('location')


# In[ ]:


for name, group in country_grouped:
    print(name)
    print(group['data_field'].unique())


# In[ ]:


countries = [s[0] for s in zika_df['location'].str.split('-')]
zika_country_grouped = zika_df.groupby(countries)
for name, group in zika_country_grouped:
    print(name)
    print(group['location'].unique())


# ## Inspecting each country and cleaning data
# 
# ### Argentina

# In[ ]:


argentina_df = zika_country_grouped.get_group('Argentina')
expected_data_fields = set(argentina_df['data_field'])
print(expected_data_fields)
for name, group in argentina_df.groupby(['report_date', 'location']):
    assert expected_data_fields == set(group['data_field'])


# In[ ]:


argentina_df['location_type'].unique()


# In[ ]:


case_series = argentina_df[argentina_df['data_field'] == 'cumulative_confirmed_local_cases'].groupby('report_date')['value'].sum()


# In[ ]:


ax = case_series.plot(title='Argentina')
ax.set_ylabel('cumulative_confirmed_local_cases')

