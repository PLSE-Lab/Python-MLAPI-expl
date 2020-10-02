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
for dirname, _, filenames in os.walk('/kaggle/input/ecdc-covid-data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # ** The focus of this notebook is to explore the worldwide COVID-19 geographic distribution.** 
# ## Note: I am using external data from [ECDC](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide) (Thanks to [salmanhiro](https://www.kaggle.com/salmanhiro))

# In[ ]:


# load ECDC current worldwide COVID-19 statistics (file saved from above link)
df = pd.read_excel('/kaggle/input/geodist/COVID-19-geographic-disbtribution-worldwide-2020-03-20.xlsx')

# View the first five rows
df.head(5)


# # Overall Analysis:

# In[ ]:


country = df['Countries and territories'].unique()

print("Number of countries Impacted: ",len(country))
print("Worldwide Cases Reported (as of March 20, 2020): ", df['Cases'].sum())
print("Worldwide Deaths Reported (as of March 20, 2020): ", df['Deaths'].sum())
print("Fatality Rate: "+"{:.2%}".format((df['Deaths'].sum()/df['Cases'].sum())))


# We can see that the Corona Virus (COVID-19) has over 240K cases as of March 20th with a high overall fatality rate of around 4%.

# ## **Analyzing Data for China to validate accuracy:**

# In[ ]:


df[df['Countries and territories']=='China']


# It appears that the data recorded by ECDC is based on daily new cases recorded by country. It may be useful to track the cumulative totals by country as well to see the overall impact of this pandemic worldwide.

# ## Reported Cases Data Visualization Preprocessing:
# Before we can look into visualizations of this dataset, we need to re-format the dataset to show dates as columns.

# In[ ]:


#df2 = df[['Countries and territories','GeoId','DateRep','Cases']]
df2 = df[['Countries and territories','DateRep','Cases']]
df2.rename(columns={'Countries and territories': 'Country','DateRep': 'DateRp'}, inplace=True)
df2.head()


# Use pivot table to display dates reported as separate columns with value as current date's new cases identified for a particular country.

# In[ ]:


df2 = df2.drop_duplicates(subset=['Country','DateRp'])
df_pivot = df2.pivot(index='Country',columns='DateRp')
#df_pivot
#df2.pivot_table(df2, index=['Country','GeoId'],columns='DateRp')
df_pivot.columns = df_pivot.columns.to_flat_index()
df_pivot.columns = ['{}'.format(x[1]) for x in df_pivot.columns if x!= 'Country']
#from datetime import datetime
df_pivot.columns = [col.replace('00:00:00', '') for col in df_pivot.columns]
#df_pivot.rename(columns = lambda x: x.strip('00:00:00'))
#df.columns = [col[:-2] for col in df.columns if col[-2:]=='_x' else col]
df_pivot.fillna(0, inplace=True)
df_pivot.head()


# **Merge above dataset with corresponding country ID codes.** _(This will be used to get the country image icon for display purposes)_

# In[ ]:


geocodes = df[['Countries and territories','GeoId']]
geocodes.drop_duplicates(subset=None, keep="first", inplace=True)
geocodes.head()
merged_left = pd.merge(left=df_pivot, right=geocodes, how='left', left_on='Country', right_on='Countries and territories')
merged_left.head()


# Update cases for each date as a running cumulative total for each country

# In[ ]:


merged_left['url'] = 'https://www.countryflags.io/'+ merged_left.GeoId +'/flat/64.png'
merged_left['url']
#merged_left.head()
merged_left.shape
for i in range(0,len(merged_left)):
    for c in range(1,merged_left.shape[1]-3): 
        merged_left.iat[i, c] = merged_left.iat[i, c] + merged_left.iat[i, c-1]
        #print(df_tot.iat[i, c])
        #df_tot.set_value(i,c,val)


# ### Review final dataset for China:
# We are now getting the cumulative total for each day as expected. Let's now run some visualizations!

# In[ ]:


merged_left[merged_left['Countries and territories'] == 'China']


# ## Saving modified dataset for visualization

# In[ ]:


merged_left.to_excel('cleandata.xlsx')


# # COVID-19 Cases Reported - A History:
# Using [Flourish](https://public.flourish.studio/visualisation/1631776/), let us see the race bar chart for cases over time as below.

# In[ ]:


import IPython
url = "https://public.flourish.studio/visualisation/1631776/"
iframe = '<iframe src=' + url + ' width=700 height=350></iframe>'
IPython.display.HTML(iframe)


# ![display image](https://raw.githubusercontent.com/rsyed1/kaggle/master/Cases2020.gif)

# # COVID-19 Deaths Reported - A History:
# ## TBD

# In[ ]:




