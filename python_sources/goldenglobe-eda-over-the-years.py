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


df = pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')


# In[ ]:


df.head()


# **WHICH NOMINEE HAS RECEIVED THE MOST AWARDS IN A YEAR?**

# In[ ]:


df_nominee_year = df.groupby(['nominee', 'year_award'])['win'].sum()


# In[ ]:


df_nominee_year[df_nominee_year==df_nominee_year.max()]


# * Max awards received by a nomination in a year is 2

# **WHICH NOMINEE HAS RECIEVED THE MOST AWARDS OVERALL?**

# In[ ]:


df_nominee = df.groupby(['nominee'])['win'].sum()


# In[ ]:


df_nominee[df_nominee==df_nominee.max()]


# * Meryl Streep has won the most - 8 awards across the years.

# **WHICH WORK HAS WON THE MOST AWARDS IN AN EVENT?**

# In[ ]:


df_film_year = df.groupby(['film','year_award'])['win'].sum()


# In[ ]:


df_film_year[df_film_year==df_film_year.max()]


# * LA LA LAND(2017) has received the most awards in an event

# In[ ]:


df.head()


# In[ ]:


df_year = pd.DataFrame()
df_year['total_nominations'] = df.groupby(['year_award'])['win'].count()
df_year['wins'] = df.groupby(['year_award'])['win'].sum()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(df_year.index, df_year['total_nominations'], label = 'nominations')
plt.plot(df_year.index, df_year['wins'], label = 'wins')
plt.legend()


# **ABOVE GRAPH SHOWS HOW THE NUMBER OF NOMINATIONS AND AWARDS HAVE CHANGED OVER THE YEARS**

# **WHICH COUNTRY HAS RECEIVED MOST AWARDS AT A CEREMONY AND OVERALL?**

# In[ ]:


import re
pattern = re.compile('[A-z ]+,')


# In[ ]:


def get_country(film):
    rs = re.findall(pattern, film)
    if(rs):
        return (rs[0]).replace(',','')
    else:
        return ''
get_country('United Kingdom, asas')


# In[ ]:


df['country']=df[df['category'].str.contains('Best Motion Picture - Foreign Language')]['film'].apply(get_country)


# In[ ]:


df_country = pd.DataFrame()


# In[ ]:


df_country=df_country.append(df[df['category'].str.contains('Best Motion Picture - Foreign Language')],ignore_index=True)


# In[ ]:


df['country'] = df[df['category'].str.contains('Foreign Film - Foreign Language')]['nominee'].apply(get_country)


# In[ ]:


df_country=df_country.append(df[df['category'].str.contains('Foreign Film - Foreign Language')],ignore_index=True)


# In[ ]:


df_country


# In[ ]:


df_country_awards_year = df_country.groupby(['country','year_award'])['win'].sum()


# In[ ]:


df_country_awards_year[df_country_awards_year==df_country_awards_year.max()]


# * SWEDEN & WEST GERMANY have won max - 2 awards in a single ceremony

# In[ ]:


df_country[df_country['year_award']==1973]


# In[ ]:


df_country_overall = df_country.groupby(['country'])['win'].sum()


# In[ ]:


df_country_overall[df_country_overall==df_country_overall.max()]


# * FRANCE has won the most awards over the years

# **IS THERE ANY CATEGORY WITHOUT AWARD OR MULTIPLE AWARDS?**

# In[ ]:


df_category_year = df.groupby(['category','year_award'])['win'].sum()


# In[ ]:


df_category_year[df_category_year==df_category_year.max()]


# * Six awards for a single category - Television Achievement in 1959

# In[ ]:


df_category_year[df_category_year==df_category_year.min()]


# * Turns out there a 90 categories across the year without any awards!

# In[ ]:




