#!/usr/bin/env python
# coding: utf-8

# ## Korean buildings are being shut down when a corona confirmer visits.
# 
# People also see the corona confirmer's route and avoid visits to the area.
# 
# Many small businesses are suffering from this. **I wanted to investigate how enormous the damage was.**
# 
# So, I downloaded the floating population data provided by SKtelecom, a Korean telecommunications company.
# 
# Unfortunately, this data is lacking because it provides free data only for Seoul. But there is no problem in identifying trends to some extent.
# (I uploaded data and description to a [public dataset](https://www.kaggle.com/incastle/seoul-floating-population-2020))
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


patient = pd.read_csv('../input/coronavirusdataset/patient.csv')
trend = pd.read_csv('../input/coronavirusdataset/trend.csv')
time = pd.read_csv('../input/coronavirusdataset/time.csv')
route = pd.read_csv('../input/coronavirusdataset/route.csv')


# In[ ]:


patient.head(12)


# ## Floating population data is still available until January
# - so we should use Corona January data 

# In[ ]:


route[(route['date'] < '2020-02-01') & ( route['province'] == 'Seoul')].groupby('city')['patient_id'].nunique()


# In[ ]:


fp_01 = pd.read_csv("../input/seoul-floating-population-2020/fp_2020_01_english.csv")


# In[ ]:


fp_01['date'] = fp_01['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y.%m.%d").date()).astype('str')
fp_01['date'] = fp_01['date'].apply(lambda x: x[8:]) ## only use day

fp_01 = fp_01.sort_values(['date', 'hour', 'birth_year', 'sex'])  ## this data is not sorted.
fp_01.reset_index(drop= True, inplace = True)


# In[ ]:


fp_01.head()


# In[ ]:


patient['confirmed_date'][0]


# In[ ]:


print("first Infected date in korea: ", patient['confirmed_date'][0])


# ### I will draw a very simple plot. i just want to introduce floating population data
# 
# - draw blue vertical line : Date of First Infection in Korea 
# - draw red vertical line : Date of first visit to the affected person
# 

# ### Gangnam-gu

# In[ ]:


## Date when the visitor went to Gangnam-gu
sorted(list(route[route['city'] == 'Gangnam-gu']['date'].unique()))


# In[ ]:





# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Gangnam-gu')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '22', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Gangnam-gu is 22 days')
plt.show()


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Gangnam-gu')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '22', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Gangnam-gu is 22 days')
plt.show()


# ## Jongno-gu

# In[ ]:


## Date when the visitor went to Jongno-gu
sorted(list(route[route['city'] == 'Jongno-gu']['date'].unique()))


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Jongno-gu')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '26', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Jongno-gu is 26 days')
plt.show()


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Jongno-gu')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '26', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Jongno-gu is 26 days')
plt.show()


# ## Jongno-gu

# In[ ]:


## Date when the visitor went to Jongno-gu
sorted(list(route[route['city'] == 'Jung-gu']['date'].unique()))


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Jung-gu') & (fp_01['date'] > '01')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '20', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Jung-gu is 19 days')
plt.show()


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Jung-gu') & (fp_01['date'] > '01')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '20', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Jung-gu is 19 days')
plt.show()


# ## Jungnang-gu

# In[ ]:


## Date when the visitor went to Jungnang-gu
sorted(list(route[route['city'] == 'Jungnang-gu']['date'].unique()))


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Jungnang-gu') & (fp_01['date'] > '01')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '28', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Jungnang-gu is 28 days')
plt.show()


# In[ ]:


tmp = fp_01[(fp_01['city'] == 'Jungnang-gu') & (fp_01['date'] > '01')]
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)
plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 
plt.axvline(x = '28', color = 'r', ls = '--', alpha = 0.6) 
plt.title('The patient"s first visit to Jungnang-gu is 28 days')
plt.show()


# In[ ]:





# In[ ]:





# ## January 24-27 is Korea holidays
# - External factors occur in the floating population pattern
# - Since there is not much data yet, we randomly want to compare patterns between ***January 17 (Friday, before the corona) vs. January 31 (Friday, after the New Year holidays)***

# In[ ]:


def plot_dist_col(train_df, test_df, title ):
    '''plot dist curves for train and test weather data for the given column name'''
    train_df = pd.DataFrame(train_df.groupby('hour')['fp_num'].sum())
    train_df.reset_index(inplace = True)
    
    test_df = pd.DataFrame(test_df.groupby('hour')['fp_num'].sum())
    test_df.reset_index(inplace = True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(data=train_df, x='hour', y='fp_num', color='green', ax=ax).set_title('fp_num', fontsize=16)
    sns.lineplot(data=test_df, x='hour', y='fp_num', color='purple', ax=ax).set_title('fp_num', fontsize=16)
    plt.xlabel('hour', fontsize=16)
    plt.title(title, fontsize=20)
    plt.legend(['17day', '31day'])
    plt.show()


# ## Gangnam - gu

# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu')]
gan17 = pd.DataFrame(gan17.groupby(['hour', 'birth_year'])['fp_num'].sum())
gan17.reset_index(inplace = True)


# In[ ]:


gan17


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu')]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Gangnam-gu')]
plot_dist_col(gan17, gan31, 'Gangnam-gu pattern')


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu') & (fp_01['birth_year'] == 20)]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Gangnam-gu') & (fp_01['birth_year'] == 20)]
plot_dist_col(gan17, gan31, 'Gangnam-gu 20 years old pattern')


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu')]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Gangnam-gu')]
plot_dist_col(gan17, gan31, 'Gangnam-gu pattern')


# In[ ]:





# ## Jungnang-gu

# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jungnang-gu')]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jungnang-gu')]
plot_dist_col(gan17, gan31, 'Jungnang-gu pattern')


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jungnang-gu') & (fp_01['birth_year'] == 20)]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jungnang-gu') & (fp_01['birth_year'] == 20)]
plot_dist_col(gan17, gan31, 'Jungnang-gu 20 years old pattern')


# In[ ]:





# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jongno-gu')]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jongno-gu')]
plot_dist_col(gan17, gan31, 'Jongno-gu pattern')


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jongno-gu') & (fp_01['birth_year'] == 20)]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jongno-gu') & (fp_01['birth_year'] == 20)]
plot_dist_col(gan17, gan31, 'Jongno-gu 20 years old pattern')


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jung-gu')]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jung-gu')]
plot_dist_col(gan17, gan31, 'Jung-gu  pattern')


# In[ ]:


gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jung-gu') & (fp_01['birth_year'] == 20)]
gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jung-gu') & (fp_01['birth_year'] == 20)]
plot_dist_col(gan17, gan31, 'Jung-gu 20 years old pattern')


# - Gangnam-gu, Jongno-gu, Jung-gu and Jungnang-gu have different regional characteristics
# 
# - Jungnang-gu is a residential area and others are industrial and consumer areas.
# 
# - Therefore, industrial areas have many floating populations during the day, while residential areas have fewer floating populations during the day.
# 
# - In combination with these characteristics, coronas are generated, the floating population in the industrial areas is markedly reduced, and the floating population is increasing in the residential areas.
# 
# - In other words, they don't go to work or entertain and go back and forth near their homes.

# ### I'll update it as soon as February data is uploade, thank you

# In[ ]:




