#!/usr/bin/env python
# coding: utf-8

# # AQI During Lockdown - India
# The dataset contains air quality data and AQI (Air Quality Index) at hourly and daily level of various stations across multiple cities in India.
# 
# ## Cities
# Ahmedabad, Aizawl, Amaravati, Amritsar, Bengaluru, Bhopal, Brajrajnagar, Chandigarh, Chennai, Delhi, Ernakulam, Gurugram, Guwahati, Hyderabad, Jaipur, Jorapokhar, Kochi, Kolkata, Lucknow, Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, Visakhapatnam  
#   
# The data has been made publicly available by the Central Pollution Control Board: https://cpcb.nic.in/ which is the official portal of Government of India. They also have a real-time monitoring app: https://app.cpcbccr.com/AQI_India/

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Import Dependencies

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime


# In[ ]:


city_data = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')
city_data.head()


# In[ ]:


city_data.shape


# In[ ]:


print("Total null records in Data:\n", city_data.isnull().sum())


# #### Citywise instamces of Severe bucket of Air Quality index.
# Here Ahmedabad, Delhi, Patna seems to be highets in case and Ahemdabad to be very much higher than others.

# In[ ]:


city_data[city_data['AQI_Bucket'] == 'Severe']['City'].value_counts()


# In[ ]:


pd.to_datetime(city_data[(city_data['AQI_Bucket'] == 'Severe') &
                         (city_data['City'] == 'Ahmedabad')]['Date']).dt.year.value_counts()


# As we can here, in the city of Ahmedabad in year 2018, 2019 more severe instances has been found. In 2018 around 3/4 year, the condition of air quality was severe. As the whole data of 2020 is not available we see it down the list.

# In[ ]:


city_data['Date'] = pd.to_datetime(city_data['Date'])

city_data[(city_data['AQI_Bucket'] == 'Severe') & 
          (city_data['City'] == 'Ahmedabad') & 
          (city_data.Date.dt.year != 2020)].Date.dt.month.value_counts()


# Here we can clearly see that the month of May, June and December we always have very less pollution as compared to other months.This might be because the Firecrackers in Diwali. To find out the real cause is firecrackers or not we have to see yearly which month Diwali was in which year and did that month spiked up the pollution.

# Firstly in 2018 diwali was in November and in 2019 it was in October.

# In[ ]:


city_data[(city_data.City == 'Ahmedabad') & 
          (city_data.Date.dt.year == 2018)].groupby(city_data.Date.dt.month)['AQI'].sum()


# In[ ]:


city_data[(city_data.City == 'Ahmedabad') & 
          (city_data.Date.dt.year == 2019)].groupby(city_data.Date.dt.month)['AQI'].sum()


# So, clearly there is spike in month of october in 2019 and in november in 2018 with respect to ther months in same year.

# In[ ]:


mis_val = city_data.isnull().sum()

mis_val_percent = 100 * mis_val / len(city_data)
print(mis_val_percent)

Mis_val = pd.concat([mis_val, mis_val_percent], axis=1)
Mis_val = Mis_val.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})


# In[ ]:


Mis_val = Mis_val[Mis_val.iloc[:,1] != 0].sort_values(by = '% of Total Values',
                                                      ascending = False).style.background_gradient(cmap = 'Reds')

Mis_val


# ### Missing values with respect to year

# In[ ]:


for i in range(2015,2020): 
    print('Year:', i, '- Missing Values',
          '\n', 100*(city_data.groupby(city_data.Date.dt.year).get_group(i).isnull().sum() / 
                                     city_data.groupby(city_data.Date.dt.year).get_group(i).shape[0]), '\n\n\n')


# ## PM2.5

# In[ ]:


city_data['PM2.5'].describe()


# In[ ]:


100*(city_data['PM2.5'].isnull().sum() / city_data.shape[0]) 


# So 16% of the data points have PM2.5 values missing in whole dataset.

# ### Plotting Mean AQI vs Month and Year
# Ignore 2020 as data is whole year data is not yet available.

# In[ ]:


by_year = city_data.groupby([city_data.Date.dt.year]).mean()
by_month = city_data.groupby([city_data.Date.dt.month]).mean()

plt.figure()
plt.xlabel('Month')
plt.ylabel('Mean_AQI')
plt.plot(by_month.index.get_level_values(0),by_month['PM2.5'])


# In[ ]:


plt.figure()

plt.xlabel('Year')
plt.ylabel('Mean_AQI')
plt.plot(by_year.index.get_level_values(0),by_year['PM2.5']) 


# In[ ]:


city_data_not_2020 = city_data[city_data.Date.dt.year != 2020]


# In[ ]:


by_month_not_2020 = city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()


# In[ ]:


plt.figure()

plt.plot(by_month_not_2020.index.get_level_values(0),by_month_not_2020['PM2.5']) 


# As we can see all over the India the trend of PM2.5 levels going down from June to September which are generally the months of Monsoon.

# In[ ]:


plt.figure()

plt.plot(city_data[city_data.City == 'Chennai'].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),
         city_data[city_data.City == 'Chennai'].groupby([city_data.Date.dt.month]).mean()['PM2.5'])


# In[ ]:


plt.figure()

plt.plot(city_data[city_data.City == 'Delhi'].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),
         city_data[city_data.City == 'Delhi'].groupby([city_data.Date.dt.month]).mean()['PM2.5'])


# In[ ]:


plt.figure()

plt.plot(city_data_not_2020[city_data_not_2020.City == 'Delhi'].groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),
         city_data_not_2020[city_data_not_2020.City == 'Delhi'].groupby([city_data_not_2020.Date.dt.month]).mean()['PM2.5'])


# In[ ]:


plt.figure()

plt.plot(city_data[city_data.City == 'Mumbai'].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),
         city_data[city_data.City == 'Mumbai'].groupby([city_data.Date.dt.month]).mean()['PM2.5'])


# In[ ]:


plt.figure()

plt.plot(city_data_not_2020[city_data_not_2020.City == 'Mumbai'].groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),
         city_data_not_2020[city_data_not_2020.City == 'Mumbai'].groupby([city_data_not_2020.Date.dt.month]).mean()['PM2.5'])


# The Monsoon clearly has effect on PM2.5 as in Delhi and Mumbai, as monsoon is mostly in June, July, August and September (6,7,8,9) PM 2.5 goes way down.  
# Whereas in Chennai where monsoon is far after than these cities, has relatively high PM2.5 in June, July and starts dropping after these months.

# ### CO - Carbon monoxide

# In[ ]:


plt.figure()

plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),
         city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()['CO'])


# In[ ]:


plt.figure()

plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),
         city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()['CO'])


# In[ ]:


plt.figure()

plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.year]).mean().index.get_level_values(0),
         city_data_not_2020.groupby([city_data_not_2020.Date.dt.year]).mean()['CO'])


# In[ ]:


city_data_not_2020.groupby([city_data_not_2020.Date.dt.year]).mean()['CO']


# In[ ]:


plt.figure()

plt.plot(city_data[city_data.Date.dt.year == 2020].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),
         city_data[city_data.Date.dt.year == 2020].groupby([city_data.Date.dt.month]).mean()['CO'])


# In[ ]:


plt.figure(figsize=(10,10))
city_data_not_2020.boxplot()


# ## Difference in three months February, March and April 2020 in Mumbai, Delhi and Chennai.

# ###  Mumbai

# In[ ]:


plt.figure(figsize=(10, 5))

plt.plot(city_data[(city_data.Date.dt.month == 2) & 
                   (city_data.Date.dt.year == 2020) &
                   (city_data.City == 'Mumbai')]['Date'],
         city_data[(city_data.Date.dt.month == 2) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Mumbai')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(10, 5))

plt.plot(city_data[(city_data.Date.dt.month == 3) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Mumbai')]['Date'],
         city_data[(city_data.Date.dt.month == 3) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Mumbai')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(10,5))

plt.plot(city_data[(city_data.Date.dt.month == 4) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Mumbai')]['Date'],
         city_data[(city_data.Date.dt.month == 4) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Mumbai')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


Mean_co_for_Mumbai_Feb_2020 = city_data[(city_data.Date.dt.month == 2) & 
                                        (city_data.Date.dt.year == 2020) & 
                                        (city_data.City == 'Mumbai')]['CO'].mean()

Mean_co_for_Mumbai_March_2020 = city_data[(city_data.Date.dt.month == 3) & 
                                          (city_data.Date.dt.year == 2020) & 
                                          (city_data.City == 'Mumbai')]['CO'].mean()

Mean_co_for_Mumbai_April_2020 = city_data[(city_data.Date.dt.month == 4) & 
                                          (city_data.Date.dt.year == 2020) & 
                                          (city_data.City == 'Mumbai')]['CO'].mean()


print('Mean Carbon oxide in Feb 2020 in Mumbai', Mean_co_for_Mumbai_Feb_2020)
print('Mean Carbon oxide in March 2020 in Mumbai', Mean_co_for_Mumbai_March_2020)
print('Mean Carbon oxide in April 2020 in Mumbai', Mean_co_for_Mumbai_April_2020)


# ### Delhi

# In[ ]:


plt.figure(figsize=(10,5))

plt.plot(city_data[(city_data.Date.dt.month == 2) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Delhi')]['Date'],
         city_data[(city_data.Date.dt.month==2) & 
                   (city_data.Date.dt.year==2020) & 
                   (city_data.City=='Delhi')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(10,5))

plt.plot(city_data[(city_data.Date.dt.month == 3) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Delhi')]['Date'],
         city_data[(city_data.Date.dt.month == 3) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Delhi')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(10,5))

plt.plot(city_data[(city_data.Date.dt.month == 4) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Delhi')]['Date'],
         city_data[(city_data.Date.dt.month == 4) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Delhi')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


Mean_co_for_Delhi_Feb_2020 = city_data[(city_data.Date.dt.month == 2) & 
                                       (city_data.Date.dt.year == 2020) & 
                                       (city_data.City == 'Delhi')]['CO'].mean()

Mean_co_for_Delhi_March_2020 = city_data[(city_data.Date.dt.month == 3) & 
                                         (city_data.Date.dt.year == 2020) & 
                                         (city_data.City == 'Delhi')]['CO'].mean()

Mean_co_for_Delhi_April_2020 = city_data[(city_data.Date.dt.month == 4) & 
                                         (city_data.Date.dt.year == 2020) & 
                                         (city_data.City == 'Delhi')]['CO'].mean()


print('Mean Carbon oxide in Feb 2020 in Delhi',Mean_co_for_Delhi_Feb_2020)
print('Mean Carbon oxide in March 2020 in Delhi',Mean_co_for_Delhi_March_2020)
print('Mean Carbon oxide in April 2020 in Delhi',Mean_co_for_Delhi_April_2020)


# ### Chennai

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Co emmision in Chennai in Month of February')

plt.plot(city_data[(city_data.Date.dt.month == 2) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Chennai')]['Date'],
         city_data[(city_data.Date.dt.month == 2) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City=='Chennai')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Co emmision in Chennai in Month of March')

plt.plot(city_data[(city_data.Date.dt.month == 3) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Chennai')]['Date'],
         city_data[(city_data.Date.dt.month == 3) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City=='Chennai')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Co emmision in Chennai in Month of April')

plt.plot(city_data[(city_data.Date.dt.month == 4) & 
                   (city_data.Date.dt.year== 2020) & 
                   (city_data.City == 'Chennai')]['Date'],
         city_data[(city_data.Date.dt.month == 4) & 
                   (city_data.Date.dt.year == 2020) & 
                   (city_data.City == 'Chennai')]['CO'])

plt.xticks(rotation=30)


# In[ ]:


Mean_co_for_Chennai_Feb_2020 = city_data[(city_data.Date.dt.month == 2) & 
                                         (city_data.Date.dt.year == 2020) & 
                                         (city_data.City == 'Chennai')]['CO'].mean()

Mean_co_for_Chennai_March_2020 = city_data[(city_data.Date.dt.month == 3) & 
                                           (city_data.Date.dt.year == 2020) & 
                                           (city_data.City == 'Chennai')]['CO'].mean()

Mean_co_for_Chennai_April_2020 = city_data[(city_data.Date.dt.month == 4) & 
                                           (city_data.Date.dt.year == 2020) & 
                                           (city_data.City == 'Chennai')]['CO'].mean()


print('Mean Carbon oxide in Feb 2020 in Chennai',Mean_co_for_Chennai_Feb_2020)
print('Mean Carbon oxide in March 2020 in Chennai',Mean_co_for_Chennai_March_2020)
print('Mean Carbon oxide in April 2020 in Chennai',Mean_co_for_Chennai_April_2020)


# In[ ]:


Mean_co_in_Feb_March_April_2020 = pd.DataFrame()
for i in city_data.City.unique():
    if i!= 'Ahmedabad':
        Mean_co_in_Feb_March_April_2020[i] = [city_data[(city_data.Date.dt.month == 2) &
                                                      (city_data.Date.dt.year == 2020) &
                                                      (city_data.City == i)]['CO'].mean(),
                                            city_data[(city_data.Date.dt.month == 3) &
                                                      (city_data.Date.dt.year == 2020) &
                                                      (city_data.City == i)]['CO'].mean(),
                                            city_data[(city_data.Date.dt.month == 4) &
                                                      (city_data.Date.dt.year == 2020) &
                                                      (city_data.City == i)]['CO'].mean()]


# In[ ]:


Mean_co_in_Feb_March_April_2020.transpose().plot(figsize=(10,10), kind='bar')


# We can see the trend of mean of carbon monoxide emission for 3 months feb, march and april of 2020 is going down. This trend we can mostly see by large margins in the dense populated cities and famous tourist destinations. One city is different about it and that is Talcher. It is showing eactly opposite trend that of populated cities. Let's see if we can find more about past emissions in the city of coal mining, Talcher.

# In[ ]:


city_data[city_data.City == 'Talcher'].isnull().sum() / len(city_data[city_data.City == 'Talcher']) 


# Missing values related to CO is about 16% in the whole dataset.

# In[ ]:


city_data[(city_data.City == 'Talcher') & 
          (city_data.Date.dt.year == 2019)].isnull().sum()


# About 15 values are missing in year 2019 in record of CO. This is small value so it won't affect our mean much.

# In[ ]:


plt.figure(figsize=(10,5))

plt.plot(city_data[(city_data.City == 'Talcher') & 
                   (city_data.Date.dt.year == 2019)]['Date'],
         city_data[(city_data.City == 'Talcher') & 
                   (city_data.Date.dt.year == 2019)]['CO'])


# In[ ]:


city_data[(city_data.City == 'Talcher') & 
          (city_data.Date.dt.year == 2019)]['CO'].mean()


# In[ ]:


city_data[(city_data.City == 'Talcher') & 
          (city_data.Date.dt.year == 2020)]['CO'].mean()


# Here the mean of carbon monoxide emission rate has not changed. Even this area has less population, it has its coal reserves so due to coal mines and not the vehicles, carbon monoxide emission rate is affected.   
# 
# So we can ignore this city and other two Brajrajnagar, Odisha.

# ## AQI

# In[ ]:


plt.figure()

plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),
         city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()['AQI'])


# In[ ]:


Mean_AQI_in_Feb_March_April_2020 = pd.DataFrame()
for i in city_data.City.unique():
    if i != 'Ahmedabad':
        Mean_AQI_in_Feb_March_April_2020[i] = [city_data[(city_data.Date.dt.month == 2) & 
                                                         (city_data.Date.dt.year == 2020) & 
                                                         (city_data.City == i)]['AQI'].mean(),
                                               city_data[(city_data.Date.dt.month == 3) & 
                                                         (city_data.Date.dt.year == 2020) & 
                                                         (city_data.City == i)]['AQI'].mean(),
                                               city_data[(city_data.Date.dt.month == 4) & 
                                                         (city_data.Date.dt.year == 2020) & 
                                                         (city_data.City == i)]['AQI'].mean()]


# In[ ]:


Mean_AQI_in_Feb_March_April_2020.transpose().plot(figsize=(10,10), kind='bar')


# In[ ]:


Mean_AQI_in_Feb_March_April_2019 = pd.DataFrame()
for i in city_data.City.unique():
    if i != 'Ahmedabad':
        Mean_AQI_in_Feb_March_April_2019[i] = [city_data[(city_data.Date.dt.month == 2) & 
                                                         (city_data.Date.dt.year == 2019) & 
                                                         (city_data.City == i)]['AQI'].mean(),
                                               city_data[(city_data.Date.dt.month == 3) & 
                                                         (city_data.Date.dt.year == 2019) & 
                                                         (city_data.City == i)]['AQI'].mean(),
                                               city_data[(city_data.Date.dt.month == 4) & 
                                                         (city_data.Date.dt.year == 2019) & 
                                                         (city_data.City == i)]['AQI'].mean()]
Mean_AQI_in_Feb_March_April_2019.transpose().plot(figsize=(10,10), kind='bar')


# In[ ]:


plt.figure()

plt.plot(city_data[city_data.Date.dt.year != 2020].groupby([city_data.Date.dt.year]).mean().index.get_level_values(0),
         city_data[city_data.Date.dt.year != 2020].groupby([city_data.Date.dt.year]).mean()['AQI'])


# In[ ]:


plt.figure()

plt.plot(city_data[city_data.Date.dt.year == 2020].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),
         city_data[city_data.Date.dt.year == 2020].groupby([city_data.Date.dt.month]).mean()['AQI'])


# AQI is dropping over the years.This might or might not be because the null values in previous three years.But if we see the 2018 and 2019 around only 11% and 4% data is missing and still we can see the downward flow of AQI. 

# ## NO - Nitrogen Oxide

# In[ ]:


Mean_no_in_Feb_March_April_2020 = pd.DataFrame()
for i in city_data.City.unique():
    if i != 'Ahmedabad':
        Mean_no_in_Feb_March_April_2020[i] = [city_data[(city_data.Date.dt.month == 2) & 
                                                        (city_data.Date.dt.year == 2020) & 
                                                        (city_data.City == i)]['NO'].mean(),
                                              city_data[(city_data.Date.dt.month == 3) & 
                                                        (city_data.Date.dt.year == 2020) & 
                                                        (city_data.City == i)]['NO'].mean(),
                                              city_data[(city_data.Date.dt.month == 4) & 
                                                        (city_data.Date.dt.year == 2020) & 
                                                        (city_data.City == i)]['NO'].mean()]


# In[ ]:


Mean_no_in_Feb_March_April_2020.transpose().plot(figsize=(10,10), kind='bar')


# In[ ]:


Mean_no_in_Feb_March_April_2019 = pd.DataFrame()
for i in city_data.City.unique():
    if i != 'Ahmedabad':
        Mean_no_in_Feb_March_April_2019[i] = [city_data[(city_data.Date.dt.month == 2) & 
                                                        (city_data.Date.dt.year == 2019) & 
                                                        (city_data.City == i)]['NO'].mean(),
                                              city_data[(city_data.Date.dt.month == 3) & 
                                                        (city_data.Date.dt.year == 2020) & 
                                                        (city_data.City == i)]['NO'].mean(),
                                              city_data[(city_data.Date.dt.month == 4) & 
                                                        (city_data.Date.dt.year == 2019) & 
                                                        (city_data.City == i)]['NO'].mean()]


# In[ ]:


Mean_no_in_Feb_March_April_2019.transpose().plot(figsize=(10,10), kind='bar')


# In[ ]:


Mean_no_in_Feb_March_April_2018 = pd.DataFrame()
for i in city_data.City.unique():
    if i != 'Ahmedabad':
        Mean_no_in_Feb_March_April_2018[i] = [city_data[(city_data.Date.dt.month == 2) & 
                                                        (city_data.Date.dt.year == 2018) & 
                                                        (city_data.City == i)]['NO'].mean(),
                                              city_data[(city_data.Date.dt.month == 3) & 
                                                        (city_data.Date.dt.year == 2020) & 
                                                        (city_data.City == i)]['NO'].mean(),
                                              city_data[(city_data.Date.dt.month == 4) & 
                                                        (city_data.Date.dt.year == 2018) & 
                                                        (city_data.City == i)]['NO'].mean()]


# In[ ]:


Mean_no_in_Feb_March_April_2018.transpose().plot(figsize=(10,10), kind='bar')


# As far as Motropolitan cities like Mumbai, Bengaluru, Delhi, Chandigad and Hydrabad are considered, we can see same trend in nearly all these cities by February, March and April. Their levels are dropping and these trends are only in the year 2020.  
# If we see the year 2019 or 2018, we cannot find this dropping trend in these cities. So we can say that Air Quality Index (AQI) has decreased due to lockdown.

# In[ ]:




