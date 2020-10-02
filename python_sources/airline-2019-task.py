#!/usr/bin/env python
# coding: utf-8

# ****Task Details****
# 1. Combine different csv files into a single dataframe
# 2. Clean the city_name columns, which also contain the abreviated state names.
# 3. Check which of the columns are redundant information (i.e. they can easily be computed from the other columns)
# 4. Find out the airports and the flight operators which correspond to maximum delay in general.

# # Imports

# In[ ]:


import os
import pandas as pd


# ### Gather Data

# In[ ]:


list_of_files = []
for (dirpath, dirnames, filenames) in os.walk('../input/airline-2019'):
    for f in filenames:
        full_path = os.path.abspath(os.path.join(dirpath, f))
        list_of_files.append(full_path)
        
num_files = len(list_of_files)

#print the number of files gathered
print(f'{num_files} files in airline-2019 directory')

#print out the absolute path to every file gathered
for file in list_of_files:
    print(file)


# # Task 1: Combine different csv files into a single dataframe

# In[ ]:


# read one dataframe
df1 = pd.read_csv(list_of_files[0])
df1


# In[ ]:


#create an empty list, which we will fill up with dataframes
df_list = []

for file in list_of_files:
    df = pd.read_csv(file)
    df_list.append(df)

    
#concatenate all dfs in the list into one dataframe
airline_2019_df = pd.concat(df_list)


# In[ ]:


airline_2019_df


# In[ ]:


#sort airline dataframe by data
airline_2019_df = airline_2019_df.sort_values(by=['FL_DATE'])


# # Task 2: Clean the city_name columns, which also contain the abreviated state names.
# 
# Currently the ORIGIN_CITY_NAME and DEST_CITY_NAME columns shows the city and state abbreviation. The state abbreviation is redundant becuase there is a column named ORIGIN_STATE_NM and DEST_CITY_NAME

# In[ ]:


#update the ORIGIN_CITY_NAME column
origin_cities = airline_2019_df['ORIGIN_CITY_NAME'].tolist()


# In[ ]:


true_cities = []
for city in origin_cities:
    city = city.split(',')[0]
    true_cities.append(city)


# In[ ]:


airline_2019_df['ORIGIN_CITY_NAME'] = true_cities


# In[ ]:


airline_2019_df


# In[ ]:


#update destination cities
dest_cities = airline_2019_df['DEST_CITY_NAME'].tolist()

true_dest = []
for city in dest_cities:
    city = city.split(',')[0]
    true_dest.append(city)

airline_2019_df['DEST_CITY_NAME'] = true_dest


# In[ ]:


airline_2019_df


# # Task 3: Check which of the columns are redundant information (i.e. they can easily be computed from the other columns)

# In[ ]:


airline_2019_df.columns


# In[ ]:


airline_2019_df.head()


# In[ ]:


#what is unamed.. are they all NaN..
print(airline_2019_df.isnull().sum())


# In[ ]:


print(len(airline_2019_df)) #Unamed 25 is full of NaN


# ### Investigate cancelled and cancellation code

# In[ ]:


airline_2019_df.loc[airline_2019_df['CANCELLED'] != 0]


# ### It seems as if canceled is redundant... if cancelled, then the cancellation code will have some value

# In[ ]:


#it seems as if canceled is redundant
canceled = airline_2019_df.loc[airline_2019_df['CANCELLED'] != 0]['CANCELLED'].tolist()
cancel_code = airline_2019_df.loc[airline_2019_df['CANCELLED'] != 0]['CANCELLATION_CODE'].tolist()
print('length of canceled list: ', len(canceled))
print('length of canceled list: ', len(cancel_code))


# In[ ]:


canceled_set = set()
cancel_codes = set()
for i, j in zip(canceled, cancel_code):
    if i not in canceled_set:
        canceled_set.add(i)
    if j not in cancel_codes:
        cancel_codes.add(j)
        
print('types of cancellations: ', canceled_set)
print('types of cancellation codes: ', cancel_codes)


# ## The canceled column is redundant, if a flight is cancelled it will be given a code of A, B, C, or D
# 
# ## We also already cleaned the city columns to take out the states, so states column is no longer redundant

# In[ ]:


# delete cancelled column
airline_2019_df = airline_2019_df.drop(columns=['CANCELLED', 'Unnamed: 25'])


# In[ ]:


airline_2019_df


# # Task 4: Find out the airports and the flight operators which correspond to maximum delay in general.
# 
# Which origin airports correspond to greater delays.. and which destination airports correspond to greater delays?
# 

# The following is the number of missing values in all of the delay columns
# * CARRIER_DELAY            6044570
# * WEATHER_DELAY            6044570
# * NAS_DELAY                6044570
# * SECURITY_DELAY           6044570
# * LATE_AIRCRAFT_DELAY      6044570
# 
# Because they all have the same amount of missing values, I'm going to assume that if one delay column has NaN, 
# they all have NaN

# In[ ]:


airline_2019_df


# Our goal is to find: 
# 
# * Which origin airports correspond with the greatest number of delays
# * Which origin airports correspond with the greatest average delay times
# * Which airlines correspond with the greatest number of delays
# * Which airlines correspond with the greatest average delay times
# 

# In[ ]:


#only work with flights that have values in the delay columns
delayed_flights = airline_2019_df.dropna(subset=['NAS_DELAY'])


# In[ ]:


delayed_flights


# In[ ]:


#was getting SettingWithCopyWarning
delayed_flights = delayed_flights.copy()
delayed_flights


# In[ ]:


delayed_flights['total_delay'] = delayed_flights['CARRIER_DELAY'] + delayed_flights['WEATHER_DELAY'] +delayed_flights['NAS_DELAY'] + delayed_flights['SECURITY_DELAY'] + delayed_flights['LATE_AIRCRAFT_DELAY']


# In[ ]:


print('total delay per airport')
total_delay_org = delayed_flights.groupby('ORIGIN').total_delay.sum().reset_index()
total_delay_org.sort_values('total_delay', ascending = False)


# In[ ]:


#num delayed.. can someone help me find the probability of delay
print('number of delayed flights per airport')
number_of_delays_org = delayed_flights.groupby('ORIGIN').total_delay.count().reset_index()
number_of_delays_org.sort_values('total_delay', ascending = False)


# In[ ]:


#average delay... 

#fillnas with 0s...
airline_2019_df = airline_2019_df.fillna(0)

#calculate total
airline_2019_df['total_delay'] = airline_2019_df['CARRIER_DELAY'] + airline_2019_df['WEATHER_DELAY'] +airline_2019_df['NAS_DELAY'] + airline_2019_df['SECURITY_DELAY'] + airline_2019_df['LATE_AIRCRAFT_DELAY']


# In[ ]:


airline_2019_df.head()


# In[ ]:


print('Average delay by airport')
avg_delay = airline_2019_df.groupby('ORIGIN').total_delay.mean().reset_index()
avg_delay = avg_delay.rename(columns={"total_delay": "avg_delay"})
avg_delay.sort_values('avg_delay', ascending = False)


# In[ ]:


#number of delays per airline
print('total delay time by airline')
airline_delay = airline_2019_df.groupby('OP_CARRIER_AIRLINE_ID').total_delay.count().reset_index()
airline_delay.sort_values('total_delay', ascending = False)


# In[ ]:


#average delay time per airline
print('average delay time by airline')
airline_avg_delay = airline_2019_df.groupby('OP_CARRIER_AIRLINE_ID').total_delay.mean().reset_index()
airline_avg_delay = airline_avg_delay.rename(columns={'total_delay':'avg_delay'})
airline_avg_delay.sort_values('avg_delay', ascending = False)


# In[ ]:


#probability of delay..


# #### So I have found the number of delays per airport, but can someone help me find the probability of delay.. 
# P = (number of delays per airport)/(number of flights from airport) 

# # Questions, Comments, Suggestions?
