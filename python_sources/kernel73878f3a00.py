#!/usr/bin/env python
# coding: utf-8

# # Project - Exploring Gun Deaths in USA

# ## 1. Processing of guns.csv dataset

# ### Data import

# In[ ]:


import csv
f = open("/kaggle/input/gun-deaths-in-the-us/guns.csv", "r")
data = list(csv.reader(f))


# ### Understanding dataset

# - Header:

# In[ ]:


headers = data[:1]
print(headers)


# - Explore first 5 rows (without header):

# In[ ]:


data = data[1:]
print(data[:6])


# - Explore sets of values in columns:

# In[ ]:


# year column  -- the year in which the fatality occurred.
years = [item[1] for item in data]
set(years)


# In[ ]:


# month column -- the month in which the fatality occurred.
month = [item[2] for item in data]
set(month)


# In[ ]:


# intent column -- the intent of the perpetrator of the crime.
intents = [item[3] for item in data]
set(intents)


# In[ ]:


# police column -- whether a police officer was involved with the shooting.
police = [item[4] for item in data]
set(police)


# In[ ]:


# sex column -- the gender of the victim.
sex = [item[5] for item in data]
set(sex)


# In[ ]:


# age column -- the age of the victim.
age = [item[6] for item in data]
print(set(age))


# In[ ]:


# race column -- the race of the victim.
races = [item[7] for item in data]
set(races)


# In[ ]:


# hispanic column  -- a code indicating the Hispanic origin of the victim.
hispanic = [item[8] for item in data]
print(set(hispanic))


# In[ ]:


# place column -- where the shooting occurred.
place = [item[9] for item in data]
set(place)


# In[ ]:


# education column -- educational status of the victim. 
education = [item[10] for item in data]
set(education)


# Education column values description:
# <br>1 -- Less than High School
# <br>2 -- Graduated from High School or equivalent
# <br>3 -- Some College
# <br>4 -- At least graduated from College
# <br>5 -- Not available

# ### Explore number of deaths by year

# In[ ]:


year_counts = {}
for item in years:
    if item in year_counts:
        year_counts[item] += 1
    else:
        year_counts[item] = 1
year_counts


# ### Explore number of deaths by month and year

# - Transformation of input dataset to datetime objects

# In[ ]:


import datetime
dates_objects = [datetime.datetime(year=int(item[1]), month=int(item[2]), day=1) for item in data]
dates_objects[:6]


# - Calculation of deaths by unique dates

# In[ ]:


date_counts = {}
for item in dates_objects:
    if item in date_counts:
        date_counts[item] += 1
    else:
        date_counts[item] = 1
date_counts        


# ### Explore correlation between deaths, sex and race
# 

# - Deaths by sex

# In[ ]:


sex_counts = {}
for item in data:
    if item[5] in sex_counts:
        sex_counts[item[5]] += 1
    else:
        sex_counts[item[5]] = 1
        
sex_counts


# - Deaths by race

# In[ ]:


race_counts = {}
for item in data:
    if item[7] in race_counts:
        race_counts[item[7]] += 1
    else:
        race_counts[item[7]] = 1
        
race_counts


# ### Intermediate findings

# - Number of deathes approximately equally for 2012, 2013 and 2014 years
# - In terms of seasons the number of deaths increased slightly by the end of the year
# - The number of deaths of men exceeds the number of deaths of women by 6 times
# - In terms of races the highest death rates correspond to White and Black races

# ## 2. Addition of census.csv dataset

# ### Data import

# In[ ]:


file = open("/kaggle/input/census/census.csv", "r")
census = list(csv.reader(file))
print(census)


# ### Understanding dataset

# - Header:

# In[ ]:


census_header = census[0]
census_header


# - Explore rows (without header):

# In[ ]:


census_data = census[1:]
census_data


# ### Calculation of weighted deaths count rate per race category

# In[ ]:


# Manual mapping of racial values between datasets
mapping = {
    "Asian/Pacific Islander": 15834141,
    "Black": 40250635,
    "Native American/Native Alaskan": 3739506,
    "Hispanic": 44618105,
    "White": 197318956
}


# In[ ]:


# Gun deaths per race by 100000 of people in a racial category
race_per_hundredk = {}
for item in race_counts:
    race_per_hundredk[item] = (race_counts[item] / mapping[item]) * 100000

race_per_hundredk


# ### Calculation of homicide rate per 100.000 people in each racial category

# In[ ]:


homicide_race_counts = {}


# In[ ]:


for key,value in enumerate(races):
    if value not in homicide_race_counts:
        homicide_race_counts[value] = 0
    if intents[key] == "Homicide":
        homicide_race_counts[value] += 1


# In[ ]:


print(homicide_race_counts)


# In[ ]:


homicide_race_per_hundredk = {}
for item,value in homicide_race_counts.items():
    homicide_race_per_hundredk[item] = (value / mapping[item])*100000
    
homicide_race_per_hundredk   


# ### Final Findings:
# - Racial categories that are most affected by gun related episodes are: Black and Hispanic

# In[ ]:




