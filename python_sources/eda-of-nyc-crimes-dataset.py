#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis of the NYC Crimes Dataset
# 
# ### EDA assigment in ECE-475 Frequentist Machine Learning at the Cooper Union
# 
# ### Due: 09/13/18
# 
# ### By: Guy Bar Yosef

# In[ ]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import collections


# For this project I picked the 'New York City Crimes' dataset from Kaggle. First lets look at the three csv files that come with the dataset:

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


crime_data = pd.read_csv('../input/Crime_Column_Description.csv')
population_data = pd.read_csv('../input/Population_by_Borough_NYC.csv')
complaint_data = pd.read_csv('../input/NYPD_Complaint_Data_Historic.csv', dtype=object)


# Before exporing any of the crime statistics, lets look at what is inside the NYC population file:

# In[ ]:


population_data.head(6)


# We see the data is seperated into boroughs and their populations (excluding Staten Island apperantly),  from the year 1950 and up to the present, jumping by decades. It also gives projections for the population of each burough into the future, continuing to the years 2020, 2030, and 2040. Besides the population at each decade, there is the precent share of the borough's population of NYC's total.
# 
# First let us see how the precent share of each borough changes over the decades through a 'Precent Stacked barplot', as seen in Olivier Gaudard's example at https://python-graph-gallery.com/13-percent-stacked-barplot/

# In[ ]:


precent_columns = ['1950 - Boro share of NYC total', '1960 - Boro share of NYC total', '1970 - Boro share of NYC total', 
                   '1980 - Boro share of NYC total', '1990 - Boro share of NYC total', '2000 - Boro share of NYC total', 
                   '2010 - Boro share of NYC total', '2020 - Boro share of NYC total', '2030 - Boro share of NYC total', 
                   '2040 - Boro share of NYC total']
years = ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020', '2030', '2040']

bronx_precent = []
brooklyn_precent = []
manhattan_precent = []
queens_precent = []
staten_precent = []

for col in precent_columns:
    bronx_precent.append(float(population_data[col][1][:-1]))
    brooklyn_precent.append(float(population_data[col][2][:-1]))  
    manhattan_precent.append(float(population_data[col][3][:-1]))
    queens_precent.append(float(population_data[col][4][:-1]))
    staten_precent.append(float(population_data[col][5][:-1]))


plt.figure(figsize=[15, 7])
plt.bar(years, bronx_precent, label="Bronx")
plt.bar(years, brooklyn_precent, bottom=bronx_precent, label="Brooklyn")
plt.bar(years, manhattan_precent, bottom = [i+j for i,j in zip(bronx_precent, brooklyn_precent)], label="Manhattan")
plt.bar(years, queens_precent, bottom = [i+j+k for i,j,k in zip(bronx_precent, brooklyn_precent, manhattan_precent)], label="Queens")
plt.bar(years, staten_precent, bottom = [i+j+k+n for i,j,k, n in zip(bronx_precent, brooklyn_precent, manhattan_precent, queens_precent)], label="Staten Island")

plt.xlabel('Years', size=15)
plt.ylabel('Precentage of NYC population', size=15)
plt.legend(bbox_to_anchor=(1,0.6), prop={'size': 15})
plt.title('Precent Share of Population for each New York City Borough', size=18)

plt.show()


# Not sure that was worth the effort, but at least I learned something..
# 
# Anyhow, we can briefly see the total population of NYC increasing:

# In[ ]:


total_pop = [int(population_data[i][0].replace(',', '')) for i in years]

plt.figure(figsize=[15,7])
plt.bar(years, total_pop)

plt.xlabel('Years', size=15)
plt.ylabel('Population', size=15)
plt.title('Total Population of New York City', size=18)

plt.show()


# Lets move on to the main part of the dataset, the actual crime statistics in NYC. First of all, there is the file with the column descriptions:

# In[ ]:


print(crime_data.to_string())


# There is clearly a lot of data here. Lets explore!
# 
# First start off simple, with seeing which 'digit offense classification code' was the most common:

# In[ ]:


dig_offense_code = {}
code_meaning = {}

dig_offense_code = complaint_data['KY_CD'].value_counts().to_dict()
code_meaning = complaint_data['OFNS_DESC'].value_counts().to_dict()

plt.figure(figsize=[15,7])
plt.bar(list(dig_offense_code.keys()), list(dig_offense_code.values()))
plt.xlabel('3 digit offense classification code', size=14)
plt.ylabel('Number of Occurances', size=14)
plt.title('Which Offense Occured the Most?', size=16)

plt.show()


# Clearly, there are a lot of offence codes... Lets only look at the most ferquent ones, lets say, those that occured over 25,000 times:

# In[ ]:


updated_dig_offense = {key: value for key, value in dig_offense_code.items() if value > 25000 }

        
plt.figure(figsize=[15,7])
plt.bar(list(updated_dig_offense.keys()), list(updated_dig_offense.values()))
plt.xlabel('3 digit offense classification code', size=14)
plt.ylabel('Number of Occurances', size=14)
plt.title('Which Offense Occured the Most?', size=16)

plt.show()


# By the by, these codes reffer to:

# In[ ]:


for key, meaning in zip(updated_dig_offense.keys(), code_meaning.keys()):
    print(key, ':', meaning)


# Let us now look at the most common days, months, and years for crime:

# In[ ]:


from collections import defaultdict

months_by_count = {i+1: 0 for i in range(12)}
days_by_count = {i+1: 0 for i in range(31)}
years_by_count = defaultdict(int)
day_of_year = {}

for i in complaint_data['CMPLNT_FR_DT']:
    if ( isinstance(i ,str) ):
        dates = i.split('/')
        months_by_count[int(dates[0])] += 1
        days_by_count[int(dates[1])] += 1
        years_by_count[int(dates[2])] += 1
        
        day_of_year[dates[0] + '/' + dates[1] ] = day_of_year.get(dates[0] + '/' + dates[1] ,0) + 1


# In[ ]:


plt.figure(figsize=[12,15])
plt.subplots_adjust(hspace=0.5)

plt.subplot(3,1,1)
plt.title('Crime by Days')
plt.xlabel('Days')
plt.ylabel('Amount of Crime Incidents')
plt.bar(days_by_count.keys(), days_by_count.values())

plt.subplot(3,1,2)
plt.title('Crime by Months')
plt.xlabel('Months')
plt.ylabel('Amount of Crime Incidents')
plt.bar(months_by_count.keys(), months_by_count.values())

plt.subplot(3,1,3)
plt.title('Crime by Years')
plt.xlabel('Years')
plt.ylabel('Amount of Crime Incidents')
plt.plot(years_by_count.keys(), years_by_count.values())

plt.show()


# Well the years graph above just shows that most of the data is very recent, with a few (maybe large and significant?) crimes from past years included. 
# 
# Perhaps more intresting are the most dangerous days of the year:

# In[ ]:


plt.figure(figsize=[10,7])

day_of_year = dict(collections.Counter(day_of_year).most_common(10))
plt.bar(day_of_year.keys(), day_of_year.values())
plt.gca().set_ylim(ymin=3500)
plt.ylabel('Amount of Crimes')
plt.xlabel('Days to Avoid!')
plt.title('Days with Most Reported Crimes')
plt.show()


# New Years Day makes sense, and is generally common knowledge. For the rest, I guess it is good to know that November and December are dangerous months, as was clear from the months graph a little above.
# 
# Thats it for now!
