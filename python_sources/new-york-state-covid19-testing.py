#!/usr/bin/env python
# coding: utf-8

# At this moment, New York State is a global center (unfortunately) in the COVID-19 pandemic. COVID-19 testing was deployed late in the US vs. elsewhere. However, to its credit, the NY State Department of Health is now providing good and timely data on COVID-19 testing. I'll examine some of it here, and also shows how the NY State data compares with data from the rest of the world. Feel free to make constructive suggestions on how to improve on this. (I can't figure out how to take the up to date data automatically from the NY State site. I tried but Kaggle would only read the first 1000 rows automatically, so I had to load it  manually. So if you know how to do that, I'd appreciate it--please put it in the comments below.)

# In[ ]:


#some standard imports
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import matplotlib.dates as mdates
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime


# In[ ]:


#what are the data files?
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# this is the NY county-by-county testing data from NY State
ny = pd.read_csv('/kaggle/input/New_York_State_Statewide_COVID-19_Testing.csv', delimiter=',')
ny.dataframeName = 'New_York_State_Statewide_COVID-19_Testing.csv'
# Fix format of Test Date column
ny['Test Date']= pd.to_datetime(ny['Test Date']) 
nRow, nCol = ny.shape
print(f'There are {nRow} rows and {nCol} columns')
ny.head(10)


# In[ ]:


#I also found some county-by-county population data.  It's based on census data, with estimates since the last census
pops = pd.read_csv('/kaggle/input/Annual_Population_Estimates_for_New_York_State_and_Counties__Beginning_1970.csv', delimiter=',')
pops.head(5)


# In[ ]:


#Delete NY State (aggregate) from the population data. For some reason they included this alongside all the counties
pops=pops[pops.Geography != 'New York State']
#Keep only the 2019 populations
pops_2019 = pops[pops['Year']==2019]
#take the top 10 counties by population
top_counties=pops_2019.nlargest(10,'Population').sort_values('Population', ascending=False)
#Remove the word 'County' so we can join to the test data set, which doesn't have this word
top_counties['Geography'] = top_counties['Geography'].str.replace(r' County', '')
top_counties_list = top_counties['Geography'].to_list()
#Change name of column for consistency, and drop some other fields
top_counties.rename(columns = {'Geography':'County'}, inplace = True)
top_counties.drop(['FIPS Code', 'Year', 'Program Type'], axis=1, inplace=True)
#This is the list of top counties by population
top_counties


# In[ ]:


#calculate the percents of tests performed, and positive results, based on the population of each county
#it's possible some people had multiple tests, but ignore that
df = ny[ny.County.isin(top_counties_list)]
df1 = pd.merge(df,top_counties,on = 'County')
df1['Cum Percent Tested']=100*df1['Cumulative Number of Tests Performed']/df1['Population']
df1['Cum Percent Positives']=100*df1['Cumulative Number of Positives']/df1['Population']
df1


# In[ ]:


graph = df1.pivot(index='Test Date', columns='County', values='Cumulative Number of Tests Performed')
graph


# In[ ]:


top_counties_list = graph.columns.to_list()

plt.figure(figsize=(12,12))
plt.plot(graph)
plt.title('Cumulative Number of Tests Performed')
plt.xlabel('Date')
plt.ylabel('Cumulative Tests')
plt.legend(top_counties_list)
plt.show()


# Westchester has been hard hit, but has been testing aggressively. Queens, Brooklyn (Kings County) and Nassua are moving up the charts. 

# In[ ]:


graph=df1.pivot(index='Test Date', columns='County', values='Cum Percent Positives')

top_counties_list = graph.columns.to_list()

plt.figure(figsize=(12,12))
plt.plot(graph)
plt.title('Cumulative Positive Tests as a Percent of Population')
plt.xlabel('Date')
plt.ylabel('Cumulative Tests')
plt.legend(top_counties_list)
plt.show()


# Westchester also leads in terms of percent of the population testing positive--closing in on 3%. 

# In[ ]:


graph = df1.pivot(index='Test Date', columns='County', values='Cum Percent Tested')

top_counties_list = graph.columns.to_list()

plt.figure(figsize=(12,12))
plt.plot(graph)
plt.title('Cumulative Percent of Population Tested')
plt.xlabel('Date')
plt.ylabel('Cumulative Percent Tested')
plt.legend(top_counties_list)
plt.show()


# Westchester has tested over 8% of the population. We'll compare that later to the US (overall) and other countries. 

# In[ ]:


# read in the worldwide data
world = pd.read_csv('/kaggle/input/full-list-cumulative-total-tests-per-thousand.csv', delimiter=',')
# Fix format of Test Date column
world['Date']= pd.to_datetime(world['Date']) 
nRow, nCol = world.shape
print(f'There are {nRow} rows and {nCol} columns')
world


# In[ ]:


#I'm going to ignore entries without test data
world = world.dropna()
#let's look at a subset of countries
country_list=['South Korea', 'United States', 'Canada', 'Italy', 'Australia','New Zealand']
c=world[world['Entity'].isin(country_list)]
c['Cum Percent Tested']=c['Total tests per thousand']/10.0
#sort by country
c.Entity = c.Entity.astype("category")
c.Entity.cat.set_categories(country_list, inplace=True)
#pd.set_option('display.max_rows', None)
c


# In[ ]:


graph = c.pivot(index='Date', columns='Entity', values='Cum Percent Tested')
plt.figure(figsize=(12,12))
plt.plot(graph)
plt.title('Cumulative Percent of Population Tested')
plt.xlabel('Date')
plt.ylabel('Cumulative Percent Tested')
plt.legend(country_list)
plt.show()


# The above data is from ourworldindata.org, which is a non-profit based out of Oxford,supported by the Bill and Melinda Gates Foundation. As you can see, the US is still behind some countries like Italy, Australia, New Zealand, and Canada, but has made improvements in testing capabilities. Korea used to be a leader but some others have surpassed it, but likely because those countries are in such bad shape and need to conduct more tests than Korea (e.g., Italy). On the whole, though, the hard-hit counties in New York state are well ahead of the US, on a national basis (e.g., Westchester has tested over 8% of its population, while the US has tested under 1.5%). So a concern is that the rest of the US doesn't really know where it stands, having done relatively little testing compared with NY. While they may have had fewer cases, to some degree that's due to lack of testing. 
# 

# In[ ]:


# read in the worldwide data
tests_per_thous = pd.read_csv('/kaggle/input/daily-covid-19-tests-per-thousand-rolling-3-day-average.csv', delimiter=',')
#I'm going to ignore entries without test data
#world = world.dropna()
# Fix format of Test Date column
tests_per_thous['Date']= pd.to_datetime(tests_per_thous['Date']) 
tests_per_thous


# In[ ]:


country_list=['South Korea', 'United States', 'Canada', 'Italy', 'Australia','New Zealand']
c=tests_per_thous[(tests_per_thous['Entity'].isin(country_list)) & (tests_per_thous['Date']>pd.Timestamp(2020, 4, 1)) ]
#sort by country
c.Entity = c.Entity.astype("category")
c.Entity.cat.set_categories(country_list, inplace=True)
c = c.reset_index()
del c['index']
c


# In[ ]:


graph = c.pivot(index='Date', columns='Entity', values='3-day rolling mean of daily change in total tests per thousand')
plt.figure(figsize=(12,12))
plt.plot(graph)
plt.title('Daily Tests Performed Per Thousand Population')
plt.xlabel('Date')
plt.ylabel('3-day rolling mean of daily change in total tests per thousand')
plt.legend(country_list)
plt.show()


# So in the US we're doing about 0.8 tests per thousand people in the population right now. Acccording to some estimates, we need to be doing 5 millions tests per day to partially open the economy (that's about 15 tests per thousand in the US population of 328 million). And some estimates require 30 million tests per day in the US in order to fully open the economy. See this: https://abcnews.go.com/US/road-map-recovery-report-20-million-coronavirus-tests/story?id=70230097
