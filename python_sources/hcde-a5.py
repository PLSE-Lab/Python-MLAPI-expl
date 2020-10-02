#!/usr/bin/env python
# coding: utf-8

# For this homework assignment, I explore home values provided by Zillow: https://www.zillow.com/research/data/
# 
# This is of particular interest to me as I hope to invest in property in the future. This datesets has home values across the U.S. from 1996 to the present down to the granular level of Metro area. I would like to explore the trends of home values across time in the US as well as Washington state. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting library for vis

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First thing I did was inport the data and look to see how it is formatted. 

# In[ ]:


#create a dataframe of imported zillow home values
homes = pd.read_csv("../input/zillow-home-value-index-zhvi/City_Zhvi_AllHomes.csv", encoding='latin-1')
homes.head()


# Since the columns run long, I thought it would be best to melt all the Date data to indicate a singular column with a new column to house the corresponding home value. 

# In[ ]:


#create new data frame of melted years columns with home values into rows 
homes2 = pd.melt(homes, id_vars=["RegionID", "RegionName", "State", "Metro", "CountyName", "SizeRank"], var_name="Date", value_name="Value")
homes2['Date']= pd.to_datetime(homes2['Date']) #change dates to date type


# In[ ]:


homes2.head()


# By taking the aggregated sum of all home values for each year, we can plot the change over time for the entire nation

# In[ ]:


#create and plot dataframe the sums of all home values for each year
national_sum = homes2.groupby('Date')['Value'].agg(SumValue=('Value', sum)).reset_index() 
plt.figure(figsize=(15,10))
plt.plot('Date', 'SumValue', data=national_sum)
plt.xlabel('Years')
plt.ylabel('Zillow Home Value')
plt.title('Zillow Home Values Across National over the Years')
plt.show()


# By only aggregating based on State, we can compare the trends on the state level on the same graph.

# In[ ]:


#create and plot dataframe the sums home values for each state for each year in a single graph
state_sum = homes2.groupby(['State', 'Date'])['Value'].agg(SumValue=('Value', sum)).reset_index()
states = np.unique(state_sum['State']) #create a list of all states

plt.figure(figsize=(15,10))

#plot each values for each individual state
for state in states:
    display_state = state_sum[state_sum['State'] == state]
    plt.plot('Date', 'SumValue', data=display_state, alpha = 0.9, label=state)

plt.legend(loc=2, ncol=10)
plt.xlabel('Years')
plt.ylabel('Zillow Home Value')
plt.title('Zillow Home Values for each State over the Years')
plt.show()


# Since the last graph is hard to read, it is better to plot the graphs in small multiples for each state trend. 

# In[ ]:


#create and plot dataframe the sums home values for each state for each year in a single graph
state_sum = homes2.groupby(['State', 'Date'])['Value'].agg(SumValue=('Value', sum)).reset_index()
states = np.unique(state_sum['State']) #create a list of all states


#plot each values for each individual state
for state in states:
    display_state = state_sum[state_sum['State'] == state]
    plt.plot('Date', 'SumValue', data=display_state, alpha = 0.9, label=state)
    plt.xlabel('Years')
    plt.ylabel('Zillow Home Value')
    plt.title('Zillow Home Values for %s over the Years' %state)
    plt.show()



# Compare the home values across states in last report (dec 2019)

# In[ ]:


plt.figure(figsize=(15,10))

#create a bar graph with home value of last reported datapoint for each state
for state in states:
    display_state = state_sum[state_sum['State'] == state]
    y = display_state.tail(1)['SumValue'] #last reported data point
    plt.bar(state, y)
    plt.xlabel("State")
    plt.ylabel("Home Value")
    
plt.title('Zillow Home Values for each state in Dec 2019')
plt.show()


# Graph pie chart of home values in a single state per city for a single month

# In[ ]:


plt.figure(figsize=(15,10))

state = "WA" #save state as a variable to be easily changed
time = "2019-12-01" #date variable in to be easily changed 

washington = homes2.loc[homes2['State'] == state] #save washington state values in new dataframe
subset = washington.loc[washington['Date'] == time] #create a new dataframe for single date

percentage = subset['Value'].array #put home values into an array
city = subset['RegionName'].array #put city labels into an array

#replace all null values with 0
pos = 0
for x in percentage:
    if np.isnan(x) == True:
        percentage[pos] = 0
    pos = pos + 1

#plot pie chart
plt.pie(percentage, labels = city)
plt.show

