#!/usr/bin/env python
# coding: utf-8

# Information of Dataset
# For more information, read [Cortez and Morais, 2007]. 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 3. month - month of the year: 'jan' to 'dec' 4. day - day of the week: 'mon' to 'sun' 5. FFMC - FFMC index from the FWI system: 18.7 to 96.20 6. DMC - DMC index from the FWI system: 1.1 to 291.3 7. DC - DC index from the FWI system: 7.9 to 860.6 8. ISI - ISI index from the FWI system: 0.0 to 56.10 9. temp - temperature in Celsius degrees: 2.2 to 33.30 10. RH - relative humidity in %: 15.0 to 100 11. wind - wind speed in km/h: 0.40 to 9.40 12. rain - outside rain in mm/m2 : 0.0 to 6.4 13. area - the burned area of the forest (in ha): 0.00 to 1090.84 (this output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/forestfires.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


# Histogram
# bins = number of bar in figure
data.rain.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


series = data['ISI']        # data['Defense'] = series
print(type(series))
data_frame = data[['ISI']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['rain']>1     # There are only 3 pokemons who have higher defense value than 200
data[x]


# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['rain']<1) & (data['temp']>30)]


# In[ ]:


data.head()


# In[ ]:


# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()


# EXPLORATORY DATA ANALYSIS
# value_counts(): Frequency counts
# 
# * Lets say value at 75% is Q3 and value at 25% is Q1. 
# * Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR
# We will use describe() method. Describe method includes:
# * count: number of entries
# * mean: average of entries
# * std: standart deviation
# * min: minimum entry
# * 25%: first quantile
# * 50%: median or second quantile
# * 75%: third quantile
# * max: maximum entry
# 
# 
# 
# * 1,4,5,6,8,9,11,12,13,14,15,16,17
# * The median is the number that is in **middle** of the sequence. In this case it would be 11.

# In[ ]:


#For example lets look frequency of months types
print(data['month'].value_counts(dropna =False))  # if there are nan values that also be counted


# In[ ]:


#for example max temp 33.3 min wind 0.4
data.describe() #ignore null entries


# In[ ]:


# Plotting all data 
data1 = data.loc[:,["FFMC","DMC","DC"]]
data1.plot()
# it is confusing


# INDEXING PANDAS TIME SERIES
# * datetime = object
# * parse_dates(boolean): Transform date to ISO 8601 (yyyy-mm-dd hh:mm:ss ) format

# In[ ]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 


# In[ ]:


# Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# INDEXING DATA FRAMES
# * Indexing using square brackets
# * Using column attribute and row label
# * Using loc accessor
# * Selecting only some columns

# In[ ]:


# indexing using square brackets
data["temp"][1]


# In[ ]:


# using loc accessor
data.loc[1,["temp"]]


# In[ ]:


# Selecting only some columns
data[["temp","rain"]]


# SLICING DATA FRAME
# * Difference between selecting columns
#  * Series and data frames
# * Slicing and indexing series
# * Reverse slicing 
# * From something to end

# In[ ]:


# Slicing and indexing series
data.loc[1:10,"temp":"rain"]   # 10 and "Defense" are inclusive


# In[ ]:


# From something to end
data.loc[1:10,"ISI":] 


# FILTERING DATA FRAMES

# In[ ]:


# Creating boolean series
boolean = data.temp > 30
data[boolean]


# In[ ]:


# Combining filters
first_filter = data.temp > 30
second_filter = data.ISI > 10
data[first_filter & second_filter]


# INDEX OBJECTS AND LABELED DATA

# In[ ]:


# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()


# HIERARCHICAL INDEXING

# In[ ]:


# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["month","temp"]) 
data1.head(100)


# In[ ]:




