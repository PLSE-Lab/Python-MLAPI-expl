#!/usr/bin/env python
# coding: utf-8

# Information of Dataset
# For more information, read [Cortez and Morais, 2007]. 
# 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 
# 2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 
# 3. month - month of the year: 'jan' to 'dec' 
# 4. day - day of the week: 'mon' to 'sun' 
# 5. FFMC - FFMC index from the FWI system: 18.7 to 96.20 
# 6. DMC - DMC index from the FWI system: 1.1 to 291.3 
# 7. DC - DC index from the FWI system: 7.9 to 860.6 
# 8. ISI - ISI index from the FWI system: 0.0 to 56.10 
# 9. temp - temperature in Celsius degrees: 2.2 to 33.30 
# 10. RH - relative humidity in %: 15.0 to 100 
# 11. wind - wind speed in km/h: 0.40 to 9.40 
# 12. rain - outside rain in mm/m2 : 0.0 to 6.4 
# 13. area - the burned area of the forest (in ha): 0.00 to 1090.84 (this output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform).

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


data.head(20) #show to only 20 rows


# In[ ]:


data.describe(include='all')


# In[ ]:


data.columns #show to features


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


data.RH.plot(kind = 'line', color = 'g',label = 'RH',figsize = (7,7),linewidth=1,alpha = 0.9,grid = True,linestyle = ':') 
data.temp.plot(color = 'r',label = 'temp',linewidth=1,figsize = (6,6), alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Days that taken measurement')              # label = name of label
plt.ylabel('Frequency')
plt.title('RH-temp Line Plot')            # title = title of plot
plt.show()


# In[ ]:


data.plot(subplots = True, figsize = (16,16))
plt.xlabel('Days that taken measurement') 
plt.ylabel('Frequencies')
plt.show()


# In[ ]:


data.wind.plot(kind = 'hist',bins = 50,figsize = (12,12),range=(0,8),grid = True,normed = True)  #bins defines number of bar; normed=normalization
plt.show()


# In[ ]:


data.wind.plot(kind = 'hist',bins = 10000,figsize = (12,12),range=(0,8),normed = True,cumulative = True)  
plt.show()


# In[ ]:


x = data['temp'] >= 30
data[x]


# In[ ]:


series = data['ISI']        # data['Defense'] = series
print(type(series))
data_frame = data[['ISI']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['rain']>=0.5     # There are only 3 pokemons who have higher defense value than 200
data[x]


# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['rain']>=0) & (data['temp'] >= 30)]


# In[ ]:


data.wind[data.temp>=30]


# In[ ]:


data.shape


# In[ ]:


print(data['ISI'].value_counts(dropna =False))    #dropna=False = to show Nan 


# In[ ]:


Mean_temp = sum(data.temp)/len(data.temp)
data["temp_level"] = ["high" if i > Mean_temp else "low" for i in data.temp]
data.loc[:7,["temp_level","temp"]]


# In[ ]:


data.boxplot(column='temp')


# In[ ]:


data_new = data.head()   
data_new
melted = pd.melt(frame=data_new,id_vars = 'month', value_vars= ['FFMC','DC'])
melted


# In[ ]:


data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# In[ ]:


data1 = data['month'].head()
data2= data['temp'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # combine to columns
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


data['RH '] = data['RH'].astype('float64')


# In[ ]:


data.dtypes


# INDEXING PANDAS TIME SERIES
# * datetime = object
# * parse_dates(boolean): Transform date to ISO 8601 (yyyy-mm-dd hh:mm:ss ) format

# In[ ]:


data.info()


# In[ ]:


data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2


# In[ ]:


data2.resample("A").mean()  #A : year M: mount


# In[ ]:


data2.resample("M").mean()  #A : year M: mount 


# In[ ]:


data2.resample("M").first().interpolate("linear")


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# In[ ]:


def div(n):
    return n**2
data2.temp.apply(div)


# In[ ]:


data1 = data.set_index(["temp","wind"]) 
data1.head(100)


# In[ ]:


data.groupby("rain").mean()


# In[ ]:





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
data.loc[1:10,"temp":"rain"]   


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
data.head()


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

