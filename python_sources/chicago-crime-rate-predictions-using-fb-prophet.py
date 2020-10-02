#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this notebook we will do time series analysis using Facebook Prophet library. Prophet is an open source software developed by the core data science team. It used in forcasting the time series data. We are going to use this library to forcast the crime rate in chicago in next 2 year.

# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet


# ## Importing data set
# 
# I am taking the data set from year 2005 to 2017. Since, data is too big, it is available in to 3 different files. So, first we will import the files in to 3 distinct dataframes and then we concatenate them in to single file.

# In[ ]:


df_1 = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv", error_bad_lines = False)
df_2 = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv", error_bad_lines = False)
df_3 = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv", error_bad_lines = False)


# In[ ]:


# concatenating the dataframes in to one 
df = pd.concat([df_1, df_2, df_3])


# ## Data preprocessing

# Let's look at first five row to get a broad understanding of data set using dataframe head function.

# In[ ]:


df.head()


# Now, let's see how many rows and columns we have in a data set.

# In[ ]:


df.shape


# As we can see in the above output, the dataframe have more than 6 million rows and 23 columns which is quite huge.

# Since, the data is huge we are pretty sure that there will be some NULL values in a columns. To check the NULL values, I am using the seaborn heatmap function to visualize the NULL values.

# In[ ]:


plt.figure(figsize = (10, 10)) 
sns.heatmap(df.isnull(), cbar = False, cmap = 'YlGnBu')


# The blue lines in the above figure are showing the NULL values. The NULL values are present in columns X Coordinates, Y Coordinates, Latitude, Longitude and Location.

# Now, our further case study will be on only 5 columns namely Date, Block, Primary Type, Description, Location Description, Arrest and Domestic.

# In[ ]:


df_new = df[['Date', 'Block', 'Primary Type', 'Description', 'Location Description', 'Arrest', 'Domestic']]
df_new.head()


# We have to convert Date column in to appropriate date time format for further time series analysis.

# In[ ]:


df_new.Date = pd.to_datetime(df_new.Date, format = '%m/%d/%Y %I:%M:%S %p')
df_new.head()


# ## Exploratory analysis and visualization

# Let's see the top 15 criminal activity happens in Chicago from 2015 to 2017.

# In[ ]:


df_new['Primary Type'].value_counts().iloc[:15]


# Theft is most most common criminal activity with more 1.2 million incidence happened between 2005 to 2017.

# Let's visualize the above information in a chart for an interactive view using seaborn countplot function.

# In[ ]:


plt.figure(figsize = (15, 10))
sns.countplot(y = 'Primary Type', data = df_new, order = df_new['Primary Type'].value_counts().iloc[:15].index)


# In[ ]:


plt.figure(figsize = (15, 10))
sns.countplot(y = 'Location Description', data = df_new, order = df_new['Location Description'].value_counts().iloc[:15].index)


# As we can see in above chart, majority of the crimes happened in Street followed by Residence.

# In[ ]:


df_new.index = pd.DatetimeIndex(df_new.Date)
df_new.head()


# Let's count the total number of crimes happened per year

# In[ ]:


df_new.resample('Y').size()


# In[ ]:


plt.plot(df_new.resample('Y').size())
plt.title('Crime count per year from 2012 to 2017')
plt.xlabel('Year')
plt.ylabel('Number of crimes')


# As we can see in above plot, the crime rate showing the decreasing trend.

# You can visualize the crime rate trend per month by just replacing the 'Y' in resample function with 'm' as seen in code below.

# In[ ]:


plt.plot(df_new.resample('m').size())
plt.title('Crime count per month from 2012 to 2017')
plt.xlabel('Month')
plt.ylabel('Number of crimes')


# ## Making predictions

# In[ ]:


# Reseting the index number of rows
chicago_prophet = df_new.resample('m').size().reset_index()
chicago_prophet


# In[ ]:


# changing the column names for clarity
chicago_prophet.columns = ['Date', 'Crime count']
chicago_prophet


# Now we have to change the column names in to 'ds' and 'y' because facebook prophet work only with columns whose names are 'ds' and 'y'. If you try to implement prophet without renaming the columns in to 'ds' and 'y', you will get an error.

# In[ ]:


chicago_prophet_df_final = chicago_prophet.rename(columns = {'Date': 'ds', 'Crime count': 'y'})
chicago_prophet_df_final


# Next, i am fitting the data using prophet.

# In[ ]:


m = Prophet()
m.fit(chicago_prophet_df_final)


# Now, we forcast the crime rate in next 2 years. This can be done by assigning 720 (365 days * 2) to periods as shown in code below.

# In[ ]:


future = m.make_future_dataframe(periods = 720)
forecast = m.predict(future)
forecast


# In[ ]:


figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Crime rate')


# In[ ]:


figure = m.plot_components(forecast)


# As we can see in above plots the crime rate is showing the decreasing trend over the years and it will countinue also in next 2 years till 2019.
# 
# Also we can notice, in a specific year the crime rate increased in summer between March and November. From mid of May to first week of September is the worst period in term of crime rate. Chicago police should be extra vigilant between this period. 

# ## Conclusion and results
# 
# 
# We found some of the useful results and insights in this notebook. These are
# 
# 1. Theft is a most common crime in Chicago.
# 2. Majority of crimes happens on street.
# 3. There is a decreasing trend in a crime rate over the years which is good news.
# 4. Majority of the crimes happened in summer between March and November in a specific year. 
