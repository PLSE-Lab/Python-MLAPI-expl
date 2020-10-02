#!/usr/bin/env python
# coding: utf-8

# **Understanding and Predicting the crime rate in Chicago**
# 
# hello everyone, it is my first time series analysis. I have used fb prophet for the first time. 
# This notebook basically covers crime rate analysis using fb prophet, i have used fb prophet to predict the crime rate for 2018,2019,2020.
# 
# please <font style='font-size:20;color:red'>upvote</font>  the notebook if you like it. 
# 
# Please give your valuable suggests.
# 
# 
# I will divide the project into few parts
# 
# 1. EDA
# 2. Predicting

# In[ ]:


import folium
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from fbprophet import Prophet

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")


# In[ ]:



df_1 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)
df_2 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)
df_3 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
df = pd.concat([df_1, df_2, df_3],ignore_index=False, axis=0)

del df_1
del df_2
del df_3


# In[ ]:


df.shape


# # **Exploratory Data Analysis**

# **Viewing the crime zone in map**
# 

# In[ ]:



Map = folium.Map(location=[41.864073,-87.706819],
                        zoom_start=11, tiles='Stamen Terrain')


loclist = df.loc[:, ['Latitude', 'Longitude']]


# In[ ]:


for i in range(len(loclist)):
    x = loclist.iloc[i][0]
    y = loclist.iloc[i][1]
    popup_text = """ <br>"""
    popup_text = popup_text.format(loclist.index[i])
    folium.Marker(location = [x, y], popup= popup_text, fill = True).add_to(Map)


# In[ ]:


Map


# Finding out number of null values and the column names

# In[ ]:


df.isnull().sum().sort_values(ascending=False).keys()


# Dropping irrelevant columns for predicting crime rates

# In[ ]:


df.drop(['Unnamed: 0','Case Number','ID','IUCR','Y Coordinate', 'X Coordinate','Updated On','FBI Code','Beat','Community Area', 'Ward', 'District','Location', 'Latitude','Longitude'],inplace=True, axis=1)


# *Formatting the datetime for future visualization*

# In[ ]:


df.Date=pd.to_datetime(df.Date,format='%m/%d/%Y %I:%M:%S %p')
df.index=pd.DatetimeIndex(df.Date)


# In[ ]:


df.head()


# **Plotting the occurances of primary type of crime in chicago**

# In[ ]:



plt.figure(figsize=(10,10))

df.groupby(['Primary Type']).size().sort_values(ascending=True).plot(kind='barh')
plt.show()


# In[ ]:


df_primary=df['Primary Type'].value_counts().iloc[:20].index
plt.figure(figsize=(15,10))
sns.countplot(y='Primary Type',data=df,order=df_primary)


# **Plotting the occurance of different location description occured in the crime**

# In[ ]:


plt.figure(figsize=(15,10))
df_loc=df['Location Description'].value_counts().iloc[:20].index
sns.countplot(y='Location Description',data=df,order=df_loc)
plt.title('Most location Description')


# > Finding out the number of record in each year from 2006 to 2018

# In[ ]:


plt.plot(df.resample('Y').size())
plt.title('No of Crime per year')
plt.xlabel('Years')
plt.ylabel('Crime')


# Finding out the nature of crime in each month. It seems in winter season, crime rate falls drastically and increases in the middle of the year. 

# In[ ]:


plt.plot(df.resample('M').size())
plt.title('No of Crime per Month')
plt.xlabel('Month')
plt.ylabel('Crime')


# Now it is time for visualizing the crime rate quarterly.

# In[ ]:


plt.plot(df.resample('Q').size())
plt.title('No of Crime per Quarter')
plt.xlabel('Quarter')
plt.ylabel('Crime')


# Let's look how does crime change with respect to day

# In[ ]:


plt.figure(figsize=(15,15))
plt.plot(df.resample('D').size())
plt.title('No of Crime per day')
plt.xlabel('Day')
plt.ylabel('Crime')


# # Now it is time for predicting the future

# In[ ]:


df_prophet=df.resample('M').size().reset_index()


# In[ ]:


df_prophet.head()


# In[ ]:


df_prophet.columns=['Date','Crime value']
df_prophet


# **Renaming the column names**

# In[ ]:


df_prophet_final=df_prophet.rename(columns={'Date':'ds','Crime value':'y'})


# **Intializing the fb prophet**

# In[ ]:


prop= Prophet()
prop.fit(df_prophet_final)


# I am predicting 1500 days after 2018. So I passed 1500 days in the period parameter. Let's look at the graph.

# In[ ]:


future = prop.make_future_dataframe(periods=1500)
forcast= prop.predict(future)
forcast


# Let's visualizate the trend how crimes occur according to prophet.

# In[ ]:


figure=prop.plot(forcast, xlabel='Date',ylabel='Crime Rate')


# In[ ]:


figure=prop.plot_components(forcast)


# From this , it is clear that, crime 
# * decreases in the winter season,
# * increases after winter season
# * highest crime rate in the June 

# 
# 
# *     Please do not hesitate to comment if you have any feedback/comment/question.
# *     An **upvote** would be much appreciated.
# 
# Thanks for reading!

# In[ ]:




