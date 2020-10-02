#!/usr/bin/env python
# coding: utf-8

# ### Purpose: Follow up to changes in #COVID19 data across countries and states.  
# **Using NYT county and WHO world datasets**   
# 
# By Myrna M Figueroa Lopez

# In[ ]:


#Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #plotting, math, stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting, regressions


# ### What is the COVID19 situation around the **world**?

# In[ ]:


#Dataset from the World Health Organization
World = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/apr13update.csv")
plt.figure(figsize=(21,8)) # Figure size
World.groupby("location")['total_cases'].max().plot(kind='bar', color='darkkhaki')


# In[ ]:


World.head(2)


# Above, the highest rate represents the numbers WORLDWIDE.    
# Then, we see which nations have significat amount of COVID19 cases.   
# As of April 14, 2020, the United States leads in number of COVID19 cases reported.   
# The USA continues to lag in testing, and the numbers could be much higher.   
# Also, China may be under-reporting cases and deaths.
# 

# Visual comparison of reported deaths among China, US, Italy, South Korea, and Brazil.

# In[ ]:


##For ease of visualization
China=World.loc[World['location']== 'China']
USA=World.loc[World['location']== 'United States']
Ital=World.loc[World['location']== 'Italy']
SK=World.loc[World['location']== 'South Korea']
Brzl=World.loc[World['location']== 'Brazil']


# In[ ]:


some1=pd.concat([USA, China, Ital, SK, Brzl]) 

some1=some1.sort_values(by=['date'], ascending=False)
some1.head(2)


# The data depends on what countries report. Even with missing data, we can see the rise and curving of cases in some countries.   
# As the data stands, the US passes all other nations in cases and deaths due to #COVID19

# In[ ]:


##Cases in some countries
plt.figure(figsize=(16,7))
sns.lineplot(x="date", y="total_cases", hue="location",data=some1)
plt.title('Cases per day') # Title
plt.xticks(some1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


##Death rates in some countries
plt.figure(figsize=(16,7))
sns.lineplot(x="date", y="total_deaths", hue="location",data=some1)
plt.title('Deaths per day') # Title
plt.xticks(some1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


US = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
US=US.drop(['fips'], axis = 1) 


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=US, marker='o', color='darkseagreen') 
plt.title('Cases per day across the USA') # Title
plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


#The numbers are exponential (the total from the previous day + that day's data)
US.sort_values(by=['cases'], ascending=False).head(10)


# ### As of April 14, 2020, NY leads in COVID19 cases and deaths.    

# In[ ]:


#Cases across states 
plt.figure(figsize=(19,7)) # Figure size
US.groupby("state")['cases'].max().plot(kind='bar', color='sienna')
plt.title('States COVID-19 case rate') # Title


# In[ ]:


#DEATH rates
plt.figure(figsize=(19,7)) 
US.groupby("state")['deaths'].max().plot(kind='bar', color='mediumvioletred')
plt.title('States COVID-19 death rate') 


# It seems that New York state has the most COVID-19 cases, followed by Illinois. NY also leads in deaths, but Michigan (not Illinois) follows in deaths from COVID-19.

# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='deaths', data=US, marker='o', color='dimgrey') 
plt.title('Deaths across the USA') # Title
plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# Washington state reported the first US case and death. These occured in King county (Seattle, the biggest city in the state, is in this county).   
# KING COUNTY continues to report the highest rates compared to other WA counties.   

# In[ ]:


WA=US.loc[US['state']== 'Washington']


# In[ ]:


WA.sort_values(by=['cases'], ascending=False).head(10)


# In[ ]:


plt.figure(figsize=(12,6)) # Figure size
WA.groupby("county")['cases'].max().plot(kind='bar', color='goldenrod')
plt.title('Total cases across WA counties') 


# In[ ]:


plt.figure(figsize=(12,6)) # Figure size
WA.groupby("county")['deaths'].max().plot(kind='bar', color='lightcoral')
plt.title('Deaths total across WA counties') 


# In[ ]:


plt.figure(figsize=(16,12))
sns.lineplot(x="date", y="deaths", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# Snohomish and King county report the most cases and deaths of COVID-19 in Washington state.
# 

# Conclusion:   
#    - The US did not lead in deaths back in Apr 3. Today, Apr 14, it does.
#    - While WA reported the first US case, the state 'smashed' the curve down.
#    - NY and Michigan has suffered the most deaths from COVID-19.
