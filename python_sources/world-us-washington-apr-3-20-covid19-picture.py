#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #plotting, math, stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting, regressions


# What is the COVID19 situation around the **world** as of April 3, 2020?

# In[ ]:


#Dataset from the World Health Organization
World = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/full_data(12).csv")
plt.figure(figsize=(21,8)) # Figure size
World.groupby("location")['total_cases'].max().plot(kind='bar', color='darkgreen')


# Above, the highest rate represents the numbers WORLDWIDE.    
# Then, we see which nations have significat amount of COVID19 cases.   
# As of April 3 2020, the United States leads in number of COVID19 cases reported.   
# It's important to note that the USA lags in testing, and the number could be much higher.   
# 

# In[ ]:


#total deaths worldwide
plt.figure(figsize=(22,10)) # Figure size
World.groupby("location")['total_deaths'].max().plot(kind='bar', color='tan')


# Again, the highest rate represents the total worldwide.   
# Here, the United States does not lead in COVID19 deaths reported.   
# I should point out that deaths lag behind cases.   
# Italy, and then Spain, lead in number of deaths reported.

# In[ ]:


US = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
US=US.drop(['fips'], axis = 1) 


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=US, marker='o', color='indigo') 
plt.title('Cases per day in the USA') # Title
plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


US.sort_values(by=['cases'], ascending=False).head(30)


# From Jan 21, 2020 up to April 3, 2020, NY leads in COVID19 cases and deaths. COOK COUNTY, ILLINOIS is within the top 30, as well.   

# In[ ]:


#total deaths worldwide
plt.figure(figsize=(19,7)) # Figure size
US.groupby("state")['cases'].max().plot(kind='bar', color='darkblue')


# It seems that New York state has the most COVID-19 cases, followed by Illinois.

# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='deaths', data=US, marker='o', color='dimgrey') 
plt.title('Deaths per day in the USA') # Title
plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


#total deaths worldwide
plt.figure(figsize=(19,7)) # Figure size
US.groupby("state")['deaths'].max().plot(kind='bar', color='coral')


# As of April 3, 2020, New York state leads in deaths, followed by Washington state. 

# Washington state reported the first US case and death. These occured in King county (Seattle, the biggest city in the state, is in this county).   
# How the counties across the state compare to King county?   

# In[ ]:


WA=US.loc[US['state']== 'Washington']


# In[ ]:


WA.sort_values(by=['cases'], ascending=False).head(30)


# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
WA.groupby("county")['cases'].max().plot(kind='bar', color='teal')


# In[ ]:


plt.figure(figsize=(12,8)) # Figure size
WA.groupby("county")['deaths'].max().plot(kind='bar', color='goldenrod')


# In[ ]:


plt.figure(figsize=(16,11))
sns.lineplot(x="date", y="deaths", hue="county",data=WA)
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# Snohomish and Pierce county follow King county in reported cases. These two counties neighbor King county and account for parts of the greater- Seattle area.    
# Deaths lead in King county, followed by Snohomish county. But in thrid place is Whatcom county (leads by a single digit over Pierce county).   
# Whatcom is not next to King county. ALso, Skagit county is in between Whatcom county and Snohomish county.   
# 

# Conclusion:   
# While I have not done a mathematical analysis of the data presented, it is informative for the purpose of educating    
# the general public. The graphs represent the increase of the USA compared to other nations, how states data reflect 
# as of April 3, 2020. Washington was the first state to deal with COVID19 cases and deaths, but seems to maintain the
# 'curve' we hear so much in the news down. Also, counties within the state with the longest history of COVID19 cases  
# do not produce similar case-rates. More data exploration is needed.
# 

# In[ ]:




