#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #plotting, math, stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting, regressions


# In[ ]:


USA = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
USA= USA.drop(['fips'], axis = 1) 


# In[ ]:


plt.figure(figsize=(20,7)) # Figure size
plt.title('COVID-19 cases across US states') # Title
USA.groupby("state")['cases'].max().plot(kind='bar', color='blue')


# In[ ]:


plt.figure(figsize=(20,7)) # Figure size
plt.title('COVID-19 deaths across US states') # Title
USA.groupby("state")['deaths'].max().plot(kind='bar', color='red')


# As of April 3, 2020, NY and Illinois reported the most cases, while NY and Michigan reported the most COVD-19 related deaths.

# In[ ]:


NY=USA.loc[USA['state']== 'New York']
LA=USA.loc[USA['state']== 'Louisiana']
WA=USA.loc[USA['state']== 'Washington']
IL=USA.loc[USA['state']== 'Illinois']
PUR=USA.loc[USA['state']== 'Puerto Rico']


# In[ ]:


# Concatenate dataframes 
States=pd.concat([NY,LA,WA,IL,PUR]) 


# In[ ]:


States=States.sort_values(by=['date'], ascending=True)


# In[ ]:


plt.figure(figsize=(15,9))
plt.title('COVID-19 cases comparison of WA, IL, NY, LA, and PR') # Title
sns.lineplot(x="date", y="cases", hue="state",data=States)
plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
plt.title('COVID-19 deaths comparison of WA, IL, NY, LA, and PR') # Title
sns.lineplot(x="date", y="deaths", hue="state",data=States, palette="cubehelix")
plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# Washington state seems to be containing the curve of reported cases. However, this could be because of limited testing availability. However, the number of deaths for that state remain low as well.   
# NY shows a sharp upward curve in cases. Population and wide testing could be a factor. Deaths in that state are the highest, compared to other states.   
# Puerto Rico reported their first cases later than NY and WA. The curve above shows a sharper increase in cases and deaths. The PR curve resembles that of NY, and not of the other states. 

# King county, the first to report a #COVID19 case in the US, has a population of about 2.189 million.   
# NYC has a population of 8.623 million.   
# Puerto Rico has a population of 3.2 million.
# 

# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=WA, marker='o', color='blue') 
plt.title('Cases per day in Washington state') # Title
plt.xticks(WA.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


plt.figure(figsize=(18,11))

sns.lineplot(x="date", y="cases", hue="county",data=WA, palette="Set3")
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.title('Cases per county in Washington state') # Title
plt.show()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=NY, marker='o', color='green') 
plt.title('Cases per day in New York') # Title
plt.xticks(NY.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# According to [NY MAG](https://nymag.com/intelligencer/article/new-york-coronavirus-cases-updates.html), 113,704 total cases in New York State. Averaging about 1800 cases per day in 34 days. 

# In[ ]:


plt.figure(figsize=(22,10))
sns.lineplot(x="date", y="cases", hue="county",data=NY)
plt.xticks(NY.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.title('Cases per county in New York state') # Title
plt.show()


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=PUR, marker='o', color='purple') 
plt.title('Cases per day in Puerto Rico') # Title
plt.xticks(PUR.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# In[ ]:


PUR


# The data above is not divided by county (in PR, municipalities). 

# 22 days from first reported patient 378 at a rate of 17.18% increase in cases  per day. 
