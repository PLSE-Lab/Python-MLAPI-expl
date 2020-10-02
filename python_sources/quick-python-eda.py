#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import mean


# In[ ]:


plt.style.use('ggplot')
data = pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')
print(data.columns)


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# Looking at a heatmap of the numerical variables to give an idea of the correlations of each column...
# I am particularly interested in what causes severity of crashes, so I'll pay attention to that...

# In[ ]:


fig = sb.heatmap(data[['TMC','Severity','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']].corr(), annot=True, linewidths=0.2)
fig = plt.gcf()
fig.set_size_inches(20,20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


# Lets look at Sunrise vs Sunset at the Severity of crashes

# In[ ]:


plt.figure(figsize=(12,8))
df = data.groupby(['Sunrise_Sunset'])
df.Severity.mean().plot(kind='bar')
plt.ylabel("Severity", fontsize=(15))
plt.xlabel("Sunrise vs Sunset", fontsize=(15))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# Crashes at sunrise has a lower mean severity than crashes at sunset

# In[ ]:


plt.figure(figsize=(20,20))
df = data
df.groupby(['Weather_Condition']).size().nlargest(10).plot.pie(autopct='%1.1f%%', fontsize=(20))


# These are the top 10 most common weather conditions crashes occur in.
# 

# In[ ]:


plt.figure(figsize=(15,10))
data['Weather_Condition'].value_counts().nlargest(10).plot.bar()
plt.ylabel("Number of Accidents")
plt.xticks(fontsize=15)


# Lets look at the mean severity compared to these weather conditions..

# In[ ]:


top10 = data['Weather_Condition'].value_counts().nlargest(10)
plt.figure(figsize=(20,10))
df = data[data['Weather_Condition'].isin(top10.index)]
sb.barplot(x='Weather_Condition', y='Severity', estimator=mean, data=df)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Weather Conditions", fontsize=15)
plt.ylabel("Severity", fontsize=15)


# Light snow has the highest mean severity of these common weather conditions

# In[ ]:


plt.figure(figsize=(20,10))
sb.barplot(x="Severity", y="Temperature(F)", estimator=mean, data=data, ci=False)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Severity", fontsize=20)
plt.ylabel("Temperature", fontsize=20)


# Severity 4 crashes have a higher mean occurance at lower temperature. Severity one crashes have a higher mean occurance at higher temperatures
# 

# In[ ]:


plt.figure(figsize=(15,10))
sb.lineplot(x="Severity", y="Visibility(mi)", ci=False, estimator=mean, data=data)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Severity", fontsize=20)
plt.ylabel("Visibility(mi)", fontsize=20)


# This shows that Severity definitely has a relationship with visibilty... lower visibility means more severe crashes on average...
