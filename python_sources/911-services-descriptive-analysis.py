#!/usr/bin/env python
# coding: utf-8

# ## 911 -Emergency call dataset -
# The data- contains the following fields:-- -  
# 
# - lat : String variable, Latitude
# - lng: String variable, Longitude
# - desc: String variable, Description of the Emergency Call
# - zip: String variable, Zipcode
# - title: String variable, Title
# - timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# - twp: String variable, Township
# - addr: String variable, Address
# - e: String variable, Dummy variable (always 1)

# ## Exploratory Data Analysis (EDA)- Python

# In[ ]:


# Importing required libraries

import numpy as np
import pandas as pd 
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


emer = pd.read_csv("../input/montcoalert/911.csv")


# #### Extracting some insights from the dataset

# In[ ]:


emer.shape


# In[ ]:


emer.info()


# In[ ]:


emer.head(5)


# In[ ]:


emer.isnull().sum()


# #### We see the data has been imported properly and there are not any NAN values in the coulmns like: timeStamp, title, desc, which will of more use to us later.

# ### Data Cleaning:
# #### However we see the last column 'e' has no significance as its a dummy column and all the entries are equal to 1, so its better to discard that column.

# In[ ]:


emer.drop("e", axis=1, inplace=ace=ace=ace = True)
emer.head(3)


# ### Converting "timeStamp" object into a proper DateTime Object:
# #### If you check the type of the timeStamp col, its a series and the values are of string data type, but we need the column is data time format to do analysis.

# In[ ]:


emer["timeStamp"] = pd.to_datetime(emer["timeStamp"])
type(emer["timeStamp"][0])


# ### Spliting the timeStamp col in different cols, Date, Month, Year, Day of week, Hour.

# In[ ]:


emer["Day of week"] = emer["timeStamp"].apply(lambda time: time.dayofweek)
emer["Date"] = emer["timeStamp"].apply(lambda time: time.day)
emer["Month"] = emer["timeStamp"].apply(lambda time: time.month)
emer["Year"] = emer["timeStamp"].apply(lambda time: time.year)
emer["Hour"] = emer["timeStamp"].apply(lambda time: time.hour)
emer.head(4)


# ### If we notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:

# In[ ]:


dtmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
emer["Day of week"] = emer["Day of week"].map(dtmap)
emer.head(4)


# ### Also we can segregate on the day & night time basis

# In[ ]:


emer["Day/Night"] = emer["timeStamp"].apply(lambda x : "Night" if int(x.strftime("%H")) > 18 else "Day")
emer.head(4)


# ## Lets find out the detailed reasons of incidents using title col:

# In[ ]:


emer["Reason"] = emer["title"].apply(lambda i:i.split(":") [0])
emer["Detailed Reason"] = emer["title"].apply(lambda i:i.split(":") [1])
emer.head(4)


# ## Extract which station recorded more number of calls.

# In[ ]:


emer["Station"] = emer["desc"].str.extract("(Station.+?);", expand=False).str.strip()
emer.head(4)


# #### Now we can remove the title and timeStamp col, as we extracted the information from it 

# In[ ]:


del emer["title"]
del emer["timeStamp"]
emer.head(4)


# ## DATA VISUALIZATION

# In[ ]:


# Find out the major reason type of calls made.

plt.figure(figsize=(10,5))
sns.set_context("paper", font_scale = 1.5)
sns.countplot(x='Reason',data=emer, palette = "mako",  saturation=0.9)
sns.set_style("ticks")


# ### Top 10 reasons for which calls were made

# In[ ]:


emer["Detailed Reason"].value_counts().head(15)


# In[ ]:


plt.figure(figsize=(20,10))
sns.set_context("paper", font_scale = 2)
sns.countplot(y='Detailed Reason', data=emer, palette="terrain", order=emer['Detailed Reason'].value_counts().index[:20])
plt.title("Top Cases registered")
sns.set_style("darkgrid")
plt.show()


# ### Top 10 Stations were incidents were reported

# In[ ]:


emer["Station"].value_counts().head(10)


# In[ ]:


plt.figure(figsize=(20,10))
sns.set_context("paper", font_scale =2)
sns.countplot(y='Station',data=emer, palette = "gist_stern", order = emer["Station"].value_counts().index[:10])
plt.title("Top 10 Station with highest call")
sns.set_style("whitegrid")


# ### Finding out if the calls made more in day or night time

# In[ ]:


emer["Day/Night"].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.set_context("paper", font_scale =1.5)
sns.countplot(x='Day/Night',data=emer,palette='gnuplot')
sns.set_style("darkgrid")


# ### Check on which day of the week calls were made more & does that have any significance

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x='Day of week',data=emer,hue='Reason',palette='cubehelix')
plt.title("Calls on each days of the week")
sns.set_style("ticks")


# In[ ]:




