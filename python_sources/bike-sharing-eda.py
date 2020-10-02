#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")
print(data.shape)
data.head()


# In[ ]:


data.info()


# ###  Preprocessing Data for EDA

# In[ ]:


data.dtypes


# In[ ]:


data["timestamp"]=pd.to_datetime(data["timestamp"])
data.dtypes


# In[ ]:


data["season"] = data["season"].replace(0,"spring").replace(1,"summer").replace(2,"autumn").replace(3,"winter")


# In[ ]:


data["is_holiday"] = data["is_holiday"].replace(0,"not holiday").replace(1,"holiday")


# In[ ]:


data["weather_code"] = data["weather_code"].replace(1,"Clear").replace(2,"scattered clouds / few clouds").replace(3,"Broken clouds").replace(4,"Cloudy").replace(7,"Rain/ light Rain shower/ Light rain").replace(10,"rain with thunderstorm").replace(26,"snowfall").replace(94,"Freezing Fog")


# In[ ]:


data["year"] = data["timestamp"].dt.year
data["month"] = data["timestamp"].dt.month
data["day"] = data["timestamp"].dt.day
data["hour"] = data["timestamp"].dt.hour
data["datetime-dayofweek"] = data["timestamp"].dt.dayofweek
data.head()


# In[ ]:


data.loc[data["datetime-dayofweek"] == 0, "weekday"] = "Monday"
data.loc[data["datetime-dayofweek"] == 1, "weekday"] = "Tuesday"
data.loc[data["datetime-dayofweek"] == 2, "weekday"] = "Wednesday"
data.loc[data["datetime-dayofweek"] == 3, "weekday"] = "Thursday"
data.loc[data["datetime-dayofweek"] == 4, "weekday"] = "Friday"
data.loc[data["datetime-dayofweek"] == 5, "weekday"] = "Saturday"
data.loc[data["datetime-dayofweek"] == 6, "weekday"] = "Sunday"
data.head()


# In[ ]:


column = ["year","month","day","hour","weekday","cnt","t1","t2","hum","wind_speed","weather_code","is_holiday","is_weekend","season"]
data = data[column]
data.head()


# ## EDA

# In[ ]:


figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(18, 8)

sns.barplot(data=data, x="year", y="cnt", ax=ax1)
sns.barplot(data=data, x="month", y="cnt", ax=ax2)
sns.barplot(data=data, x="day", y="cnt", ax=ax3)
sns.barplot(data=data, x="hour", y="cnt", ax=ax4)
plt.show()


# ### We make number of sharing per hour, year, month, and day. We can make some inferences about the content of the graph.
# - It does not appear to have grown over the years. 2017 data is not enough
# - The date does not appear to have a significant impact on the number of sharing
# - It looks like there are a lot of sharing between June and August.
# - Looks like a lot of sharing at 7-9 and 17-19.

# In[ ]:


figure, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1)

figure.set_size_inches(18, 20)

sns.pointplot(data=data, x="hour", y="cnt", ax=ax1)
sns.pointplot(data=data, x="hour", y="cnt", hue="is_holiday", ax=ax2)
sns.pointplot(data=data, x="hour", y="cnt", hue="weekday", ax=ax3)
sns.pointplot(data=data, x="hour", y="cnt", hue="season", ax=ax4)
sns.pointplot(data=data, x="hour", y="cnt", hue="weather_code", ax=ax5)

plt.show()


# In[ ]:


plt.figure(figsize=[18,15])
sns.pointplot(data=data, x="hour", y="cnt", hue="weather_code")
plt.show()


# ### I created  graph that included various conditions. I can check the meaningful results.
# - You can see the number of sharing increasing between 7-9 and 17-19. We need to check the detailed customer, but we can infer that the amount of bicycle sharing increases during rush hour or school hours. We look forward to having a good effect when we come up with a target marketing plan.
# - sharing looks different holiday and workingday. On woringday, the amount of sharing increases during rush hour and back-to-school hours, but on holiday, it increases during lunchtime. I think I need a different way of marketing on holiday and workinday.
# - In a similar context, if you compare the days of the week, you can see the difference between weekends and weekdays. Similarly, services must be provided separately from weekends and weekdays.
# - The effect of seasonal and weather on bike sharing seems to be consistent with common sense. In the summer when it's good to ride a bike, there's the most sharing, and in the winter when it's not good to ride a bike, there's the least sharing. On bad weather,there are fewer sharing, and on good weather, there are more sharing.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




