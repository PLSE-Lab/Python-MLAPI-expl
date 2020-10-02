#!/usr/bin/env python
# coding: utf-8

# This is my first notebook. Any comments / suggestions please leave then below :)

# <h1> Comparing climbing and weather data </h1>
# <ol>
#   <li>Explore the data</li>
#     <ol>
#         <li>Climbing</li>
#         <li>Weather</li>
#     </ol>
#   <li>Joining and analysing tables</li>
# </ol> 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_climbing = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv")
df_weather = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv")


# <h3> 1A Explore climbing data </h3>

# In[ ]:


df_climbing.info()


# In[ ]:


df_climbing.describe()


# In[ ]:


df_climbing.head(10)


# Variables present in this dataset
# <ul>
#     <li>Date: Day of the record</li>
#     <li>Route: The route used to climb the Mt Rainier</li>
#     <li>Attempted: Number of people who attempted the climb</li>
#     <li>Succeeded: Number of people who succeeded in reaching the summit</li>
#     <li>Success Percentage: The ratio success</li>
# </ul>
# 
# Even if it is not explicitly stated, I'll consider that "Attempted" means people that have tried to climb the mountain, but failed to do so. Which means that <b>Attempted</b> and <b>Succeeded</b> add up to the total number of people that climbed the mountain.

# In[ ]:


df_climbing[df_climbing["Date"] == "10/3/2015"]


# There are a few repeated lines for the same combination of Date and Route, I believe it means that each observation represents a "climbing group". From this it can be concluded that in the same climbing group some can climb the mountain and some cannot. Let's take a look at the group distribution where all people succeeded, all failed, and all the others.

# In[ ]:


# 2 for all succeeded
# 1 for partial success
# 0 no one succeed

df_climbing["all_group_climbed"] = df_climbing["Succeeded"].apply(lambda x: 2 if x > 0 else 0)
df_climbing.loc[ (df_climbing["Attempted"] > 0) & (df_climbing["Succeeded"] > 0) , "all_group_climbed"] = 1


# In[ ]:


df_climbing["all_group_climbed"].unique()


# Looks like there is no case where everybody in the group succeeded, could it mean that <b>Attempted</b> actually is the total of people that tried to climb the mountain? In that case, it should not be any observation with more Succeeded than Attempted. Let's check.

# In[ ]:


df_climbing[df_climbing["Succeeded"] > df_climbing["Attempted"]]


# ok, so we have some. In that case we have to make a decision. I believe it is safer to assume that for some reason, there is no group that everybody has succeeded in than to assume that these rows above have some kind of error. I may be wrong, but well, it's a decision.

# In[ ]:


df_climbing["all_group_climbed"].value_counts().plot(kind='bar')


# Groups where some people can climb the mountain are more common than groups where everyone fails.

# In[ ]:


gb = df_climbing.groupby(["Route"]).sum().reset_index()

gb = gb[["Route","Attempted","Succeeded"]].melt("Route", var_name="a", value_name="b")
gb["b"] = gb["b"].apply(lambda x: 600 if x > 600 else x)
ax=sns.barplot(x='Route', y='b', hue='a', data=gb)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
print("")


# Just a small note: I cut values larger than 600 for better viewing. But it's easy to see that some routes are much more used than others.

# <h3> 1B Explore Weather data </h3>

# In[ ]:


df_weather.info()


# In[ ]:


df_weather.describe()


# In[ ]:


df_weather.head()


# In[ ]:


no_date_weather = df_weather[["Battery Voltage AVG","Temperature AVG","Relative Humidity AVG","Wind Speed Daily AVG","Wind Direction AVG","Solare Radiation AVG"]]
g = sns.pairplot(no_date_weather)


# Taking a look a the relation between variables. Look's like the behavior of the variables is rather independent, with the exception of <b>Temperature AVG</b> and <b>Solare Radiation AVG</b>, that have a slight linear relation. That should be seen in the heatmap following.

# In[ ]:


sns.heatmap(no_date_weather.corr())


# <h3> Joining and analysing tables </h3>

# I'll merge all rows with the same combination of Date and Route in the climbing table.

# In[ ]:


df_climbing["Date and Route"] = df_climbing["Date"] + "#" + df_climbing["Route"]


# In[ ]:


# Doing some processing
new_df_climbing = df_climbing.groupby("Date and Route").mean().reset_index()
new_df_climbing = new_df_climbing.drop("all_group_climbed", axis=1)
new_df_climbing["Success Percentage"] = new_df_climbing["Succeeded"] / new_df_climbing["Attempted"]


# In[ ]:


# Split the Date and Route column into the individuals Date column and Route column
new = new_df_climbing["Date and Route"].str.split("#", n = 1, expand = True) 
new_df_climbing["Date"] = new[0]
new_df_climbing["Route"] = new[1]
new_df_climbing = new_df_climbing.drop("Date and Route", axis=1)


# In[ ]:


df = new_df_climbing.merge(df_weather, on="Date", how="inner")


# In[ ]:


df.head()


# Alright so far. Let's take a look at the correlation map and see if we can draw conclusions about which variables in the weather data are most related to the success rate.

# In[ ]:


df.corr()["Success Percentage"]


# In[ ]:


sns.heatmap(df.corr())

