#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing libraries

from numpy import *
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pandas_profiling import ProfileReport
import plotly.express as px
plt.style.use("ggplot")


# In[ ]:


# creating the dataframe
df = pd.read_csv("../input/flight-route-database/routes.csv")
df.head()


# In[ ]:


# extracting the information of the dataframe using info and describe method
df.info()
df.describe()
# all the columns of the dataframe are of object type except for the stops


# In[ ]:


# making a profile report on the dataframe
report = ProfileReport(df)
report


# ## checking the dataframe for any null value and visualising it using heatmap**

# In[ ]:


print(df.isna().sum())
sns.heatmap(df.isnull() , yticklabels = False)


# ### The column "codeshare" has many missing values so dropping that column

# In[ ]:


df.drop(" codeshare" , axis = 1 , inplace = True)
df.head()


# # Ques1 - What are the top 10 airlines based on the number of flights?

# In[ ]:


# fixing the names of the column for any spaces
df.columns = ['airline', 'airline ID', 'source airport', 'source airport id',
       'destination airport', 'destination airport id', 'stops',
       'equipment']
df.columns


# In[ ]:


airline = df[["airline"]]
airline["count"] = airline.groupby(airline.airline)["airline"].transform("count")
airline = airline.drop_duplicates()
airline = airline.sort_values(by = "count" , ascending = False)
airline = airline.head(10)

# plotting a pie chart
fig = px.pie(data_frame = airline , values = "count" , names = "airline" , template = "seaborn")
fig.update_traces(textinfo = "percent+label" , pull = 0.05 , rotation = 90)
fig.show()

# plotting the bar chart
plt.figure(figsize = (8 , 6))
plt.bar(data = airline , x = "airline" , height = "count" , alpha = 0.7)
plt.xlabel("Airlines")
plt.ylabel("Number of flights")


# # Ques2- What are the top 10 aircraft types?

# In[ ]:


equipment = df[["equipment"]]
equipment["count"] = equipment.groupby(equipment.equipment)["equipment"].transform("count")
equipment = equipment.drop_duplicates()
equipment = equipment.sort_values(by = "count" , ascending = False)
equipment = equipment.head(10)

# plotting a pie chart
fig = px.pie(data_frame = equipment , values = "count" , names = "equipment" , template = "seaborn")
fig.update_traces(textinfo = "percent+label" , pull = 0.05 , rotation = 90)
fig.show()

# plotting the bar chart
plt.figure(figsize = (8 , 6))
plt.bar(data = equipment , x = "equipment" , height = "count" , alpha = 0.7)
plt.xlabel("Equipment")
plt.ylabel("Number of equipments")


# # Ques3 - What are the top 10 departure airports?

# In[ ]:


sr = df[["source airport"]]
sr["count"] = sr.groupby(sr["source airport"])["source airport"].transform("count")
sr = sr.drop_duplicates()
sr = sr.sort_values(by = "count" , ascending = False)
sr = sr.head(10)

# plotting a pie chart
fig = px.pie(data_frame = sr , values = "count" , names = "source airport" , template = "seaborn")
fig.update_traces(textinfo = "percent+label" , pull = 0.05 , rotation = 90)
fig.show()

# plotting the bar chart
plt.figure(figsize = (8 , 6))
plt.bar(data = sr , x = "source airport" , height = "count" , alpha = 0.7)
plt.xlabel("Source Airports")
plt.ylabel("Number of departure flights")


# # Ques4 - What are the top 10 destination airports?

# In[ ]:


df.columns


# In[ ]:


da = df[["destination airport"]]
da["count"] = da.groupby(da["destination airport"])["destination airport"].transform("count")
da = da.drop_duplicates()
da = da.sort_values(by = "count" , ascending = False)
da = da.head(10)

# plotting a pie chart
fig = px.pie(data_frame = da , values = "count" , names = "destination airport" , template = "seaborn")
fig.update_traces(textinfo = "percent+label" , pull = 0.05 , rotation = 90)
fig.show()

# plotting the bar chart
plt.figure(figsize = (8 , 6))
plt.bar(data = da , x = "destination airport" , height = "count" , alpha = 0.7)
plt.xlabel("Destination Airports")
plt.ylabel("Number of arrived flights")


# # Ques5 - What are the number of flights having 1 stops and extracting it's details.

# In[ ]:


df["stops"].value_counts()


# ### Only 11 flights have 1 stops rest other are non stop flights

# In[ ]:


# extracting their details

df_1_stop = df[df["stops"] == 1]
df_1_stop


# In[ ]:




