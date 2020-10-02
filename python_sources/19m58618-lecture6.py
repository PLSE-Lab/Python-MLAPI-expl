#!/usr/bin/env python
# coding: utf-8

# # Assignment 6
# 
# Here shows Corona cases for New Zealand

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df = df[df["Country/Region"]=="New Zealand"]
df = df.groupby("ObservationDate").sum()
print(df)


# # Plot of daily confirmed, daily deaths, daily recovery  
# 

# In[ ]:


df["daily_confirmed"] = df["Confirmed"].diff()
df["daily_deaths"] = df["Deaths"].diff()
df["daily_recovery"] = df["Recovered"].diff()
df["daily_confirmed"].plot()
df["daily_deaths"].plot()
df["daily_recovery"].plot()
plt.show


# # Interactive chart
# 
# We can see from the chart that the first confirmed case in New Zealand is on 4th March and first death case is on 29th March.
# There is an increasing daily confirmed cases trend from 17th March to 5th April.After that, daily confirmed cases gradually have decreased.
# From 9th April, daily recovery became more than daily confirmed+daily deaths, additionally,daily deaths cases remains very few always,only 1 or 2 or 3.It can be said that New Zealand controled the virus relatively well. 
# 

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df["daily_confirmed"].values,name="Daily confirmed")
daily_deaths_object = go.Scatter(x=df.index,y=df["daily_deaths"].values,name="Daily deaths")
daily_recovery_object = go.Scatter(x=df.index,y=df["daily_recovery"].values,name="Daily recovery")

layout_object = go.Layout(title="New Zealand daily cases 19M58618",xaxis=dict(title="Date"),yaxis=dict(title="Number of people"))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html("New Zealand_daily_cases_19M58618.html")


# # Informative table

# In[ ]:


df1 = df#[["daily_confirmed"]]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap="gist_ncar").highlight_max("daily_confirmed").set_caption("Daily Summaries")
display(styled_object)
f = open("table_19M58618.html","w")
f.write(styled_object.render())


# # Global ranking

# In[ ]:


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df = df[df["ObservationDate"]=="06/10/2020"]
df = df.groupby(["Country/Region"]).sum()
df1 = df.sort_values(by=["Confirmed"],ascending=False).reset_index()
print("Confirmed Ranking of New Zealand in 20200610: ",df1[df1["Country/Region"]=="New Zealand"].index.values[0]+1)




# # How the national government of New Zealand is addressing the issue 
# 
# New Zealand has an alert system for the coronavirus threat. The alert level will be updated daily by Government.As the threat level rises, increasingly severe measures are rolled out to combat the spread of the virus.The threat levels range from one to four, with four being the most severe. Importantly, the threat levels can apply to the whole country or to specific regions depending on what is happening on the ground.
# 
# No matter the threat level, essential shops like supermarkets and pharmacies will remain open but people are encouraged to shop normally so supermarkets have time to restock their shelves. And the public health staff actively investigate every case they are notified about and activate contact tracing.
# 
# LEVEL 4: DON'T GO ANYWHERE
# 
# LEVEL 3: LIMIT TRAVEL, PUBLIC VENUES CLOSED
# 
# LEVEL 2: REDUCE CONTACT, LIMIT NON-ESSENTIAL TRAVEL, WORK FROM HOME
# 
# LEVEL 1: KEEP OUT PANDEMIC; BORDER RESTRICTIONS
# 
# ----*Coronavirus: New Zealand has a threat system, here's how it works and how it will affect you*

# 
