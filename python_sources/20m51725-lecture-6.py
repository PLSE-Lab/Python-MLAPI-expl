#!/usr/bin/env python
# coding: utf-8

# # This is excercise 6
#  
# We look at Corona cases for a specific country.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country = "Russia"
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
#print(np.unique(df["Country/Region"].values))
df = (df[df["Country/Region"]==selected_country])
df = df.groupby("ObservationDate").sum()
print(df)


# Note: This "df" is actually a cumulative information.
# June 7, 2020 shows the total cases since initial date.
# 
# How can we calculate day-to-day cases?? Or daily Increases??

# In[ ]:


df["daily_confirmed"] = df["Confirmed"].diff()
df["daily_deaths"] = df["Deaths"].diff()
df["daily_recovery"] = df["Recovered"].diff()
df["daily_confirmed"].plot()
df["daily_recovery"].plot()
df["daily_deaths"].plot()


# In[ ]:


print(df)


# # How about we make an interactive chart?
# 
# To do this, we'll need to load two of the following modules.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df["daily_confirmed"].values,name="Daily confirmed")
daily_deaths_object = go.Scatter(x=df.index,y=df["daily_deaths"].values,name="Daily deaths")
daily_recovered_object = go.Scatter(x=df.index,y=df["daily_recovery"].values,name="Daily recovery")

layout_object = go.Layout(title= "Russia daily cases 20M51725",xaxis=dict(title="Date"),yaxis=dict(title="Number of people"))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html("Russia_daily_cases_20M51725.html")


# file:///C:/Users/Owner/Downloads/Russia_daily_cases_20M51725.html

# # How can we make an informative table?
# 
# Maybe color the entries large values as some bright color, low values as some dark color.

# In[ ]:


df1 = df#[["daily_confirmed"]]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap="gist_ncar").highlight_max("daily_confirmed").set_caption("Daily Summaries")
display(styled_object)
f = open("table_20M51725.html","w")
f.write(styled_object.render())


# file:///C:/Users/Owner/Downloads/table_20M51725.html

# # How can we calculate global ranking?
# 
# Maybe selecting only the latest date???

# In[ ]:


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df1 = df.groupby(["ObservationDate","Country/Region"]).sum()
df2 = df[df["ObservationDate"]=="06/07/2020"].sort_values(by=["Confirmed"],ascending=False).reset_index()
print(df2[df2["Country/Region"]=="Russia"])


# # Confirmed ranking
# 
# From the news, the ranking is
# 1. United states
# 2. Brazil
# 3. Russia
# 
# so, I survey these 3 countries and my country Japan.

# In[ ]:


df2 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df2 = (df2[df2["Country/Region"]=="Russia"])
df2 = df2.groupby("ObservationDate").sum()

df3 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df3 = (df3[df3["Country/Region"]=="US"])
df3= df3.groupby("ObservationDate").sum()

df4 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df4 = (df4[df4["Country/Region"]=="Brazil"])
df4= df4.groupby("ObservationDate").sum()

df5 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df5 = (df5[df5["Country/Region"]=="Japan"])
df5= df5.groupby("ObservationDate").sum()

df2["daily_confirmed"] = df2["Confirmed"].diff()
df3["daily_confirmed"] = df3["Confirmed"].diff()
df4["daily_confirmed"] = df4["Confirmed"].diff()
df5["daily_confirmed"] = df5["Confirmed"].diff()

df2["daily_confirmed"].plot()
df3["daily_confirmed"].plot()
df4["daily_confirmed"].plot()
df5["daily_confirmed"].plot()


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

US_confirmed_object = go.Scatter(x=df3.index,y=df3["daily_confirmed"].values,name="United States")
Brazil_confirmed_object = go.Scatter(x=df4.index,y=df4["daily_confirmed"].values,name="Brazil")
Russia_confirmed_object = go.Scatter(x=df2.index,y=df2["daily_confirmed"].values,name="Russia")
Japan_confirmed_object = go.Scatter(x=df5.index,y=df5["daily_confirmed"].values,name="Japan")

layout_object = go.Layout(title= "ranking of confirmed number",xaxis=dict(title="Date"),yaxis=dict(title="Number of people"))
fig = go.Figure(data=[US_confirmed_object,Brazil_confirmed_object,Russia_confirmed_object,Japan_confirmed_object],layout=layout_object)
iplot(fig)
fig.write_html("ranking of confirmed.html")


# # Death ranking
# 
# The number of dead people in Russia is not so large.
# the ranking is..
# 
# 1. United states
# 2. Brazil
# 3. British
# 
# So I survey the US and Brazil, my target country Russia, and my country Japan.

# In[ ]:


df2 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df2 = (df2[df2["Country/Region"]=="Russia"])
df2 = df2.groupby("ObservationDate").sum()

df3 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df3 = (df3[df3["Country/Region"]=="US"])
df3= df3.groupby("ObservationDate").sum()

df4 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df4 = (df4[df4["Country/Region"]=="Brazil"])
df4= df4.groupby("ObservationDate").sum()

df5 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df5 = (df5[df5["Country/Region"]=="Japan"])
df5= df5.groupby("ObservationDate").sum()

df2["daily_deaths"] = df2["Deaths"].diff()
df3["daily_deaths"] = df3["Deaths"].diff()
df4["daily_deaths"] = df4["Deaths"].diff()
df5["daily_deaths"] = df5["Deaths"].diff()

df2["daily_deaths"].plot()
df3["daily_deaths"].plot()
df4["daily_deaths"].plot()
df5["daily_deaths"].plot()


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

US_deaths_object = go.Scatter(x=df3.index,y=df3["daily_deaths"].values,name="United States")
Brazil_deaths_object = go.Scatter(x=df4.index,y=df4["daily_deaths"].values,name="Brazil")
Russia_deaths_object = go.Scatter(x=df2.index,y=df2["daily_deaths"].values,name="Russia")
Japan_deaths_object = go.Scatter(x=df5.index,y=df5["daily_deaths"].values,name="Japan")

layout_object = go.Layout(title= "ranking of deaths",xaxis=dict(title="Date"),yaxis=dict(title="Number of people"))
fig = go.Figure(data=[US_deaths_object,Brazil_deaths_object,Russia_deaths_object,Japan_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html("ranking of deaths.html")


# # Recovered ranking
# 
# From the news, the ranking is
# 
# 1. United states
# 2. Brazil
# 3. Russia
# 
# so, I survey these 3 countries and my country Japan.

# In[ ]:


df2 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df2 = (df2[df2["Country/Region"]=="Russia"])
df2 = df2.groupby("ObservationDate").sum()

df3 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df3 = (df3[df3["Country/Region"]=="US"])
df3= df3.groupby("ObservationDate").sum()

df4 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df4 = (df4[df4["Country/Region"]=="Brazil"])
df4= df4.groupby("ObservationDate").sum()

df5 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df5 = (df5[df5["Country/Region"]=="Japan"])
df5= df5.groupby("ObservationDate").sum()

df2["daily_recovered"] = df2["Recovered"].diff()
df3["daily_recovered"] = df3["Recovered"].diff()
df4["daily_recovered"] = df4["Recovered"].diff()
df5["daily_recovered"] = df5["Recovered"].diff()

df2["daily_recovered"].plot()
df3["daily_recovered"].plot()
df4["daily_recovered"].plot()
df5["daily_recovered"].plot()


# Maybe the data of Brazil is not accurate, so excluded.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

US_recovered_object = go.Scatter(x=df3.index,y=df3["daily_recovered"].values,name="United States")

Russia_recovered_object = go.Scatter(x=df2.index,y=df2["daily_recovered"].values,name="Russia")
Japan_recovered_object = go.Scatter(x=df5.index,y=df5["daily_recovered"].values,name="Japan")

layout_object = go.Layout(title= "ranking of recovered",xaxis=dict(title="Date"),yaxis=dict(title="Number of people"))
fig = go.Figure(data=[US_recovered_object,Russia_recovered_object,Japan_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html("ranking of recovered.html")


# # Result and Consideration
# 
# I found that the Russia has a large number of infected people, but few dead, and the number of the recovery is high for the number of the infected. But according to my Russian teacher, Russian information is not as accurate as Japan. So we do not really know.
# And my country Japan, the number of infected people is increasing and everyone is afraid , but it is very low in the worldwide.

# In[ ]:


#Ranking by 'Confirmed' case
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
latest = data[data.index=='06/02/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

#New Zealand's Ranking
print('Ranking of Russia: ', latest[latest['Country/Region']=='Russia'].index.values[0]+1)


# # News
# 
# In Russia, in order to prevent infection, people who have mild or asymptomatic infections are confirming whether they are quarantining the location information application on their smartphones. This is called a digital concentration camp and is being discussed in Russia.
# In addition, the capital block of Moscow was lifted from June 9th, despite the fact that the infection had not subsided.
# 
# https://news.biglobe.ne.jp/international/0609/reu_200609_7298383882.html
# https://www.yomiuri.co.jp/world/20200406-OYT1T50176/
# 
