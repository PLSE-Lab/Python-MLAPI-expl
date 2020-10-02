#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


country_wise_latest = pd.read_csv("../input/corona-virus-report/country_wise_latest.csv")
country_wise_latest = country_wise_latest.set_index("Country/Region")
country_wise_latest.head()


# In[ ]:


country_wise_latest.info()


# In[ ]:


ser = country_wise_latest[["Confirmed","Deaths","Recovered","Active"]].groupby(country_wise_latest["WHO Region"]).sum()
ser.plot(kind="bar")

plt.xlabel("WHO Region")
plt.ylabel("Number of cases in the magnitude of 10^6");


# The above plot shows that the maximum number of confirmed cases are currently in the *`Americas`* with the least number of confirmed cases in *`Western Pacific`* region.
# 
# With a pie plot a better representation is found as the data is normalized to between 0 and 100.

# In[ ]:


ser1 = country_wise_latest[["Confirmed","Deaths","Recovered","Active"]].groupby(country_wise_latest["WHO Region"]).sum()

fig, ax = plt.subplots(2,2,figsize=(15,15))

ax[0,0].pie(ser1["Confirmed"],autopct="%1.1f%%")
ax[0,0].legend(ser1.index,loc="upper right")
ax[0,0].set_title("Confirmed Cases Per Region")

ax[0,1].pie(ser1["Deaths"],autopct="%1.1f%%")
ax[0,1].legend(ser1.index,loc="upper right")
ax[0,1].set_title("Deaths per Region")

ax[1,0].pie(ser1["Recovered"],autopct="%1.1f%%")
ax[1,0].legend(ser1.index,loc="upper right")
ax[1,0].set_title("Cases Recovered per Region")

ax[1,1].pie(ser1["Active"],autopct="%1.1f%%")
ax[1,1].legend(ser1.index,loc="upper right")
ax[1,1].set_title("Cases still Active per Region");


# The maximum number of cases confiremd is the highest in the *Americas* with second highest *Europe*. The high number of confiremd cases could be due to the greater health care system present in those regions with faster testing and documentation of suspected people. *Europe* was also a hotbed for Covid-19 in the early stages of transmission.
# 
# The greatest number of active cases, unsuprisingly is in the *Americas*. There is hardly any difference between all 4 types. This indicates that the curev in this region has maybe stabilised for now, and on the downward trend.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_country_wise_latest = country_wise_latest[["Confirmed","Confirmed last week","1 week change","1 week % increase"]]
df_country_wise_latest = scaler.fit_transform(df_country_wise_latest)

df_country_wise_latest = pd.DataFrame(df_country_wise_latest)
df_country_wise_latest["WHO Region"] = country_wise_latest["WHO Region"].values
df_country_wise_latest.columns = ["Confirmed","Confirmed last week","1 week change","1 week % increase","WHO Region"]
df_country_wise_latest[["Confirmed","Confirmed last week","1 week % increase"]].groupby(df_country_wise_latest["WHO Region"]).mean().plot(kind='bar',legend=True);


# The data is normalized so as to make a comparison easier.

# *Africa* has the highest increase per week. This could be attributed to the recent *Ebola* outbreak in Central Africa with the worst in the Congo.
# 
# In the *Americas* the number of cases is increasing but at a steady rate. They could see a downward curve soon, maybe in 3 months.
# 
# *Easter Meditarranean* has seen a spike in cases, which is evident in the weekly percentage increase. So has *Europe*, which is evident of a second wave in the region. If there is a lockdown, it should be extended for the safety of the populace.
# 
# *Westen Pacific* is currently safe for now compared to the rest of the world, with a slight increase in the cases week to week. Most of the nations in this region are separated by water bodies making a natural barrier for the countries protection. The spike could in the week increase could be due to the second wave in South Korea and surrounding countries. 
# 
# *South East Asia* is currently reducing surprisingly given the fact that population in cites are densely packed into small areas. Maybe the lockdown in this region actually helped. Though the data may be inaccurate here due to inadequate testing.

# In[ ]:


day_wise = pd.read_csv("../input/corona-virus-report/day_wise.csv")
day_wise.set_index("Date",drop=False,inplace=True)
day_wise["Date"] = pd.to_datetime(day_wise["Date"])
day_wise.head()


# In[ ]:


day_wise.info()


# In[ ]:


day_wise[["Date","Confirmed","Active","Recovered","Deaths"]].groupby(pd.Grouper(key='Date',freq='M')).sum().plot(marker="*",legend=True)
plt.title("Cases per Month\n")
plt.xlabel("Month")
plt.ylabel("Number of cases in all countries");


# The number of cases took started to increase rapidly after March, with a decrease after June, in which most countries had started the 3rd phase of lockdowns. 
# 
# Initially Covid-19 was limited to Wuhan, before becoming a pandemic. Even though lockdown in certain countries had been implemented before March, the number of cases spiked due to the 14 day incubation period creating a false safety in certain public places, with suspected patients visiting and spreading the virus. the number of cases increased even when in isolation due the above mentioned incubation period and the symptoms not being adequately addressed.

# In[ ]:


day_wise[["Date","No. of countries"]].groupby(pd.Grouper(key="Date",freq='W')).mean().plot(marker="*",legend=True)


plt.title("Number of countries in which Covid was confirmed \n")
plt.xlabel("Week")
plt.ylabel("Number of Countries");


# Now all the countries have cases as compared to the start of the year with less than 25 countries in total till Feburary. From March to April all the countries confiremd at least one Covid-19 case. This could be due to the number of people going from one place to another before the lockdowns were initiated.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = day_wise[["Confirmed","Deaths","Recovered","New cases","New deaths","New recovered"]]
df = scaler.fit_transform(df)

df = pd.DataFrame(df)
df["Date"]=day_wise["Date"].values
df.columns = ["Confirmed","Deaths","Recovered","New cases","New deaths","New recovered","Date"]

df.groupby(pd.Grouper(key="Date",freq='M')).sum().plot(kind="bar",legend=True)

plt.title("Cases and Numerb of New Cases recorded per Month")
plt.xlabel("Month")
plt.ylabel("Numerb of Cases and New Cases");


# In April the numerb of *New Deaths* spiked which is evident as there was no Standard Operating Procedure for treating the patients, and the ones with a frail immune system fell prey to the virus.<br>
# *New <>* is the numerb of cases as compared to the previous month.

# In[ ]:


full_grouped = pd.read_csv("../input/corona-virus-report/full_grouped.csv")
full_grouped["Date"] = pd.to_datetime(full_grouped["Date"])
# full_grouped.set_index(["Date","WHO Region"],drop=False,inplace=True)
full_grouped.head()


# In[ ]:


full_grouped.info()


# In[ ]:


fig,ax = plt.subplots(6,1,figsize=(6,25))
for i,region in enumerate(full_grouped["WHO Region"].unique()):
    ax[i].plot(full_grouped[["Date","Confirmed","Deaths","Recovered","Active"]][full_grouped["WHO Region"]==region].groupby(pd.Grouper(key="Date",freq='W')).sum(),'*-')
    ax[i].legend(["Confirmed","Deaths","Recovered","Active"],loc='upper left')
    ax[i].set_title(region)
    fig.tight_layout(pad=3.0);
    pass


# The above 6 graphs show how each region as setup by WHO was affected by the pandemic.

# Please do feel free to comment and fork the notebook.
