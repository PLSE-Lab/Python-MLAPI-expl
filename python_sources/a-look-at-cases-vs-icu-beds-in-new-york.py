#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Imports and read in data
import numpy as np
import pandas as pd

icu_info = pd.read_csv("../input/icu-beds-by-county-in-the-us/data-FPBfZ.csv")
covid_df = pd.read_csv("../input/covid19-in-usa/us_states_covid19_daily.csv")
state_abbrs = pd.read_csv("../input/state-abbreviations/state_abbrev.csv", usecols=["State", "Abbreviation"])


# In[ ]:


# A quick look at our data
icu_info.head()


# In[ ]:


covid_df.head()


# Merging data now.

# In[ ]:


#ICU beds are by county, we need state data
icu_state = icu_info.groupby("State").sum().reset_index()
icu_state=icu_state.merge(state_abbrs, how='left', on="State") #Other dataset references state abbreviations
icu_state=icu_state.rename(columns={"Abbreviation": "state"})
icu_state=icu_state.drop("State", axis=1)


# In[ ]:


icu_state.head()


# In[ ]:


#Merging
covid_df = covid_df.merge(icu_state, how="left", on="state")
covid_df = covid_df[covid_df["ICU Beds"].notna()] #This drops the territorities(e.g. American Samoa) b/c no icu data available
covid_df["date"] = pd.to_datetime(covid_df["date"], format='%Y%m%d')
covid_df.head()


# Let's take a look at how well New York is handling the Covid-19 crisis.

# In[ ]:


df_ny = covid_df[covid_df["state"]=="NY"].set_index("date")
df_ny.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
#sns.set()
df_ny["inIcuCurrently"].fillna(method='ffill',inplace=True)
ax = df_ny.plot(y="inIcuCurrently", legend=False)
ax.set_xlabel("Date")
ax.set_ylabel("Number of People")
df_ny.plot(y="ICU Beds", ax=ax, legend=False, color="r")
ax.figure.legend()
plt.show()


# In[ ]:


# Looking at the data, we see that the number of people in the ICU surpasses the number of beds somtime in early April
# Let's find this point. 
num_beds = df_ny['ICU Beds'][0]
print("num beds ", num_beds)
last_less = df_ny[df_ny['inIcuCurrently']<=num_beds].iloc[0] #Get last day number of people in ICU is less than the number of beds
print("Day before beds run out in NY", last_less.name)


# How does this compare with predictions?
# According to healthleadersmedia.com ... 
# ![image.png](attachment:image.png)
