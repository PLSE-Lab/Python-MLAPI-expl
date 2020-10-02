#!/usr/bin/env python
# coding: utf-8

# # CORONAVIRUS IN PAKISTAN
# 
# ## 1. Initial Steps and basic Data Preprocessing

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'], format = '%m/%d/%Y')


df.head()
df.shape

df_pakistan = df[df['Country/Region'] == 'Pakistan']
df_pakistan.head()
df_pakistan.shape


# ### 1.1 Missing Values
# 
# Up until 10th June 2020, Pakistan's daily breakdown of cases is on the national level, hence the 'Province/State' column has NaNs in it. In order to make it easier for us later, I just replace these with 'National'. 

# In[ ]:


len(df_pakistan['ObservationDate'].unique())

df_pakistan['Province/State'] = df_pakistan['Province/State'].fillna('National') # to signify that this number represents that national number of cases

df_pakistan.isna().sum() #perfect, no missing values nonw


# ## 2. EDA/Visualization
# 
# ### 2.1. Timeline of Confirmed Covid-19 Cases in Pakistan
# 
# As there is no national data beyond 10th June, I had to create another dataframe that only contains the total infections on the national level after 10th June. The first line of code in the bottom snippet pertains to that. 
# 
# **Observations:**
# 
# We can see the cases, much like many other countries, have been increasing quite rapidly. We can see that the infection rate started to accelerate around 10th May. This is because on 9th May, Pakistan decided to [lift the lockdown](https://www.reuters.com/article/us-health-coronavirus-pakistan-lockdown/after-pakistans-lockdown-gamble-covid-19-cases-surge-idUSKBN23C0NW).
# 
# But we can see that the rate has decelarated slightly in the last two weeks. 

# In[ ]:


df_pak_national = df_pakistan.groupby('ObservationDate').sum() #beyond 9th June, we only have provincial totals, hence we need to sum for the national value
df_pak_national.tail()
fig = px.line(df_pak_national, y = 'Confirmed', title = 'Timeline of Confirmed Covid-19 Cases in Pakistan')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step = 'all')
        ])
    )
)
fig.show()


# ### 2.2. Provincial Timeline of Confirmed Covid-19 Cases in Pakistan since 10th June
# 
# Since data at the provincial level is only available since 10th June, the x-axis here is much shorter.
# 
# **Observations:**
# 
# We can see that, at the moment, Sindh is leading the tally while Punjab is lagging closely behind. An interesting thing to note here is the switch that is taking place mid-June. Even though Sindh was the initial epicenter of Covid-19 in Pakistan, a timely and strict lockdown slowed down the rate of infections whereas the infection rate in Punjab started accelerating until it overtook Sindh. However, this again switch mid June. An important thing to note here is the fact that Punjab's population is double that of Sindh. 

# In[ ]:


fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Confirmed', color = 'Province/State', title = 'Provincial Timeline of Confirmed Covid-19 Cases in Pakistan since 10th June')
fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])

fig.show()


# ### 2.3. Timeline of Confirmed Covid-19 Fatalities in Pakistan

# In[ ]:


fig = px.line(df_pak_national, y = 'Deaths', title = 'Timeline of Confirmed Covid-19 Fatalities in Pakistan')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step = 'all')
        ])
    )
)
fig.show()


# ### 2.4. Provincial Timeline of Confirmed Covid-19 Fatalities in Pakistan since 10th June
# 
# 
# **Observations:**
# 
# Even though Sindh has more cases, Punjab has consistenly had a higher number of fatalities.

# In[ ]:


fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Deaths', color = 'Province/State', title = 'Provincial Timeline of Confirmed Covid-19 Fatalities in Pakistan since 10th June')
fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])

fig.show()


# ### 2.5. Timeline of Confirmed Covid-19 Recoveries in Pakistan

# In[ ]:


fig = px.line(df_pak_national, y = 'Recovered', title = 'Timeline of Confirmed Covid-19 Recoveries in Pakistan')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step = 'all')
        ])
    )
)
fig.show()


# ### 2.6. Provincial Timeline of Confirmed Covid-19 Recoveries in Pakistan since 10th June
# 

# In[ ]:


fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Recovered', color = 'Province/State', title = 'Provincial Timeline of Confirmed Covid-19 Recoveries in Pakistan since 10th June')
fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])

fig.show()


# In[ ]:


df_pak_national['Death_Rate'] = df_pak_national['Deaths'] / df_pak_national['Confirmed'] * 100
df_pak_national['Recovery_Rate'] = df_pak_national['Recovered'] / df_pak_national['Confirmed'] * 100


# ### 2.7. Timeline of Covid-19 Fatality Rates in Pakistan
# 
# Fatality Rate = (Fatalities / Confirmed Cases) * 100
# 
# **Observations:**
# 
# We can see there is a continuous increase in the fatality rate between March and mid May. However, since May, the fatality rate has stabilized between 1.9% and 2.1%.

# In[ ]:


fig = px.line(df_pak_national, y = 'Death_Rate', title = 'Timeline of Covid-19 Fatality Rates in Pakistan')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step = 'all')
        ])
    )
)
fig.show()


# ### 2.8. Timeline of Confirmed Covid-19 Recovery Rates in Pakistan

# In[ ]:


fig = px.line(df_pak_national, y = 'Recovery_Rate', title = 'Timeline of Confirmed Covid-19 Recovery Rates in Pakistan')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step = 'all')
        ])
    )
)
fig.show()


# ### 2.9. Provincial Timeline of Covid-19 Fatality Rates in Pakistan since 10th June
# 
# **Observations:**
# 
# We can see that Khyber Pakhtunkhwa has a much higher fatality rate compared to other provinces. 
# 
# On the other hand, Islamabad has the lowest. It makes sense since it is the capital and has a much better healthcare infrastructure compared to the rest of the country.

# In[ ]:


df_pakistan['Death_Rate'] = df_pakistan['Deaths'] / df_pakistan['Confirmed'] * 100
df_pakistan['Recovery_Rate'] = df_pakistan['Recovered'] / df_pakistan['Confirmed'] * 100

fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Death_Rate', color = 'Province/State', title = 'Provincial Timeline of Covid-19 Fatality Rates in Pakistan since 10th June')
fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])

fig.show()


# ### 2.10. Provincial Timeline of Covid-19 Recovery Rates in Pakistan since 10th June
# 
# **Observations:**
# 
# The territory of Gilgit-Baltistan has a surprisingly high recovery rate. It is closely followed by Islamabad.
# 

# In[ ]:


fig = px.line(df_pakistan, x = 'ObservationDate', y = 'Recovery_Rate', color = 'Province/State', title = 'Provincial Timeline of Covid-19 Recovery Rates in Pakistan since 10th June')
fig.update_layout(xaxis_range=['2020-06-10','2020-07-08'])

fig.show()


# ### 2.11. Weekly Tally of Confirmed Cases, Recoveries and Deaths in Pakistan
# 
# Weekly breakdown shows us that the Confirmed Cases curve is slowly flattening out. However, it must be noted that Pakistan is in a precarious state. If it slips on its precationary measures, it can experience something like what US is currently going through right now.

# In[ ]:


df_pak_national["WeekOfYear"] = df_pak_national.index.weekofyear

week = []
pak_weekly_cases = []
pak_weekly_recoveries = []
pak_weekly_deaths = []

w = 1

for i in list(df_pak_national["WeekOfYear"].unique()):
    pak_weekly_cases.append(df_pak_national[df_pak_national["WeekOfYear"] == i]["Confirmed"].iloc[-1])
    pak_weekly_recoveries.append(df_pak_national[df_pak_national["WeekOfYear"] == i]["Recovered"].iloc[-1])
    pak_weekly_deaths.append(df_pak_national[df_pak_national["WeekOfYear"] == i]["Deaths"].iloc[-1])
    week.append(w)
    w = w + 1
    
fig = go.Figure()
fig.add_trace(go.Scatter(x = week, y = pak_weekly_cases,
                         mode = 'lines+markers',
                         name = 'Weekly Tally of Confirmed Cases'))
fig.add_trace(go.Scatter(x = week, y = pak_weekly_recoveries,
                         mode = 'lines+markers',
                         name = 'Weekly Tally of Recoveries'))
fig.add_trace(go.Scatter(x = week, y = pak_weekly_deaths,
                         mode = 'lines+markers',
                         name = 'Weekly Tally of Deaths'))
fig.update_layout(title = "Weekly Tally of Cases, Recoveries and Deaths in Pakistan",
                  xaxis_title = "Week Number of 2020",yaxis_title = "Total Number of Cases",
                  legend = dict(x = 0,y = 1,traceorder = "normal"))
fig.show()


# ## 3. Forecasting
# 
# ******************TO BE UPDATED******************

# In[ ]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(df_national_pak)
model_fit = model.fit()

yhat = model_fit.predict(len(data), len(data))

