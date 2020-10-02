#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# ![corona](https://amazonia.fiocruz.br/wp-content/uploads/2020/03/Coronavirus-2.jpg)
# 
# The aim of this notebook is to study some of the numerical data we have for the US states, and specially the data for New York. Most of us know about NY's situation which is one of the worst in the world until now, and analysing the data may clarify a few things. 
# 
# We will mainly use three datasets:
# 
# - Urgent Care Facilities
# - US County Healthcare Rankings 2020
# - Covid Sources For Counties 

# In[ ]:


import numpy as np
import pandas as pd

covid_stats = pd.read_csv("/kaggle/input/uncover/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv", dtype={"fips": str})
health_rankings = pd.read_csv("/kaggle/input/uncover/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv")
icu_facilities = pd.read_csv("/kaggle/input/uncover/hifld/hifld/urgent-care-facilities.csv")


# Before going any further, we should check all the fields that are available for each one of the datasets. Since they are all dimensionally large, not all the fields appear when we `head` them, and even if we change the display configuration by setting `pd.set_option('display.max_columns', 100)`, it is still not that intuitive. So I'll just check the column names so all of them can be visualized at the same time.

# In[ ]:


print("COVID STATS COLUMNS:")
print(covid_stats.keys())
print("========================================================================")
print("HEALTH RANKINGS COLUMNS:")
print(health_rankings.keys())
print("========================================================================")
print("ICU FACILITIES COLUMNS:")
print(icu_facilities.keys())
print("========================================================================")


# Show the cell below to see all the keys of health rankings, as there are so many that not even the method above was able to get them all.

# In[ ]:


for key in health_rankings.keys():
    print(key)


# I'm more interested in just a few variables from health rankings, so I'll just get the ones I need, or the ones I think could be useful at some point, in a new dataframe.

# In[ ]:


columns = ["fips", "state", "county", "num_deaths", "years_of_potential_life_lost_rate", 
          "percent_fair_or_poor_health", "average_number_of_physically_unhealthy_days",
          "average_number_of_mentally_unhealthy_days", "percent_smokers", "num_primary_care_physicians",
          "preventable_hospitalization_rate", "num_unemployed", "labor_force", "income_ratio", "num_households",
          "overcrowding", "life_expectancy"]

health_rankings_selected = health_rankings[columns]


# It is important to deal with null values in the dataset, and covid_stats has a considerable amount of them. To do so, I'll impute all the nulls for 0 values. This may be a source of error, as it is not clear why those values are null, but it seems that there are just not any registred cases in such states at the moment. 

# In[ ]:


covid_stats.fillna(0)


# Now let's explore each dataset one by one, so that we'll get an understanding of the data before trying to put all of them combined.
# 
# ## The virus and the health system of the US 
# 

# In[ ]:


import plotly.graph_objects as go

covid_stats_last = covid_stats[covid_stats["date"] == "2020-03-30"].copy()

fig = go.Figure(data=go.Choropleth(
    locations=covid_stats_last['state'], # Spatial coordinates
    z = covid_stats_last['positive'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = "Reds",
    colorbar_title = "# positives",
))

fig.update_layout(
    title_text = 'Number of positive tests per US state',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# It is clear from the above that, as we all know from the news, New York is the new epicenter of the pandemic. All the other states have way less infected people, but the numbers are just small by comparison with NY - those numbers are huge for a disease like this one. In comparison, let's check the distribution of ICU units across the US. 
# 
# The absolute number of units isn't very helpful, as there are states with more or less people than others, and so we expect there to be more or less ICUs, respectively. Therefore we should investigate the ratio of units per a given number of inhabitants. In this case, I'll use 100,000. The population per state data is from 2017 US Census, found here at Kaggle.

# In[ ]:


icus_per_state = pd.DataFrame(icu_facilities.groupby(["state"])["id"].nunique()).reset_index()

us_census = pd.read_csv("../input/us-census-demographic-data/acs2017_county_data.csv")

# abbreviation and full state name csv
states_abbrev = pd.read_csv("https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv")

# merging the abbreviations within the census dataset 
us_census = us_census.merge(states_abbrev, on = "State")
us_census = us_census.rename(columns = {"State": "fullState", "Abbreviation": "state"})

#population per state
popState = us_census.groupby("state")["TotalPop"].sum().reset_index()

icupopdf = icus_per_state.merge(popState, on = "state")

icupopdf["ICUPopRatio"] = icupopdf["id"]/icupopdf["TotalPop"] * 100000

fig = go.Figure(data=go.Choropleth(
    locations=icupopdf['state'], # Spatial coordinates
    z = icupopdf['ICUPopRatio'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = "Blues",
    colorbar_title = "# of ICUs per 100k people",
))

fig.update_layout(
    title_text = 'ICU density per state',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# From the above map it is clear the discrepancy between the number of ICU units between states and that in the state that it is most needed - New York -, the number of ICUs per hundred thousend people is very small (0.71). Only Pennsylvania has a smaller density (0.47). Therefore the situation in New York is critical not only because the virus has spread more there, but also because there are very few ICU units for those who need it, which eventually will lead to more deaths. That's why it is necessary to build hospitals just to treat COVID-19 pacients, and that [seems to be happening](https://www.nytimes.com/2020/03/25/nyregion/nyc-coronavirus-hospitals.html).
# ___
# 
# ## COVID in New York
# We know from the news (even if you are not from the US, such as myself) and we've seen in this notebook that New York is critical. So I think it is worth to use the data we have to check the evolution of the virus in this city. First, one important thing to check it the rate of increase in the number of positive cases. 

# In[ ]:


import plotly.express as px

fig = px.line(covid_stats[covid_stats["state"] == "NY"], x="date", y="positiveincrease", title='Daily increase in COVID-19 cases')
fig.show()


# The number of daily increases is not yet in its peak. There is a small decrease by the end of March, but we need more data in order to confirm that the spread has started to slow down. With the available data, this downwards movement could be just like in March 24 in the graph. This is, however, the "speed" of the virus. Even if it slows down, the number of cases are still increasing. So it is important to check the number of total positive per day.

# In[ ]:


fig = px.line(covid_stats[covid_stats["state"] == "NY"], x="date", y="positive", title='Total increase in COVID-19 cases in New York')
fig.show()


# As espected from the previous plot, the Positive line is still increasing and has not seen the peak by the day of the last datapoint (30-March-2020).
# ___
# 
# As many of us already know, the issue is not the disease by itself, as most of the people who is contaminated with the virus doesn't present harsh health problems. The greatest issue is the number of hospitals - and more precisely, ICUs - the state has available to treat those that do indeed need a more intensive treatment. The less of such units we have available, the more people who end up really sick don't get the required treatment, and the number of deaths increase. So, in a certain aspect, it is all about having hospitals and equipment to treat them all. We already have information about the ICU units, but we still didn't see how many people are actually in the hospital. So we'll commit to that investigation.

# In[ ]:


covid_ny = covid_stats[covid_stats["state"] == "NY"].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_ny["date"], y=covid_ny["hospitalized"],
                    mode='lines',
                    name='Hospitalized'))
fig.add_trace(go.Scatter(x=covid_ny["date"], y=covid_ny["death"],
                    mode='lines',
                    name='Deaths'))


# We don't have data for dates before March 15 for deaths and before March 21 for hospitalized cases. This is perhaps because the deaths started on that day - I couldn't find news about the first death registry, but maybe someone the US can confirm it -, and I'm not sure why hospitalized ones weren't registred before that. 
# 
# We see the increase in both numbers, but the the hospitalized patients increase is steepest. Moreover, we can notice a clear, sudden increase from day 24 to 25. Coincidentally or not, this is on the same time windows the positive results restarted to increase in the Daily Increase graph. Of course, as the number of sick people increase, we expect the same to the number of hospitalized cases, but there may be a reason related to both variables that happened as well. Not only that, but the number of hospitalized people increased faster than the number of daily increases. 
