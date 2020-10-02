#!/usr/bin/env python
# coding: utf-8

# Coronavirus disease (COVID-19) has greatly changed the lives of people around the world. Covid-19 spreads between people during close contact and can cause symptoms ranging from mild to very severe. According to the World Health Organization (WHO), there are no vaccines nor specific antiviral treatments for COVID-19. Recommended measures to prevent infection include frequent hand washing, maintaining physical distance from others, quarantine, covering coughs, and keeping unwashed hands away from the face. The US suffered the most from the virus (6/3/20 more than 1.8 million cases according to Johns Hopkins University). In this regard, many public places in the US have closed, and people are less likely to leave their homes. In this study, I will analyze how mobility of US residents has changed during quarantine days.

# # Mobility in the US with COVID-19

# ### Datasets sources:
# 
# https://www.kaggle.com/roche-data-science-coalition/uncover
# (covid tracking project)
# 
# https://www.apple.com/covid19/mobility
# 
# https://www.google.com/covid19/mobility
# 
# https://www.kaggle.com/lucasvictor/us-state-populations-2018

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Part 1
# ### Covid statistics by the US

# #### Reading Datasets and Pre-processing

# In[ ]:


covid_track_by_us = pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv")
covid_track_for_all_us = pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/covid_tracking_project/covid-statistics-for-all-us-daily-updates.csv")


# In[ ]:


pd.to_datetime(covid_track_for_all_us["date"])
pd.to_datetime(covid_track_by_us["date"])

#Because Google data begins on 2020-02-15, we will bring all the data to this form
#In addition, until this date in the table are NaN values that won't be helpful
covid_track_for_all_us = covid_track_for_all_us[covid_track_for_all_us["date"] > "2020-02-14"]
covid_track_by_us = covid_track_by_us[covid_track_by_us["date"] > "2020-02-14"]

sum_by_states = covid_track_by_us.groupby("state").sum().sort_values(by=["death"], ascending=False)
sum_by_states.reset_index(inplace=True)


# In[ ]:


states = pd.read_csv("/kaggle/input/usa-information/states.csv")
states_population = pd.read_csv("/kaggle/input/us-state-populations-2018/State Populations.csv")
states_population = pd.merge(left=states, right=states_population, on="State")

sum_by_states = pd.merge(left=sum_by_states, right=states_population, left_on='state', right_on='Abbreviation')
sum_by_states.drop(columns="Abbreviation", inplace=True)
sum_by_states.rename(columns={"State" : "statename"}, inplace=True)
sum_by_states["ratio of population to death"] = sum_by_states["2018 Population"]/sum_by_states["death"]


# #### EDA

# In[ ]:


fig, ax1 = plt.subplots(figsize=(21,10))

plt.title("COVID-19 Total Hospitalized and Deaths in the US", fontsize=40)

ax1.set_ylabel('Persons', color='black', fontsize=30)
ax1.tick_params(axis='y', labelcolor='black', labelsize=20, length=10, width=10)

ax1.bar(covid_track_for_all_us["date"].iloc[::-1], covid_track_for_all_us["death"].iloc[::-1], 
        color='tab:red', 
        label='Total Deaths, last: {} in {}'.format(covid_track_for_all_us["death"].iloc[0], covid_track_for_all_us["date"].iloc[0])
       )

ax2=ax1
ax2.plot(covid_track_for_all_us["date"].iloc[::-1], covid_track_for_all_us["hospitalized"].iloc[::-1], 
        color='tab:blue', 
        lw=8,
        label='Total Hospitalized, last: {} in {}'.format(covid_track_for_all_us["hospitalized"].iloc[0], covid_track_for_all_us["date"].iloc[0])
       )


ax1.legend(loc=1, bbox_to_anchor=(0.68,1), fontsize=30)
ax1.set_xlabel('Date', color='black', fontsize=30)
ax1.tick_params(axis='x', labelcolor='black', labelsize=30, length=10, width=10, rotation=45)


x = covid_track_for_all_us["date"].iloc[::-7]
ax1.set_xticks(x)
ax1.set_xticklabels(x)

fig.tight_layout()
plt.show();


# This graph shows that half of the hospitalized people do not survive. And the increase in the number of deaths and hospitalizations occurs linearly (approximately). Most likely, the linearity of this process is due to shelter-in-place order.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(21,10))

plt.title("COVID-19 daily Test positive results, Hospitalized and Deaths in the US", fontsize=40)
ax1.set_ylabel('Persons', color='black', fontsize=30)
ax1.tick_params(axis='y', labelcolor='black', labelsize=20, length=10, width=10)

ax1.bar(covid_track_for_all_us["date"].iloc[::-1], 
        covid_track_for_all_us["deathincrease"].iloc[::-1], 
        color='tab:red', 
        label='Deaths p/ day, last: {} in {}'.format(covid_track_for_all_us["deathincrease"].iloc[0], 
                                                     covid_track_for_all_us["date"].iloc[0]))


ax2 = ax1
ax2.plot(covid_track_for_all_us["date"].iloc[::-1], 
        covid_track_for_all_us["hospitalizedincrease"].iloc[::-1], 
        lw=8,
        color='tab:blue', 
        label='Hospitalized p/ day, last: {} in {}'.format(covid_track_for_all_us["hospitalizedincrease"].iloc[0], 
                                                           covid_track_for_all_us["date"].iloc[0]))


ax3 = ax1
ax3.plot(covid_track_for_all_us["date"].iloc[::-1], 
        covid_track_for_all_us["positiveincrease"].iloc[::-1], 
        lw=8,
        color='purple', 
        label='Test positive results p/ day, last: {} in {}'.format(covid_track_for_all_us["positiveincrease"].iloc[0], 
                                                           covid_track_for_all_us["date"].iloc[0]))

ax1.legend(loc=1, bbox_to_anchor=(0.68,1), fontsize=30)
ax1.set_xlabel('Date', color='black', fontsize=30)
ax1.tick_params(axis='x', labelcolor='black', labelsize=30, length=10, width=10, rotation=90)

x = covid_track_for_all_us["date"].iloc[::-7]
ax1.set_xticks(x)
ax1.set_xticklabels(x)

fig.tight_layout()
plt.show();


# Peaks and dips on the Test positive results curve associated with the days of the week (weekend decline). Deaths and hospitalizations remain at almost the same level. An interesting case is the peak on the hospitalization curve at 2020-05-01. And a smaller peak at 2020-05-08.

# In[ ]:


sum_by_states.sort_values(by="2018 Population", ascending=False)[["statename", "death", "2018 Population","ratio of population to death"]].style.bar(
    subset=["death", "ratio of population to death"], color='#d65f5f').hide_index()


# As we can see from the table, the most affected states in absolute numbers are New York, New Jersey, Michigan. Wyoming, Hawaii and Montana suffered in relative numbers.

# ## Part 2
# 
# ### Apple and Google mobility data

# ##### Apple
# 
# The data from Apple shows a relative volume of direction requests by type of transport (diving, walking or public transportation) made to Apple maps compared to a reference value in January 13, 2020.

# ##### Google
# 
# The mobility reports from Google uses percentage change of visits and permanence time to show tendences of displacement of people over time in different category of locations such as retail and leisure, markets and pharmacies, parks, public transport stations, workplaces and residential areas. The reference value used in its dataset is the median of the corresponding day of the week, over the five-week period from January 3 to February 6, 2020. According to Google, the most recent data represents approximately 2 3 days ago, which is the time needed to produce the data sets.

# #### Reading Datasets and Pre-processing

# In[ ]:


apple_m_d = pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/apple_mobility_trends/mobility-trends.csv").query('region == "United States"')
apple_m_d.drop(["geo_type", "region"], axis=1, inplace=True)
apple = pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/apple_mobility_trends/mobility-trends.csv").query('region == "United States"')

apple_m_d = pd.DataFrame(columns=["date"], data=apple["date"])
apple_m_d.dropna(inplace=True)

apple_m_d = pd.merge(left=apple_m_d, right=apple[apple["transportation_type"]=="driving"][["date","value"]].rename(columns={"value" : "apple_driving"}), how="left", on="date")
apple_m_d = pd.merge(left=apple_m_d, right=apple[apple["transportation_type"]=="walking"][["date","value"]].rename(columns={"value" : "apple_walking"}), how="left", on="date")
apple_m_d = pd.merge(left=apple_m_d, right=apple[apple["transportation_type"]=="transit"][["date","value"]].rename(columns={"value" : "apple_transit"}), how="left", on="date")

apple_m_d.drop_duplicates(inplace=True)

apple_m_d = apple_m_d[apple_m_d["date"] > "2020-02-14"]


# In[ ]:


google_m_d_states = pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/google_mobility/us-mobility.csv")

google_m_d_states.dropna(inplace=True)
google_m_d_mean = google_m_d_states[google_m_d_states["state"]=="Total"]
pd.to_datetime(google_m_d_mean["date"])
google_m_d_mean.drop(["state", "county"], axis=1, inplace=True)
google_m_d_mean.rename(columns={"retail" :  "google_retail_and_recreation", 
                           "grocery_and_pharmacy" : "google_grocery_and_pharmacy", 
                           "parks" : "google_parks",
                           "transit_stations" : "google_transit_stations",
                           "workplaces" : "google_workplaces",
                           "residential" : "google_residential"
                          },
                 inplace = True)


# In[ ]:


apple_and_google_m_d = pd.merge(left=google_m_d_mean, right=apple_m_d, how="left", on="date")
apple_and_google_m_d.drop_duplicates(inplace=True)


# #### EDA

# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))

ax = sns.heatmap(pd.DataFrame(apple_and_google_m_d).dropna().corr(), annot=True,
                 linewidths=.5, annot_kws={"size": 22})
sns.set(font_scale=3)
plt.title("Correlation Heatmap of Apple and Google data \n", fontsize = 35)
plt.legend(fontsize='xx-large')
plt.tick_params(axis='y', labelsize=25)
plt.tick_params(axis='x', labelsize=25)
plt.show();


# In[ ]:


fig, ax1 = plt.subplots(figsize=(21,15))

plt.title("Apple and Goggle Mobility Data", fontsize=40)

ax1.tick_params(axis='y', labelcolor='black', labelsize=35, length=10, width=10)
ax1.plot(apple_m_d["date"], apple_m_d["apple_driving"], 
         lw=5,
         color=(1,0,0, 0.5),
         label='Driving')

ax1.set_xlabel('Date', color='black', fontsize=30)
ax1.tick_params(axis='x', labelcolor='black', labelsize=30, length=10, width=10, rotation=75)

ax2=ax1

ax2.plot(apple_m_d["date"], apple_m_d["apple_transit"], 
         lw=5,
         color=(1,0,0, 0.3),
         label='Transit')

ax3=ax1

ax3.plot(apple_m_d["date"], apple_m_d["apple_walking"], 
         lw=5,
         color=(1,0,0, 0.7),
         label='Walking')
ax3.legend(loc=1, bbox_to_anchor=(0.6,0.9), fontsize=30)


####
ax4 = ax1.twinx()
ax4.tick_params(axis='y', labelcolor='black', labelsize=35, length=10, width=10)
ax4.plot(google_m_d_mean["date"], google_m_d_mean["google_retail_and_recreation"], 
         lw=5,
         color=(0.5,0,0.5, 0.5),
         label='Retail and recreation')

ax4.set_xlabel('Date', color='black', fontsize=30)
ax4.tick_params(axis='x', labelcolor='black', labelsize=30, length=10, width=10, rotation=75)


ax5=ax4
ax5.plot(google_m_d_mean["date"], google_m_d_mean["google_grocery_and_pharmacy"], 
         lw=5,
         color=(0.5,0,0.5, 0.3),
         label='Grocery and pharmacy')

ax6=ax4
ax6.plot(google_m_d_mean["date"], google_m_d_mean["google_parks"], 
         lw=5,
         color=(0.5,0,0.5, 0.7),
         label='Parks')

ax7=ax4
ax7.plot(google_m_d_mean["date"], google_m_d_mean["google_transit_stations"], 
         lw=5,
         color=(0,0,1, 0.7),
         label='Transit stations')

ax8=ax4
ax8.plot(google_m_d_mean["date"], google_m_d_mean["google_workplaces"], 
         lw=5,
         color=(0,0,1, 0.5),
         label='Workplaces')

ax9=ax4
ax9.plot(google_m_d_mean["date"], google_m_d_mean["google_residential"], 
         lw=5,
         color=(0,0,1, 0.3),
         label='Residential')
ax9.legend(loc=1, bbox_to_anchor=(0.3,0.3), fontsize=20)


x = google_m_d_mean["date"].iloc[::-7]
ax1.set_xticks(x)
ax1.set_xticklabels(x)

fig.tight_layout()
plt.show();


# Peaks and dips every seven days associated with the days of the week. More important is the decline in all graphs from 2020-03-09 to 2020-03-23 and further slow return to normal walking and driving indicators.

# Finally, we can combine Covid statistics data with Mobility data and see some patterns.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(21,15))

plt.title("Google Mobility Data and COVID-19 daily data in the US", fontsize=40)

ax1.set_ylabel('Persons', color='black', fontsize=30)
ax1.tick_params(axis='y', labelcolor='black', labelsize=20, length=10, width=10)

ax1.bar(covid_track_for_all_us["date"].iloc[::-1], 
        covid_track_for_all_us["deathincrease"].iloc[::-1], 
        color='tab:red', 
        label='Deaths p/ day, last: {} in {}'.format(covid_track_for_all_us["deathincrease"].iloc[0], 
                                                     covid_track_for_all_us["date"].iloc[0]))


ax2 = ax1
ax2.plot(covid_track_for_all_us["date"].iloc[::-1], 
        covid_track_for_all_us["hospitalizedincrease"].iloc[::-1], 
        lw=8,
        color="yellow", 
        label='Hospitalized p/ day, last: {} in {}'.format(covid_track_for_all_us["hospitalizedincrease"].iloc[0], 
                                                           covid_track_for_all_us["date"].iloc[0]))


ax3 = ax1
ax3.plot(covid_track_for_all_us["date"].iloc[::-1], 
        covid_track_for_all_us["positiveincrease"].iloc[::-1], 
        lw=8,
        color="green", 
        label='Test positive results p/ day, last: {} in {}'.format(covid_track_for_all_us["positiveincrease"].iloc[0], 
                                                           covid_track_for_all_us["date"].iloc[0]))

ax1.legend(loc=1, bbox_to_anchor=(0.68,1), fontsize=20)
ax1.set_xlabel('Date', color='black', fontsize=30)
ax1.tick_params(axis='x', labelcolor='black', labelsize=30, length=10, width=10, rotation=90)

###
ax4 = ax1.twinx()
ax4.tick_params(axis='y', labelcolor='black', labelsize=35, length=10, width=10)
ax4.plot(google_m_d_mean["date"], google_m_d_mean["google_retail_and_recreation"], 
         lw=5,
         color=(0.5,0,0.5, 0.5),
         label='Retail and recreation')

ax4.set_xlabel('Date', color='black', fontsize=30)
ax4.tick_params(axis='x', labelcolor='black', labelsize=30, length=10, width=10, rotation=75)


ax5=ax4
ax5.plot(google_m_d_mean["date"], google_m_d_mean["google_grocery_and_pharmacy"], 
         lw=5,
         color=(0.5,0,0.5, 0.3),
         label='Grocery and pharmacy')

ax6=ax4
ax6.plot(google_m_d_mean["date"], google_m_d_mean["google_parks"], 
         lw=5,
         color=(0.5,0,0.5, 0.7),
         label='Parks')

ax7=ax4
ax7.plot(google_m_d_mean["date"], google_m_d_mean["google_transit_stations"], 
         lw=5,
         color=(0,0,1, 0.7),
         label='Transit stations')

ax8=ax4
ax8.plot(google_m_d_mean["date"], google_m_d_mean["google_workplaces"], 
         lw=5,
         color=(0,0,1, 0.5),
         label='Workplaces')

ax9=ax4
ax9.plot(google_m_d_mean["date"], google_m_d_mean["google_residential"], 
         lw=5,
         color=(0,0,1, 0.3),
         label='Residential')
ax9.legend(loc=1, bbox_to_anchor=(0.3,0.3), fontsize=20)


x = google_m_d_mean["date"].iloc[::-7]
ax1.set_xticks(x)
ax1.set_xticklabels(x)

fig.tight_layout()
plt.show();


# As we can see from the summary graph, from 2020-03-11 to 2020-04-04 there is a sharp increase in the positive results of COVID-19 tests. This is partly due to the availability of tests. Around the same time, the number of stays in the resident is growing, when other activities are falling.

# ## Conclusion

# A stable number of severe cases and deaths from coronavirus, most likely indicates social isolation efficiency and timely testing. This is mentioned in many scientific publications.

# # Thanks
