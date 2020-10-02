#!/usr/bin/env python
# coding: utf-8

# As the amount of infected numbers per day in Germany decreases, I thought it was time for me to take a look back from the beginning of COVID-19 to see how the virus has progressed. 
# 
# 
# 
# When comparing Germany COVID-19 statistics with Europe's it seems as though Germany has the right approach to keep the death rate and infected people relatively low . This is why I decided to do an exploratory analysis to see what insights I can gain and to see if other countries can learn from Germany. 
# 
# 
# 
# Sadly, this is not the end of COVID-19 but hopefully as the virus spreads through the rest of the world we can use data and science to decrease it's impact. 
# 
# 

# In[ ]:


## import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


## import data

cov = pd.read_csv('/kaggle/input/covid19-tracking-germany/covid_de.csv')
demo = pd.read_csv('/kaggle/input/covid19-tracking-germany/demographics_de.csv')


# In[ ]:


## covid-19 Germany cases per state

cov.head(10)


# In[ ]:


## general demographic information in germany states

demo.head(10)


# In[ ]:


cov.dtypes


# In[ ]:


#convert date to datetime for time series analysis 

d = cov["date"]
cov["date"] = pd.to_datetime(d)


# # Demographic Exploration 

# In[ ]:


demo.dtypes


# In[ ]:


gender_age = demo.groupby(['gender', 'age_group']).sum().reset_index()


# In[ ]:


gender_age.dtypes


# In[ ]:




fig = px.bar(gender_age, y='population', x='gender', color='age_group')
fig.update_layout(title='distribution of ages per gender')
fig.show()


# In[ ]:


state_age = demo.groupby(['state', 'age_group']).sum().reset_index()


# In[ ]:




fig = px.bar(state_age, y='population', x='state', color='age_group')
fig.update_layout(title='distribution of ages per state')
fig.show()


# - Baden-Wuerttemberg, Bayern, Nordrhein-Westfalen and Hessen have the highest amount of people > 60 , the largest amount of high risk individuals
# - Baden-Wuerttemberg, Bayern and Nordrhein-Westfalen are also the most populated

# # Germany Exploration
# 
# 

# In[ ]:


cov_cases = cov.groupby(['date']).sum().reset_index()


# In[ ]:


fig, ax = plt.subplots(figsize=(16,9))
ax.plot(cov_cases["date"],
        cov_cases["cases"],
        color="g");
ax.set_title("germany confirmed cases per day");
ax.spines["top"].set_visible(False);
ax.spines["right"].set_visible(False);


# In[ ]:


fig, ax = plt.subplots(figsize=(16,9))
ax.plot(cov_cases["date"],
        cov_cases["deaths"],
        color="r");
ax.set_title("germany confirmed deaths per day");
ax.spines["top"].set_visible(False);
ax.spines["right"].set_visible(False);


# In[ ]:


fig, ax = plt.subplots(figsize=(16,9))
ax.plot(cov_cases["date"],
        cov_cases["recovered"],
        color="b");
ax.set_title("germany recovered cases per day");
ax.spines["top"].set_visible(False);
ax.spines["right"].set_visible(False);


# # Age and gender Exploration

# In[ ]:


## group by gender 

cov_gen = cov.groupby(['gender']).sum().reset_index()


# In[ ]:


## calculate percentages for more comparable results

cov_gen["death_percentage"] = round(cov_gen["deaths"]/cov_gen["cases"] * 100,0)
cov_gen["cases_recovering"] = round(cov_gen["recovered"]/cov_gen["cases"] * 100,0)



# In[ ]:


cov_gen


# - The above table shows that overall females are more likely to get infected but less likely to die, in contrast males are less likely to get infected but more likely to die once infected. 

# In[ ]:


cov_age = cov.groupby(['age_group']).sum().reset_index()


# In[ ]:



cov_age["death_percentage"] = round(cov_age["deaths"]/cov_age["cases"]* 100,1)


# In[ ]:


cov_age


# - people aged between 0-34 have a near 0% chance of dying when infected 
# - people > 80 have a 23% chance of dying once infected 
# - the older you are the more chance you have of dying once infected

# In[ ]:


cov_age_gen = cov.groupby(['age_group','gender']).sum().reset_index()


# In[ ]:



cov_age_gen["death_percentage"] = round(cov_age_gen["deaths"]/cov_age_gen["cases"] * 100,1)
cov_age_gen["cases_per_age"] = round(cov_age_gen["cases"]/cov_age_gen["cases"].sum() * 100,1)


# In[ ]:


cov_age_gen


# - Females are more likely to get infected then males when they are > 80% but they are 11% less likely to die
# - In general, the numbers show that females are more likely to get infected but less likely to die.
# - The 35-59 age bracket is where the most people are infected but have a near 0.1%(F)-0.6%(M) chance of dying.

# # State Exploration
# 

# In[ ]:


state_df = cov.groupby(['state','date']).sum().reset_index()


# In[ ]:


state_df["state"].unique()


# In[ ]:


state_df["state"] = state_df["state"].str.replace("Baden-Wuerttemberg","Baden").str.replace("Mecklenburg-Vorpommern","Mecklenburg").str.replace("Nordrhein-Westfalen","Nordrhein").str.replace("Sachsen-Anhalt","Sachsen_A").str.replace("Schleswig-Holstein","Schleswig").str.replace("Rheinland-Pfalz","Rheinland")


# In[ ]:


listofstates = state_df["state"].unique()

## create for loop to split dfs by state for analysis 

listofdfs = []

for state in listofstates:
    locals()['df_' + state] = state_df[(state_df.state== state)]
    listofdfs.append(['df_'+ state][0])


# In[ ]:


listofdfs = [ df_Baden,
              df_Bayern,
              df_Berlin,
              df_Brandenburg,
              df_Bremen,
              df_Hamburg,
              df_Hessen,
              df_Mecklenburg,
              df_Niedersachsen,
              df_Nordrhein,
              df_Rheinland,
              df_Saarland,
              df_Sachsen,
              df_Sachsen_A,
              df_Schleswig,
              df_Thueringen]


# In[ ]:


def cumsum(df):
    df['cumsum_deaths'] = df["deaths"].cumsum()
    df['cumsum_cases'] = df["cases"].cumsum()
    df['cumsum_recovered'] = df["recovered"].cumsum()


# In[ ]:


for i in listofdfs:
    cumsum(i)


# In[ ]:


## merge df with cumsum figures

merged_df = pd.concat([df_Baden,
                            df_Bayern,
                            df_Berlin,
                            df_Brandenburg,
                            df_Bremen,
                            df_Hamburg,
                            df_Hessen,
                            df_Mecklenburg,
                            df_Niedersachsen,
                            df_Nordrhein,
                            df_Rheinland,
                            df_Saarland,
                            df_Sachsen,
                            df_Sachsen_A,
                            df_Schleswig,
                            df_Thueringen], axis=0)


# In[ ]:


merged_df


# # Cumulative Cases per State

# In[ ]:




fig = px.line(merged_df,
              x="date",
              y="cumsum_cases",
              color="state",
              line_group="state",
              hover_name="state")
fig.update_layout(
              title="cumulative cases per state",
              yaxis_title="cumulative_cases")

fig.show()


# - Western and Southern states of Germany have the first confirmed cases, Baden-Wuerttemberg, Bayern, and Nordrhein-Westfalen. All three have the highest population in Germany and therefore a greater opportunity to spread
# - Nordrhiein-Westfalen has the highest confirmed infection rates which can be traced back to a carnival
# 
# - There seems to be a correlation with the population and amount of people infected, the higher populated states have the most cases confirmed 

# In[ ]:


## create for loop for multiple graphs per state

for i in listofdfs:

    fig = px.line(i,
                  x="date",
                  y="cumsum_cases",
                  color="state",
                  line_group="state",
                  hover_name="state")
    fig.update_layout(
                  title="cumulative cases per state",
                  yaxis_title="cumulative_cases",)
    fig.show()


# # Cumulative deaths per State

# In[ ]:




fig = px.line(merged_df,
              x="date",
              y="cumsum_deaths",
              color="state",
              line_group="state",
              hover_name="state")
fig.update_layout(
              title="cumulative deaths per state",
              yaxis_title="cumulative_deaths"
)
fig.show()


#  - The most populated states Baden-Wuerttemberg, Bayern, and Nordrhein-Westfalen have the highest amount of deaths 

# In[ ]:


## create for loop for multiple graphs per state

for i in listofdfs:

    fig = px.line(i,
                  x="date",
                  y="cumsum_deaths",
                  color="state",
                  line_group="state",
                  hover_name="state")
    fig.update_layout(
                  title="cumulative deaths per state",
                  yaxis_title="cumulative_deaths",)
    fig.show()


# # Cumulative Recoveries Per State

# In[ ]:




fig = px.line(merged_df,
              x="date",
              y="cumsum_recovered",
              color="state",
              hover_name="state")
fig.update_layout(
              title="cumulative recoveries per state",
              yaxis_title="cumulative_recoveries"
          )

fig.show()


# In[ ]:


## create for loop for multiple graphs per state

for i in listofdfs:

    fig = px.line(i,
                  x="date",
                  y="cumsum_recovered",
                  color="state",
                  hover_name="state")
    fig.update_layout(
                  title="cumulative recoveries per state",
                  yaxis_title="cumulative_recoveries",)
    fig.show()


# # Conclusions
# 
# 
# - Females are more likely then males to get infected but less likely to die once infected, in contrast males are less likely then females to get infected but more likely to die once infected.
# - States with higher populations correlate to a higher infection rate and a higher death rate
# - States who reported some of their first cases in Germany such as Baden-Wuerttemberg, Bayern, and Nordrhein-Westfalen have the highest total number of cases.
# - States such as Bremen, Sachsen_Anhalt, and Mecklenburg  who were the last to report their first cases of COVID-19 have the lowest total number of cases.
# - Age brackets 15-34 and 35-59 account for 62% of total cases in Germany, with a death toll rate of near 1% for these age brackets this contributes to a very low overall death rate. 
# - Germans > 80 only account for 11% of the total cases with a higher amount of females being infected who have a higher recovery rate this also accounts for an overall low death rate. 
# 
# 
# 

# # Interesting follow up questions
# 
# - Are females more likely to get tested then males?
# - Why has Germany got a lower age average for infection then surrounding countries?
# - Why are females less likley to die once they have contracted the virus?

# # Reasons why the Germany is better off then other European Countries
# 
# * A robust health system
# * Track and trace
# * Low age average of the people contracting the virus
# * Heavy testing
# * Trust in government 

# # Interesting Links
# 
# * https://berlinspectator.com/2020/05/01/chronology-germany-and-the-coronavirus-2/
# * https://www.rki.de/EN/Home/homepage_node.html
# * https://www.worldometers.info/coronavirus/country/germany/
# 

# 
