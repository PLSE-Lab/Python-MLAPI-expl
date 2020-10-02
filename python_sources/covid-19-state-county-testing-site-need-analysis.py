#!/usr/bin/env python
# coding: utf-8

# # Analysis Part I: Need for COVID-19 Testing Sites by State

# This analysis aims to determine where two pop-up testing centers should be built in the United States. A lower score in this scoring system indicates a higher need for testing centers. 
# 
# Part I of the analysis examines state-wide data to identify target states. Metrics (by state) used include median annual household income, median age, peak date, number of testing centers per capita, and number of testing centers per case. Only 26 states with peak dates that have not yet passed (as of April 15th) are examined in this analysis.
# 
# 
# 
# *Sources of data include NPR for peak dates, World Population Review for median age and population data, and the Kaiser Family Foundation for median annual household income data. The data on the total number of cases per state came from the Guardian and the data on the number of test sites per state came from the members of the Get Tested COVID-19 web scraping team.*
# 
# *Analysis completed by Yashi Sanghvi (Cornell '21) and Amanda Zhang (Yale '21) with Get Tested COVID-19.*

# In[ ]:


pip install datascience


# In[ ]:


#imports for use
import numpy as np
from datascience import *

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)


# In[ ]:


#loading in data with all scrapted test sites
sites_data=Table.read_table('../input/covid19-state-county-level-analysis/filtered_sites_masked.csv')
sites_data.group('State')


# In[ ]:


#load in dataset
full_dataset=Table.read_table('../input/covid19-state-county-level-analysis/COVID-19 Full Dataset.csv')


# In[ ]:


#organize peak dates data and create array
peaks = full_dataset.select('State', 'Days from April 15th')
peak_array = peaks.column('Days from April 15th')


# In[ ]:


#scoring based on peak dates
new_peak= make_array()
for x in peak_array:
    if x <= 7:
      new_peak=np.append(new_peak, 4)
    elif x <= 14:
      new_peak=np.append(new_peak, 3)
    elif x <= 21:
      new_peak=np.append(new_peak, 2)
    elif x >= 22:
      new_peak=np.append(new_peak, 1)
print(new_peak)


# In[ ]:


#organize median annual household income data and create array
income = full_dataset.select('State', 'Median Income')
income_array = income.column('Median Income')


# In[ ]:


#scoring based on median incomes
new_income= make_array()
for x in income_array:
    if x <= 50000:
      new_income=np.append(new_income, 1)
    elif x <= 60000:
      new_income=np.append(new_income, 2)
    elif x <= 70000:
      new_income=np.append(new_income, 3)
    else:
      new_income=np.append(new_income, 4)
print(new_income)


# In[ ]:


#organize median age data and create array
age = full_dataset.select('State', 'Median Age')
age_array = age.column('Median Age')


# In[ ]:


#scoring based on median age
new_age= make_array()
for x in age_array:
    if x <= 35:
      new_age=np.append(new_age, 3)
    elif x <= 40:
      new_age=np.append(new_age, 2)
    else: 
      new_age=np.append(new_age, 1)
print(new_age)


# In[ ]:


#calculating number of test sites per capita by state
#create sites and population arrays
sites_array = full_dataset.column('Number of Test Centers')
pop_array = full_dataset.column('Population')
#divide number of test sites by population
sites_per_capita = sites_array/pop_array
#adjust raw values into scores by multiplying by a factor of 1 million
adjusted_per_capita = sites_per_capita*1000000


# In[ ]:


#calculating number of test sites divided by total cases per state
#create total cases array
cases_array = full_dataset.column('Total Cases')
#divide number of test sites by total cases
sites_per_case = sites_array/cases_array
#adjust raw values into scores by multiplying by a factor of 1 thousand
adjusted_per_case = sites_per_case*1000


# In[ ]:


#create an array of alphabetized states
states_array = full_dataset.column('State')
#summed scores
summed_scores_1 = np.add(new_age, new_income)
summed_scores_2 = np.add(adjusted_per_capita, adjusted_per_case)
summed_scores_3 = np.add(summed_scores_1,summed_scores_2)
summed_scores = np.add(summed_scores_3,new_peak)


# In[ ]:


#create new table with scores
scored_table= Table().with_columns('State', states_array,'Peak Date Score', new_peak, 'Income Score', new_income, 'Age Score', new_age,
                          'Test Sites per Capita Score', adjusted_per_capita, 'Test Sites per Case Score', adjusted_per_case, 
                          'Summed Score', summed_scores)
sorted_scored_table = scored_table.sort('Summed Score')
sorted_scored_table


# From this sorted scoring table, we can see that the top 10 lowest scores (highest need) are Kansas, Minnesota, Georgia, Iowa, Oklahoma, Missouri, Texas, Massachusetts, Virgina, and Kentucky. More detailed county-level analysis will be conducted to identify zip-codes with the highest need within three of these ten states. These three states will be selected out of these 10 with consideration of unquantifiable factors along with these scores (e.g. Georgia's governor has recently reopened the state, likely increasing infection rates).

# # Analysis Part II: Need for COVID-19 Testing Sites by County for Selected States (MN, GA, MA)

# Part II of the analysis examines county-level data of three selected states, Minnesota (MN), Georgia (GA), and Massachusetts (MA) to identify target counties. Metrics (by county) used include median annual household income, median age, percent poverty, total COVID-19 cases per capita, and total COVID-19 deaths per capita.
# 
# **Reason Why MA was Selected:**
# Massachusetts' status as a coronavirus hotspot is confirmed by the climbing death rate. It is currently ranked [3rd in the number of COVID-19 cases](https://www.foxnews.com/health/massachusetts-new-coronavirus-hot-spot) in the US, and these numbers are continuing to rise. Furthermore, due to the high prevalence of crowded housing in minority communities, the risk of COVID-19 spreading Massachusetts only increases.
# 
# **Reason Why GA was Selected:** 
# With Georgia [re-opening businesses](https://www.nytimes.com/interactive/2020/04/24/opinion/coronavirus-covid-19-georgia-reopen.html), it is even more imperative that more test centers are established in Georgia. Georgia has extremely low testing rates, with only 1% of its citizens having been tested. Its testing rate continues to fall behind that of other states while the incidence of COVID-19 cases in Georgia continue to rise. With these increased number of cases and a high population of at-risk individuals, more testing will help to identify asymptomatic carriers in Georgia.
# 
# **Reason Why MN was Selected:** 
# Minnesota now records over [1 coronavirus-related death every hour](https://www.startribune.com/this-won-t-be-the-last-hard-day-minnesota-gov-tim-walz-says-as-daily-covid-deaths-hit-28/569963432/). With Minnesota's reopening of some workplaces, concerns of COVID-19 spreading are growing. Group-living housing, such as assisted-living facilities, are becoming hot spots for corona virus due to Minnesota's large elderly population. It is essential that there are more testing centers in Minnesota to get a better estimate of the true number of COVID-19 cases and to protect its population.
# 
# *All population, median income, and percent of population in poverty data was sourced from IndexMundi/US Census Bureau. GA median age data was sourced from the Federal Reserve Bank of St. Louis while GA total cases and deaths data was sourced from the GA Department of Public Health. MN total cases and deaths data was sourced from the MN Department of Health while MA total cases and deaths data was sourced from the NY Times. MN and MA median age data was sourced from LiveStories/US Census Bureau.*

# In[ ]:


#load in datasets for each state
mn_dataset=Table.read_table('../input/covid19-state-county-level-analysis/MN COVID-19 County Level Dataset.csv')
ga_dataset=Table.read_table('../input/covid19-state-county-level-analysis/GA COVID-19 County Level Dataset.csv')
ma_dataset=Table.read_table('../input/covid19-state-county-level-analysis/MA COVID-19 County Level Dataset.csv')

#filtered data sets for ga and mn (eliminate NaNs)
mn_dataset=mn_dataset.where('Total Cases', are.above(0))
ga_dataset=ga_dataset.where('Total Cases', are.above(0))

#create array of county names
mn_counties = mn_dataset.column('County')
ga_counties = ga_dataset.column('County')
ma_counties = ma_dataset.column('County')


# In[ ]:


#create relevant income arrays for each state
new_mn_income= make_array()
mn_income_arr = mn_dataset.column('Median Income')

new_ga_income= make_array()
ga_income_arr = ga_dataset.column('Median Income')

new_ma_income= make_array()
ma_income_arr = ma_dataset.column('Median Income')


# In[ ]:


#define function for scoring median incomes by county
def income_scorer (arr, new_arr):
  for i in arr:
      if i <= 30000:
        new_arr=np.append(new_arr, 1)
      elif i <= 50000:
        new_arr=np.append(new_arr, 2)
      elif i <= 70000:
        new_arr=np.append(new_arr, 3)
      elif i <= 90000:
        new_arr=np.append(new_arr, 4)
      else:
        new_arr=np.append(new_arr, 5)
  return new_arr


# In[ ]:


#store income scores in array
mn_income_scores = income_scorer(mn_income_arr,new_mn_income)
ga_income_scores = income_scorer(ga_income_arr,new_ga_income)
ma_income_scores = income_scorer(ma_income_arr,new_ma_income)


# In[ ]:


#create relevant age arrays for each state
new_mn_age= make_array()
mn_age_arr = mn_dataset.column('Median Age')

new_ga_age= make_array()
ga_age_arr = ga_dataset.column('Median Age')

new_ma_age= make_array()
ma_age_arr = ma_dataset.column('Median Age')


# In[ ]:


#define function for scoring median ages by county
def age_scorer (arr, new_arr):
  for i in arr:
      if i <= 30:
        new_arr=np.append(new_arr, 4)
      elif i <= 40:
        new_arr=np.append(new_arr, 3)
      elif i <= 50:
        new_arr=np.append(new_arr, 2)
      else:
        new_arr=np.append(new_arr, 1)
  return new_arr


# In[ ]:


#store age scores in array
mn_age_scores = age_scorer(mn_age_arr,new_mn_age)
ga_age_scores = age_scorer(ga_age_arr,new_ga_age)
ma_age_scores = age_scorer(ma_age_arr,new_ma_age)


# In[ ]:


#create relevant poverty arrays for each state
new_mn_poverty= make_array()
mn_poverty_arr = mn_dataset.column('% of Pop in Poverty')

new_ga_poverty= make_array()
ga_poverty_arr = ga_dataset.column('% of Pop in Poverty')

new_ma_poverty= make_array()
ma_poverty_arr = ma_dataset.column('% of Pop in Poverty')


# In[ ]:


#define function for scoring percent poverty by county
def poverty_scorer (arr, new_arr):
  for i in arr:
      if i <= 10:
        new_arr=np.append(new_arr, 6)
      elif i <= 15:
        new_arr=np.append(new_arr, 5)
      elif i <= 20:
        new_arr=np.append(new_arr, 4)
      elif i <= 25:
        new_arr=np.append(new_arr, 3)
      elif i <= 30:
        new_arr=np.append(new_arr, 2)
      else:
        new_arr=np.append(new_arr, 1)
  return new_arr


# In[ ]:


#store poverty scores in array
mn_poverty_scores=poverty_scorer(mn_poverty_arr,new_mn_poverty)
ga_poverty_scores=poverty_scorer(ga_poverty_arr,new_ga_poverty)
ma_poverty_scores=poverty_scorer(ma_poverty_arr,new_ma_poverty)


# In[ ]:


#define function to calculate adjusted total cases per capita (adjusted by 10,000 for scoring system)
def cases_adjuster(cases, pop):
  return (cases/pop)*10000

#store adjusted values
mn_cases_arr=cases_adjuster(mn_dataset.column('Total Cases'), mn_dataset.column('Population'))
ga_cases_arr=cases_adjuster(ga_dataset.column('Total Cases'), ga_dataset.column('Population'))
ma_cases_arr=cases_adjuster(ma_dataset.column('Total Cases'), ma_dataset.column('Population'))

#create new arrays
new_mn_cases=make_array()
new_ga_cases=make_array()
new_ma_cases=make_array()


# In[ ]:


#define function for scoring adjusted cases per capita by county
def cases_scorer (arr, new_arr):
  for i in arr:
      if i <= 15:
        new_arr=np.append(new_arr, 10)
      elif i <= 30:
        new_arr=np.append(new_arr, 9)
      elif i <= 45:
        new_arr=np.append(new_arr, 8)
      elif i <= 60:
        new_arr=np.append(new_arr, 7)
      elif i <= 75:
        new_arr=np.append(new_arr, 6)
      elif i <= 90:
        new_arr=np.append(new_arr, 5)
      elif i <= 105:
        new_arr=np.append(new_arr, 4)
      elif i <= 120:
        new_arr=np.append(new_arr, 3)
      elif i <= 135:
        new_arr=np.append(new_arr, 2)
      else:
        new_arr=np.append(new_arr, 1)
  return new_arr


# In[ ]:


#store cases scores in array
mn_cases_scores=cases_scorer(mn_cases_arr,new_mn_cases)
ga_cases_scores=cases_scorer(ga_cases_arr,new_ga_cases)
ma_cases_scores=cases_scorer(ma_cases_arr,new_ma_cases)


# In[ ]:


#define function to calculate total deaths per capita (adjusted by 100,000 for scoring system)
def deaths_adjuster(deaths, pop):
  return (deaths/pop)*100000

#store adjusted values
mn_deaths_arr = deaths_adjuster(mn_dataset.column('Total Deaths'), mn_dataset.column('Population'))
ga_deaths_arr = deaths_adjuster(ga_dataset.column('Total Deaths'), ga_dataset.column('Population'))
ma_deaths_arr = deaths_adjuster(ma_dataset.column('Total Deaths'), ma_dataset.column('Population'))

#create new arrays
new_mn_deaths=make_array()
new_ga_deaths=make_array()
new_ma_deaths=make_array()


# In[ ]:


#define function for scoring adjusted deaths per capita by county
def deaths_scorer (arr, new_arr):
  for i in arr:
      if i <= 15:
        new_arr=np.append(new_arr, 10)
      elif i <= 30:
        new_arr=np.append(new_arr, 9)
      elif i <= 45:
        new_arr=np.append(new_arr, 8)
      elif i <= 60:
        new_arr=np.append(new_arr, 7)
      elif i <= 75:
        new_arr=np.append(new_arr, 6)
      elif i <= 90:
        new_arr=np.append(new_arr, 5)
      elif i <= 105:
        new_arr=np.append(new_arr, 4)
      elif i <= 120:
        new_arr=np.append(new_arr, 3)
      elif i <= 135:
        new_arr=np.append(new_arr, 2)
      else:
        new_arr=np.append(new_arr, 1)
  return new_arr


# In[ ]:


#store deaths scores in array
mn_deaths_scores=cases_scorer(mn_deaths_arr,new_mn_deaths)
ga_deaths_scores=cases_scorer(ga_deaths_arr,new_ga_deaths)
ma_deaths_scores=cases_scorer(ma_deaths_arr,new_ma_deaths)


# In[ ]:


#define function to sum score
def summed_arrays(income_arr, age_arr, pov_arr, cases_arr, deaths_arr):
  sum_1=np.add(income_arr, age_arr)
  sum_2=np.add(cases_arr, deaths_arr)
  sum_3= np.add(sum_1, sum_2)
  return np.add(sum_3,pov_arr)

#store summed scores in array
mn_summed_scores=summed_arrays(mn_income_scores, mn_age_scores, mn_poverty_scores, mn_cases_scores, mn_deaths_scores)
ga_summed_scores=summed_arrays(ga_income_scores, ga_age_scores, ga_poverty_scores, ga_cases_scores, ga_deaths_scores)
ma_summed_scores=summed_arrays(ma_income_scores, ma_age_scores, ma_poverty_scores, ma_cases_scores, ma_deaths_scores)


# In[ ]:


#define function to create a table with per county scores for each state
def create_table(county_arr, income_arr, age_arr, pov_arr, cases_arr, deaths_arr, summed_arr):
  return Table().with_columns('County', county_arr, 'Income Score',income_arr, 'Age Score', age_arr, 'Poverty Score', pov_arr,'Cases Score', cases_arr, 'Deaths Score', deaths_arr, 'Summed Score', summed_arr)


# In[ ]:


#create table displaying Minnesota county scores
mn_table=create_table(mn_counties, mn_income_scores, mn_age_scores, mn_poverty_scores, mn_cases_scores, mn_deaths_scores, mn_summed_scores)
mn_table.sort('Summed Score')


# In[ ]:


#create table displaying Georgia county scores
ga_table=create_table(ga_counties, ga_income_scores, ga_age_scores, ga_poverty_scores, ga_cases_scores, ga_deaths_scores, ga_summed_scores)
ga_table.sort('Summed Score')


# In[ ]:


#create table displaying Massachusetts county scores
ma_table=create_table(ma_counties, ma_income_scores, ma_age_scores, ma_poverty_scores, ma_cases_scores, ma_deaths_scores, ma_summed_scores)
ma_table.sort('Summed Score')


# From this analysis, the top 10 counties in need all appear to be from GA, with the top three counties being Randolph, Early and Terrell. These results align with other data points suggesting that these are counties being highly impacted by the disease. All three of these counties are located in Southwest GA where [coronavirus is hitting hard](https://www.washingtonpost.com/nation/2020/04/26/coronavirus-southwest-georgia/?arc404=true) - in several of these counties, African Americans make up most of the population and about 30% of residents live in poverty. Of the [20 counties](https://www.statista.com/statistics/1109053/coronavirus-covid19-cases-rates-us-americans-most-impacted-counties/) in the nation with the most impacted by COVID-19, four are in southwest Georgia (Randolph is 6th, Terrell is 11th, Early is 13th, and Dougherty is 16th).
# 
# Suffolk (MA), the lowest scoring county out of the counties in Massachusetts and Minnesota, is ranked 25th on the list of counties most impacted by COVID-19.
