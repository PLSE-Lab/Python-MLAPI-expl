#!/usr/bin/env python
# coding: utf-8

# # Step 1. Acquiring the data

# On this project I will use World Indicator Datasets

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../input/Indicators.csv')
data.shape


# # Step 2. Data preparation

# We need to clean up the data that we use for this research. these are the objectives:
# 
# 1. Data must be comparable, means that both death rate and health expenditure should have the same amount of data.
# 2. We only use those countries that actively increasing their health expenditure. We will analyze if by increasing the health expenditure will also decreasing its death rate / mortality rate. We use 5 years span to reduce bias on fluctuating values of health expenditure. Why 5 years? https://en.wikipedia.org/wiki/First_five-year_plan

# In[ ]:


health_gdp_indicator_name = 'Health expenditure, total \(% of GDP\)'
health_gdp_indicator_mask = data['IndicatorName'].str.contains(health_gdp_indicator_name)
health_gdp_indicator_df = data[health_gdp_indicator_mask]
health_gdp_indicator_df.head()


# We need to find out the year range of this data:

# In[ ]:


print(health_gdp_indicator_df.Year.min(),' to ',health_gdp_indicator_df.Year.max())


# In[ ]:


death_indicator_name = 'Death rate, crude \(per 1,000 people\)'
death_indicator_mask = data['IndicatorName'].str.contains(death_indicator_name)
death_indicator_df = data[death_indicator_mask]
death_indicator_df.head()


# *The crude death rate is the number of deaths occurring among the population of a given geographical area during a given year, per 1,000 mid-year total population of the given geographical area during the same year.*

# 
# We need to find out the year range of this data:
# 

# In[ ]:


print(death_indicator_df.Year.min(),' to ',death_indicator_df.Year.max())


# So we have different number of data between health GDP and the death rate, so we just use death indicator data above year 1994 to conduct this research

# In[ ]:


death_indicator_df_reduced = death_indicator_df[death_indicator_df['Year'] > 1994]


# lets see if we have same number of data now:

# In[ ]:


death_indicator_df_reduced.shape


# In[ ]:


health_gdp_indicator_df.shape


# The data is still in different number of rows, lets check the number of countries
# 

# In[ ]:


len(death_indicator_df_reduced['CountryName'].unique().tolist())


# In[ ]:


len(health_gdp_indicator_df['CountryName'].unique().tolist())


# OK, we have different number of countries. So we need to find countries that exists in both dataframes, lets pick the countries

# In[ ]:


countries1 = np.array(death_indicator_df_reduced['CountryName'].unique().tolist())
countries2 = np.array(health_gdp_indicator_df['CountryName'].unique().tolist())
both_countries = np.intersect1d(countries1, countries2)

dt_country_mask = death_indicator_df_reduced['CountryName'].isin(both_countries)
hg_country_mask = health_gdp_indicator_df['CountryName'].isin(both_countries)

death_indicator_df_cropped = death_indicator_df_reduced[dt_country_mask]
health_gdp_indicator_df_cropped = health_gdp_indicator_df[hg_country_mask]


# In[ ]:


len(death_indicator_df_cropped['CountryName'].unique().tolist())


# In[ ]:


len(health_gdp_indicator_df_cropped['CountryName'].unique().tolist())


# OK now we have same number of countries in both data, let's call shape again to check number of rows:

# In[ ]:


death_indicator_df_cropped.shape


# In[ ]:


health_gdp_indicator_df_cropped.shape


# Oh no... we still have different number of rows, let's check if the data covers the whole period between 1995 to 2013 

# In[ ]:


dt_year_count = death_indicator_df_cropped.groupby('Year').count()
dt_year_count[['CountryName','CountryCode']]


# In[ ]:


hg_year_count = health_gdp_indicator_df_cropped.groupby('Year').count()
hg_year_count[['CountryName','CountryCode']]


# Hmmm.. the number of countries on each year is not always the same!
# 
# This means we have to find out which countries that have complete historical data from 1995 to 2013
# 
# Let's define the helper functions first:

# In[ ]:


def does_it_has_complete_history(country_name,df):
    mask1 = df['CountryName'].str.contains(country_name) 
    mask2 = df['Year'].isin(df['Year'].unique().tolist())
    # apply our mask
    full = df[mask1 & mask2]
    if len(full['Year']) == len(df['Year'].unique().tolist()):
        return True
    else:
        return False

def does_the_country_increasing(country_name,df):
    values = df[df.CountryName == country_name]['Value']
    values = values[0::5] # only check the difference in 5 years span
    if strictly_increasing(values):
        return True
    else:
        return False

def does_the_country_decreasing(country_name,df):
    values = df[df.CountryName == country_name]['Value']
    values = values[0::5] # only check the difference in 5 years span
    if strictly_decreasing(values):
        return True
    else:
        return False

#referenced from https://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


# Now let's check which countries that exists on both datasets and the country is **increasing its health expenditure per 5 years**

# In[ ]:


import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups') # need to silence this warning :)

country_that_decreasing = []
country_that_increasing = []
country_that_has_history = []
for country in both_countries:
    dt_has_history = does_it_has_complete_history(country,death_indicator_df_cropped)
    hg_has_history = does_it_has_complete_history(country,health_gdp_indicator_df_cropped)
    if dt_has_history & hg_has_history: # only country that exists on both dataframe
        country_that_has_history.append(country)
        hg_increasing = does_the_country_increasing(country, health_gdp_indicator_df_cropped)
        if hg_increasing:
            country_that_increasing.append(country)
            hg_decreasing = does_the_country_decreasing(country, health_gdp_indicator_df_cropped)
            if hg_decreasing:
                country_that_decreasing.append(country)
        
    


# Let's check which countries are actively increasing their health expenditure per 5 years:

# In[ ]:


len(country_that_increasing)


# We have 57 countries that actively increasing their health expenditure per 5 years.
# 
# Let's check which countries are decreasing their health expenditure per 5 years:

# In[ ]:


len(country_that_decreasing)


# We have 0 countries that actively decreasing their health expenditure per 5 years. 
# 
# Let's check the number of countries that appear on both datasets. The countries can be a combination of those countries who are increasing, decreasing, or not significantly changing their health expenditure. We need this to calculate the world average health expenditure.

# In[ ]:


len(country_that_has_history)


# OK we got 192 countries.
# 
# Now let's see which countries are actively increasing their health expenditure:

# In[ ]:


mask = health_gdp_indicator_df_cropped['CountryName'].isin(country_that_increasing)
health_gdp_indicator_df_cropped[mask].head(60)


# Nice! My country... **Indonesia** is one of them, so lets do some sanity check here...
# 
# Let's check if Indonesia is appear in both data frame, and has increasing value of health expenditure per 5 years:

# In[ ]:


values = health_gdp_indicator_df_cropped[health_gdp_indicator_df_cropped.CountryName == 'Indonesia'][['Year','Value']]
values


# Positive, Indonesia is increasing its health expenditure per 5 years, now lets check if it also appears in death rate data frame

# In[ ]:


values = death_indicator_df_cropped[death_indicator_df_cropped.CountryName == 'Indonesia'][['Year','Value']]
values


# Positive, it also appears in death rate dataframe
# 
# OK let's redo what we did before, but this time we use only countries that exists on both dataframes, and increasing its expenditure per 5 years

# In[ ]:


# those that only increasing their health expenditure
dt_country_mask_2 = death_indicator_df_cropped['CountryName'].isin(country_that_increasing)
hg_country_mask_2 = health_gdp_indicator_df_cropped['CountryName'].isin(country_that_increasing)

death_indicator_clean = death_indicator_df_cropped[dt_country_mask_2]
health_gdp_indicator_clean = health_gdp_indicator_df_cropped[hg_country_mask_2]

# all countries that has complete death rate and health expenditure history from 1995 to 2013
dt_country_mask_3 = death_indicator_df_cropped['CountryName'].isin(country_that_has_history)
hg_country_mask_3 = health_gdp_indicator_df_cropped['CountryName'].isin(country_that_has_history)

death_indicator_all = death_indicator_df_cropped[dt_country_mask_3]
health_gdp_indicator_all = health_gdp_indicator_df_cropped[hg_country_mask_3]


# In[ ]:


death_indicator_clean.shape


# In[ ]:


health_gdp_indicator_clean.shape


# In[ ]:


death_indicator_all.shape


# In[ ]:


health_gdp_indicator_all.shape


# In[ ]:


death_indicator_clean_years = death_indicator_clean.groupby('Year').count()
death_indicator_clean_years[['CountryName','CountryCode']]


# In[ ]:


health_gdp_indicator_clean_years2 = health_gdp_indicator_clean.groupby('Year').count()
health_gdp_indicator_clean_years2[['CountryName','CountryCode']]


# In[ ]:


death_indicator_all_years = death_indicator_all.groupby('Year').count()
death_indicator_all_years[['CountryName','CountryCode']]


# In[ ]:


health_gdp_indicator_all_years2 = health_gdp_indicator_all.groupby('Year').count()
health_gdp_indicator_all_years2[['CountryName','CountryCode']]


# **BINGO!!!**, now we have sychronized country indicator data for death rate and health expenditure

# # Step 3. Data analyzing & visualization

# OK now that we have the clean data, its time to analyze it

# Let's find the world average health expenditure over time:

# In[ ]:


health_gdp_indicator_all_mean = health_gdp_indicator_all.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(health_gdp_indicator_all_mean['Year'].values, health_gdp_indicator_all_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(health_gdp_indicator_all.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('World\'s yearly average of ' + health_gdp_indicator_all.iloc[0].IndicatorName )

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,9])
plt.show()


# In the figure above, we see that most countries are increasing their health expenditure.
# 
# OK now let's see if this also impacted the death rate on average:
# 

# In[ ]:


death_indicator_all_mean = death_indicator_all.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(death_indicator_all_mean['Year'].values, death_indicator_all_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(death_indicator_all.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('World\'s yearly average of  ' + death_indicator_all.iloc[0].IndicatorName)

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,11])
plt.show()


# In the figure above, we can clearly see that on average, the death rate is also decreasing. let's see the correlation on scatter plot:

# In[ ]:


fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_title('Death rate vs. health expenditure (world yearly average)',fontsize=10)
axis.set_xlabel('Yearly average of ' + health_gdp_indicator_all.iloc[0].IndicatorName,fontsize=10)
axis.set_ylabel('Yearly average of ' + death_indicator_all.iloc[0].IndicatorName,fontsize=9)

X = health_gdp_indicator_all_mean['Value']
Y = death_indicator_all_mean['Value']

fig.dpi = 110
fig.figsize = (10,5)
axis.scatter(X, Y)
plt.show()


# In[ ]:


np.corrcoef(health_gdp_indicator_all_mean['Value'],death_indicator_all_mean['Value'])


# It has a strong negative correlation, now let's see the relationship on those countries that actively increasing their health expenditure. If our main hypotesis is correct, then we should have a stronger negative correlation value.

# In[ ]:


death_indicator_clean_mean = death_indicator_clean.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(death_indicator_clean_mean['Year'].values, death_indicator_clean_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(death_indicator_clean.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('Yearly average of ' + death_indicator_clean.iloc[0].IndicatorName + ' on country that increases its health expenditure')

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,11])
plt.show()


# In[ ]:


health_gdp_indicator_clean_mean = health_gdp_indicator_clean.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(health_gdp_indicator_clean_mean['Year'].values, health_gdp_indicator_clean_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(health_gdp_indicator_clean.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('Yearly average of ' + health_gdp_indicator_clean.iloc[0].IndicatorName + ' on country that increases its health expenditure')

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,9])
plt.show()


# Let's draw a scatter plot of both data:

# In[ ]:


fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_title('Death rate vs. health expenditure yearly average on 57 countries',fontsize=10)
axis.set_xlabel('Yearly average of ' + health_gdp_indicator_clean.iloc[0].IndicatorName,fontsize=10)
axis.set_ylabel('Yearly average of ' + death_indicator_clean.iloc[0].IndicatorName,fontsize=9)

X = health_gdp_indicator_clean_mean['Value']
Y = death_indicator_clean_mean['Value']

fig.dpi = 110
fig.figsize = (10,5)
axis.scatter(X, Y)
plt.show()


# This look like a strong negative relationship. let's see the correlation:

# In[ ]:


np.corrcoef(health_gdp_indicator_clean_mean['Value'],death_indicator_clean_mean['Value'])


# -0.97332126 Is a stronger negative relationship than the world average of -0.95838109

# # Step 4. Report the findings

# ***1. The higher spending of health expenditure of a country leads to lower death rate on that country.***
# 
# ***2. Only 57 countries that actively increasing their health expenditure (per 5 years time)***
# 
# ***3. No country is decreasing their health expenditure (per 5 years time)***
# 
# ***4. On average, most countries are increasing their health expenditure over time***
