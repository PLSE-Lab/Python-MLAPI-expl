#!/usr/bin/env python
# coding: utf-8

# <span style='color:Red ; font-size: 250%'> Benford's Law - A tool to detect data manipulation on confirmed cases of Covid-19 (Part 1 of 3) </span> 
# #### Author: [Rafael Klanfer Nunes](https://www.linkedin.com/in/rafaelknunes/)
# #### **Date**: 11/apr/2020
# #### **Data Source**: [Data Repository by Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)
# #### **Disclaimer**: In this first part we will be analyzing the numbers of confirmed cases of Covid-19. Look for parts 2 and 3 for an investigation on the number of deaths and recovered cases of Covid-19.
# #### **KAGGLE Notebook (Part 1: confirmed)**: https://www.kaggle.com/rafaelknunes/benford-law-to-detect-covid-19-manipulation-1of3
# #### **KAGGLE Notebook (Part 2: recovered)**: https://www.kaggle.com/rafaelknunes/benford-law-to-detect-covid-19-manipulation-2of3
# #### **KAGGLE Notebook (Part 3: deaths)**: https://www.kaggle.com/rafaelknunes/benford-law-to-detect-covid-19-manipulation-3of3

# In[ ]:


import rkn_module_benford_law as rkn_benford

import sys
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# # 1. Notebook Goals
# 
# * 1) Using Benford's Law theory, test whether the numbers of confirmed Covid-19 cases informed by governments are experiencing some kind of manipulation.
# * 2) This work uses the following utility script: "RKN Module - Benford Law". For more on how to use this script, please access:
# https://www.kaggle.com/rafaelknunes/rkn-module-benford-law-tutorial/notebook

# # 2. Loading the DataSet

# In[ ]:


path = "../input/inputbenfordcovid19/time_series_covid19_confirmed_global.csv"

data = pd.read_csv(path, encoding='ISO-8859-1', delimiter=',' , low_memory=False)

data_mod = data.copy()

df = pd.DataFrame(data_mod)


# # 3. Analysis on the Original Data Set
# 
# In this section we want to analyze the data for confirmed cases took from [Data Repository by Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19). This dataset shows the number of confirmed cases of Covid-19 per country and per country's state when possible. The period of analysis goes from 22/jan/2020 to 31/mar/2020.
# 
# Then, let's check for:
# 
# * Number of NaN in each variable
# * Variable types
# * Variable categories

# In[ ]:


# df to store the analysis
df_analysis = pd.DataFrame(columns=['var_name', 'var_NaN', 'var_not_NaN', "var_min", "var_max" , "var_mean" , 'var_type', 'var_categ'])


# In[ ]:


# Column names in dataSet
coluna_name_list = list(df.columns.values)


# In[ ]:


# This routine will store on df_analise the number os NaN each variable from dataset has. Alto the variable type and its categories.
for i in coluna_name_list:
    if(df[i].dtypes == "object"):
        lista=[i, df[i].isna().sum(), df[i].count(), "NA", "NA", "NA", df[i].dtypes, "numerical variable"]
        df_length = len(df_analysis)
        df_analysis.loc[df_length] = lista
    else:
        lista=[i, df[i].isna().sum(), df[i].count(), df[i].min(), df[i].max(), df[i].mean(), df[i].dtypes, "numerical variable"]
        df_length = len(df_analysis)
        df_analysis.loc[df_length] = lista


# In[ ]:


# Set var_name as index
df_analysis.set_index('var_name', inplace=True)


# In[ ]:


# For each non numerical variable assign its possible values
for i in coluna_name_list:
    if(df_analysis.loc[i, "var_type"] == "object"):
        df_analysis.loc[i, "var_categ"] = list(df[i].unique())
    else:
        pass


# In[ ]:


df_analysis.sort_values("var_NaN", ascending=False)


# # 4. Creating modified data sets
# 
# For this section we want to aggregate the daily numbers into weekly numbers. This aggregation is important since many cases that happen in one day may only be informed some days later. 
# 
# We also gonna concatenate the country name and the country state into a unique column named: Country_State
# 
# Then, we will produce two excel files:
# 
# * **df_merge_confirmed.xlsx**: This file contains the number of new cases per day, per week and the accumulated up to the date. These numbers are shown per country and per day (from 22/jan/2020 to 31/mar/2020). This file will not be used on the work, but may be useful for researchers.
# 
# * **df_week_confirmed.xlsx**: This file contains in each row the name of the Country/State plus the number of confirmed new cases of Covid-19 for each week from 22/jan/2020 to 31/mar/2020. This is the data that will be used further for the Benford's analysis.

# In[ ]:


# Create a new column with country names. If a country has no provinces/state desaggregation, so it will show the term: Single Unity
df["Province/State"] = df["Province/State"].replace(np.NaN, "Single Unity")
df["Country_State"] = df["Country/Region"] + "_" + df["Province/State"]


# In[ ]:


# Remove unecessary columns
del df["Province/State"]
del df["Country/Region"]
del df["Lat"]
del df["Long"]


# In[ ]:


df


# In[ ]:


# Send columns to rows
df_melt = df.melt(id_vars=["Country_State"],
       var_name="Date",
       value_name="Confirmed_Accumulated")


# In[ ]:


# Assign date type
df_melt['Date'] = pd.to_datetime(df_melt['Date'])


# In[ ]:


# Sort by country and date
df_melt = df_melt.sort_values(["Country_State", "Date"], ascending = (True, True))


# In[ ]:


# New column to receive the new cases for each day
df_melt["Confirmed_New_Day"] = 0
# This function will assign to each country/day the number of new cases confirmed, based on the difference among accumulated cases of the actual and last day.
country_before = df_melt.iloc[0,0]

for row in range(1, df_melt.shape[0], 1):
    country_actual = df_melt.iloc[row,0]
    if(country_actual == country_before):
        df_melt.iloc[row,3] = df_melt.iloc[row,2] - df_melt.iloc[row-1,2]
    else:
        df_melt.iloc[row,3] = df_melt.iloc[row,2]
    country_before = country_actual


# In[ ]:


# Reset index: drop = False
df_melt.reset_index(inplace = True, drop = True)


# In[ ]:


# Create column to store de number of the week
df_melt['Date_week'] = pd.DatetimeIndex(df_melt['Date']).week


# In[ ]:


# New dataFrame grouped by country and week. The column value represents the number of new cases in each week per country.
df_agg_week = (df_melt.groupby(['Country_State', 'Date_week']).sum()).copy()


# In[ ]:


# Remove columns && rename columns
del df_agg_week["Confirmed_Accumulated"]
df_agg_week = df_agg_week.rename(columns = {"Confirmed_New_Day": "Confirmed_New_Week"})


# In[ ]:


# Set index to columns: drop = False
df_agg_week.reset_index(inplace = True, drop = False)


# ## Data to be used further on our analysis

# In[ ]:


# IMPORTANT: This is the data that will be used on our analysis. However, will keep on the code to create a more robust database
df_week = df_agg_week.copy()
del df_week["Date_week"]
df_week.to_excel("df_week_confirmed.xlsx")


# In[ ]:


df_week


# ## Usefull Data for researchers

# In[ ]:


# Create key-column (Country_week) to join both dataFrames
df_melt["Country_week"] = df_melt["Country_State"] + "-" + df_melt["Date_week"].astype(str)
df_agg_week["Country_week"] = df_agg_week["Country_State"] + "-" + df_agg_week["Date_week"].astype(str)


# In[ ]:


# Reorder column
df_agg_week = df_agg_week[['Country_week', 'Confirmed_New_Week']]
df_melt = df_melt[["Country_week", 'Country_State', 'Date', "Date_week", "Confirmed_Accumulated", "Confirmed_New_Day"]]


# In[ ]:


# Create the final dataFrame with number of accumulated cases, daily cases and weekly cases. Key-column: Country_week
df_merge = df_melt.merge(df_agg_week, on="Country_week")


# In[ ]:


# Remove key column
del df_merge["Country_week"]


# In[ ]:


# Show final data and send to excell
df_merge.to_excel("df_merge_confirmed.xlsx")


# In[ ]:


# Note that we have few weeks of information per country. Insufficient for an analysis ungrouped per country.
df_merge


# # 5. Application: Confirmed cases of Covid-19 per week
# 
# For this analysis we are taking our treated data set with only the number of new cases per week. 
# 
# The idea is to select a sample size from this data set and analyze the frequency in which appears numbers from 1 to 9 in the first position of the values selected. Then we compare these frequencies to those predicted by Benford's theory. We will be using a chi-squared test for statistical significance. 
# 
# Since the sample per country is very small (only few weeks of information per country), we will be only interested to analyze frequencies of the entire data set as a whole. Later, in the end of the year when we have more data per country, then we may analyze frequencies per country too.

# ## 5.1. Analysis of the data set

# In[ ]:


# Note that it is possible to have a negative new number of cases. Meaning that in such a week the government corrected the numbers informed in the previous week.
df_week.describe()


# In[ ]:


# Some deeper analysis
df_analysis_desc = (df_week.groupby(['Country_State']).describe()).copy()
df_analysis_desc


# ## 5.2 Running the script
# 
# * This work uses the following utility script: "RKN Module - Benford Law". For more on how to use this script, please access:
# https://www.kaggle.com/rafaelknunes/rkn-module-benford-law-tutorial/notebook

# In[ ]:


# Getting hints (1 for aggregated analysis)
rkn_benford.hints(df_week, 1)


# In[ ]:


# df_week: data set with the values to be analyzed
# 1: Aggregated analysis (Since the sample per country is very small, we are only interested to analyze frequencies of the entire data set as a whole.)
# 30: Number of rounds we will run the code in order to produced an averaged chi-squared value.
# 1000: Sample size for the first digit analysis.
# 780: Sample size for the second digit analysis.
# 334: Sample size for the third digit analysis.
# 1: Number of graphs to produce with the best chi-sq values.
# 1: Number of graphs to produce with the worst chi-sq values. Same as the graph before.
table_app = rkn_benford.benford(df_week, 1, 30, 1000, 780, 334, 1, 1, "output_confirmed.xlsx", "")


# ### 5.2.1. First digit results
# 
# In this application we will use a sample size of 1000 out of 1270 possibles. The chi-squared obtained is very low. Only for a very low level of significance (around 5%) we would reject the hypothesis that this distribution is equal to the one predicted by Benford. In such case we are inclined to not consider that there is any kind of data manipulation.
# 
# However, our analysis focused only on the entire data. Maybe an analysis per country would reveal that some countries follow and others do not follow Benford's law. Unfortunately there is not enough data to make this investigation at the present moment.

# In[ ]:


# Order by city name
table_app[0].sort_values(by=['units'], inplace=True)
# Format table values
results_d1 = table_app[0].style.format({
    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,
    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 
    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,
    })


# In[ ]:


# The table results may differ from graph results, since they differ on the sample size selected.
results_d1


# In[ ]:


fig = plt.figure(figsize=(8,6), dpi=200)

a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D1__Aggregated_confirmed.png'))

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);


# ### 5.2.2. Second digit results
# 
# Again we find a very low chi-squared value (4.24) which means that for the second digit the values informed by governments around the world adhere to the Benford's Law.
# 
# Further analysis per country still needed.

# In[ ]:


# Order by city name
table_app[1].sort_values(by=['units'], inplace=True)
# Format table values
results_d2 = table_app[1].style.format({
    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,
    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 
    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,
    })


# In[ ]:


results_d2


# In[ ]:


fig = plt.figure(figsize=(8,6), dpi=200)

a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D2__Aggregated_confirmed.png'))

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);


# ### 5.2.3. Third digit results
# 
# 
# Finally, we find a very low chi-squared value (4.75) for the third digit. Which means that the values informed by governments around the world adhere to the Benford's Law.
# 
# Further analysis per country still needed.

# In[ ]:


# Order by city name
table_app[2].sort_values(by=['units'], inplace=True)
# Format table values
results_d3 = table_app[2].style.format({
    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,
    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 
    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,
    })


# In[ ]:


results_d3


# In[ ]:


fig = plt.figure(figsize=(8,6), dpi=200)

a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D3__Aggregated_confirmed.png'))

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);


# # 6. Final Remarks
# 
# Along this work we tested the values of new confirmed cases of Covid-19 per week informed by governments around the world. We took data from 22/jan/2020 to 31/mar/2020.
# 
# For the first, second and third digits the conclusion was the same: No evidence of data manipulation. However, we still need to make an specific investigation per country. Nevertheless, this will only be possible after we have more weeks of data, meaning a sufficient sample size.

# # THANKS FOR READING!
