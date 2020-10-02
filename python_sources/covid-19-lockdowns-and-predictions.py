#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The purpose of this notebook is to analyse the time series of the covid-19 outbreak for each country. In the notebook we will have a quick look at the overall evolution of cases and then work our way towards normalising and adjusting the time series to make them comparable across countries. The analysis consists of two main components: 
# * Normalizing the time series of each country by population and splitting China into some of it's regions.
# * Realigning the timeseries by setting the time "0" of each time series to be the day which is the answer to the following question: Which day was the first day that at least 1 in 500.000 in the given country/province was infected with the coronavirus?
# 
# The main result in this notebook is the final graph showing a log-normalized population and timeline-adjusted for the 8 countries with the most coronavirus cases as of 9. March 2020 plotted along with four of the five Chinese provinces with the most coronavirus cases.
# 
# ---
# The full analysis of the results can be read in this article:
# * https://www.linkedin.com/pulse/covid-19-next-lockdown-goes-nicolaj-schmit/?trackingId=LYcZqVdhRSum9YPUJcGiOA%3D%3D

# # Imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')


# # Parameters

# In[ ]:


N = 10 # Number of countries to analyse
gamma = 500_000 # starting criteria threshold - at least 1 in a 500.000 infected
M = 50
offset = 0
factor = 1_000_000 # one in a million


# # Load main dataset data

# In[ ]:


df_pivot = pd.read_csv(r'../input/covid-current-dataset/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv') # r'../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')#, error_bad_lines=False)
print(df_pivot.describe())
df = df_pivot.melt(id_vars=['Province/State','Country/Region','Lat','Long'], var_name='date', value_name='cumulative_cases')
df = df.assign(date = pd.to_datetime(df['date']), format='%d/%m/%y', inplace=True)
assert isinstance(df.date.iloc[0],pd._libs.tslibs.timestamps.Timestamp)
df.rename(columns={'Country/Region': 'country','Province/State': 'province'}, inplace=True)


# In[ ]:


df_country = df.groupby(['country','date'])['cumulative_cases'].sum().reset_index()
df_province = df.groupby(['province','date'])['cumulative_cases'].sum().reset_index()


# In[ ]:


print('First date available in dataset is:', str(df.date.min()))


# The first date available in the John Hopkins University data set is 2020-01-22. Hence, we will enrich the dataset with data from European Centre for Disease Prevention and Control (https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases)

# # Enrichment of data

# In[ ]:


data_path = r'../input/historical-covid-cases/COVID-19-geographic-disbtribution-worldwide-6-march-2020.xls' # COVID-19-geographic-disbtribution-worldwide-6-march-2020.xls'
df_extra = pd.read_excel(data_path)
df_extra.rename(columns={'CountryExp': 'country', 'DateRep': 'date', 'NewConfCases': 'cases'}, inplace=True)
print('SHAPE')
print(df_extra.shape) 
print('COLUMNS')
print(df_extra.columns)
print('DATATYPES')
print(df_extra.dtypes)
print('SUMMARY STATISTICS')
print(df_extra.describe())


# The data set used to enrich goes all the way back to the beginning, i.e. last day of 2019

# In[ ]:


print('First date available in ECDC dataset is:',df_extra.date.min())


# We will also be interested in the evolution in cases in the affected areas in China. Before the 2020-01-20 only Hubei had a significant amount; we will make a rough estimate that 80% of all registered cases in China during that period was registered in Hubei.

# In[ ]:


# Quick'n'Dirty - EARLY HUBEI ESTIMATE 
df_hubei = df_extra[df_extra['country'] == 'China']
df_hubei = df_hubei.assign(cases = np.round(df_hubei['cases'] * 0.8))
df_hubei = df_hubei.assign(country = 'Hubei')
df_extra = df_extra.append(df_hubei)
df_extra = df_extra.sort_values(by=['country', 'date']).reset_index(drop=True)
df_extra['cumulative_cases'] = df_extra.groupby('country')['cases'].transform(pd.Series.cumsum)
df_extra_in_scope = df_extra[df_extra['date'] < df.date.min()] # CUT-OFF date
#df_extra_in_scope.date.max()

# Do some renaming to get aligned
df_extra_in_scope['country'] = df_extra_in_scope['country'].replace(['China', 'United States of America', 'United Kingdom', 'Cases on an international conveyance Japan'], ['Mainland China', 'US', 'UK', 'Other'])
df_extra_in_scope = df_extra_in_scope[['country', 'date', 'cumulative_cases']]
#df_extra_in_scope[df_extra_in_scope['country'] == 'Mainland China'].sort_values(by='cumulative_cases', ascending=False)
df_extra_in_scope


# Now we have enrichmentdata available for Hubei and on a country level we enrich the John Hopkins coronavirus dataset

# In[ ]:


# Enrich John Hopkins dataset
df_country = df_country.append(df_extra_in_scope[df_extra_in_scope['country'] != 'Hubei'])
df_province = df_province.append(df_extra_in_scope[df_extra_in_scope['country'] == 'Hubei'].rename(columns={'country': 'province'}))


# Having joined the data let's have a quick look at the total number of cases

# # Summary statistics

# In[ ]:


print('Total number of cases')
total_cases = df_country.groupby('country').max()['cumulative_cases'].sum()
print(total_cases) #.sum())

print('Number of cases registered in a sub-province')
province_cases =  df_province.groupby('province').max()['cumulative_cases'].sum()
print(province_cases)


# And the list of countries with the most cases are:

# In[ ]:


df_country.groupby('country').max().sort_values(by=['cumulative_cases'], ascending=False).head(N)


# Provinces with most cases

# In[ ]:


df_province.groupby('province').max().sort_values(by='cumulative_cases', ascending=False).head(N)


# Next up we can have a look at the evolution over time at an aggregate level:

# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
df_country.groupby('date').sum().sort_values(by='date')[['cumulative_cases']].plot(title='Evolution of total number cases over time', ax=ax)
#print(df_country.groupby('date').sum().sort_values(by='date')[['cumulative_cases']].iloc[-1])
plt.show()


# # Helper functions

# In[ ]:


def log_cases(df, groupby_cols, input_col='cumulative_cases', output_col='log_cumulative_cases'):
    df_temp = df.groupby(groupby_cols).sum().sort_values(by=groupby_cols, ascending=True).reset_index(drop=False)
    df_temp[output_col] = np.log(1+df_temp[input_col])
    return df_temp

def adjusted_cases(df, input_col='cumulative_cases', adjust_col='population', output_col='adj_cumulative_cases', factor=1_000_000):
        df[output_col] = df[input_col] * factor / df[adjust_col]
        return df

def plot_simple_evolution(df):
    df[['log_cumulative_cases']].plot(title='Log evolution of total number cases over time')
    plt.show()

def plot_evolution(ax, df, x, y, hue='country', exclusion_list=None, inclusion_list=['Italy', 'South Korea', 'Mainland China'], marker='.', alpha=0.8):
    
    # Filter on inclusion list
    if inclusion_list is not None:
        df = df[df[hue].isin(inclusion_list)]

    if exclusion_list:
        df = df[~df[hue].isin(exclusion_list)]
        
    # Linear plot
    sns.lineplot(x=x, y=y, hue=hue, data=df, ax=ax, marker=marker, alpha=alpha)
    
    title_text = f'{y} over time {x} by {hue}'
    if exclusion_list is not None:
        title_text += ' w/o ' + ', '.join(exclusion_list)
    ax.set_title(label=title_text)

    return ax #plt.show()


# Next we can to a log-transformation of the number of cases, and further we can plot the evolution of cases by country (top 10) and Chinese provinces (top 5).

# # Country og province evolution

# In[ ]:


df_temp = log_cases(df_country, groupby_cols=['date'])
plot_simple_evolution(df_temp)

topNcountries = df_country.groupby('country').max().sort_values('cumulative_cases', ascending=False).index[offset:N+offset].values.tolist()
top5province = df_province.groupby('province').max().sort_values('cumulative_cases', ascending=False).index[:5].values.tolist()

df_country_temp = log_cases(df_country, input_col='cumulative_cases', output_col='log_cumulative_cases', groupby_cols=['date', 'country'])
df_province_temp = log_cases(df_province, input_col='cumulative_cases', output_col='log_cumulative_cases', groupby_cols=['date', 'province'])


# Plots
fig, ax = plt.subplots(figsize=(20,6), ncols=3)
plot_evolution(ax=ax[1], df=df_country_temp, x='date', y='cumulative_cases', inclusion_list=topNcountries)
plot_evolution(ax=ax[0], df=df_country_temp, x='date', y='cumulative_cases', inclusion_list=topNcountries, exclusion_list=['Mainland China'])
plot_evolution(ax=ax[2], df=df_country_temp, x='date', y='log_cumulative_cases', inclusion_list=topNcountries)
plt.suptitle('Countries')
plt.show()


# Plots
fig, ax = plt.subplots(figsize=(12,6), ncols=2)
plot_evolution(ax=ax[0], df=df_province_temp, x='date', y='cumulative_cases', hue='province', inclusion_list=top5province)
plot_evolution(ax=ax[1], df=df_province_temp, x='date', y='log_cumulative_cases', hue='province', inclusion_list=top5province)
plt.suptitle('Provinces')
plt.show()


# To make the graphs more comparable we will normalize the number of corona cases in a given country/province by normalizing with the population of the country/province. The data set used for population size as of 2016 can be found here: https://datahub.io/JohnSnowLabs/population-figures-by-country#resource-population-figures-by-country-csv

# In[ ]:


pop_data_path = r'../input/historical-covid-cases/population-figures-by-country-csv_csv.csv' # get a 403 forbidden error when trying to download directly from link; so needed to store it locally first
df_pop = pd.read_csv(pop_data_path)
df_pop['Country'] = df_pop['Country'].replace(['China', 'Korea, Rep.', 'United States', 'Iran, Islamic Rep.', 'United Kingdom'], ['Mainland China', 'South Korea', 'US', 'Iran', 'UK']) # Align naming of south korea
df_pop['population'] = df_pop['Year_2016']
df_pop = df_pop[['Country','population']]
# We need to add a few regions manually by looking up the number on Google :-) 
df_pop_added = pd.DataFrame([('Hubei', 60_000_000),
                             ('Guangdong', 113_000_000),
                             ('Henan', 94_000_000),
                             ('Zhejiang ', 57_000_000),
                             ('Hunan', 67_000_000),
                             ('Hong Kong', 7_000_000),
                             ('Taiwan', 23_780_000),
                            ], columns=['Country', 'population'])
df_pop = df_pop.append(df_pop_added).reset_index(drop=True)


# In[ ]:


df_country_pop = df_country.merge(df_pop[['Country', 'population']], left_on=['country'], right_on=['Country'])
df_province_pop = df_province.merge(df_pop[['Country', 'population']], left_on=['province'], right_on=['Country'])


# Having loaded the populations, we now adjust the coronavirus timeseries with the population of the country/province

# In[ ]:


df_country_temp = adjusted_cases(df_country_pop, input_col='cumulative_cases', adjust_col='population', output_col='adj_cumulative_cases', factor=1_000_000)
df_province_temp = adjusted_cases(df_province_pop, input_col='cumulative_cases', adjust_col='population', output_col='adj_cumulative_cases', factor=1_000_000)
df_country_temp = log_cases(df_country_temp, input_col='adj_cumulative_cases', output_col='log_adj_cumulative_cases', groupby_cols=['date', 'country'])
df_province_temp = log_cases(df_province_temp, input_col='adj_cumulative_cases', output_col='log_adj_cumulative_cases', groupby_cols=['date', 'province'])


# The population adjusted and the log-population adjusted number are given below:

# In[ ]:


df_country_temp[df_country_temp['country'].isin(topNcountries)].groupby('country').max().sort_values(by='adj_cumulative_cases', ascending=False)


# # Populations adjusted timeseries

# Having adjusted for population we can plot the updated time-series:

# In[ ]:


# Plots
fig, ax = plt.subplots(figsize=(20,6), ncols=3)
plot_evolution(ax=ax[0], df=df_country_temp, x='date', y='adj_cumulative_cases', inclusion_list=topNcountries)
plot_evolution(ax=ax[1], df=df_country_temp, x='date', y='adj_cumulative_cases', inclusion_list=topNcountries, exclusion_list=['Mainland China'])
plot_evolution(ax=ax[2], df=df_country_temp, x='date', y='log_adj_cumulative_cases', inclusion_list=topNcountries)
plt.suptitle('Countries')
plt.show()

# Plots
fig, ax = plt.subplots(figsize=(12,6), ncols=2)
plot_evolution(ax=ax[0], df=df_province_temp, x='date', y='adj_cumulative_cases', hue='province', inclusion_list=top5province)
plot_evolution(ax=ax[1], df=df_province_temp, x='date', y='log_adj_cumulative_cases', hue='province', inclusion_list=top5province)
plt.suptitle('Provinces')
plt.show()


# The countries with the highest ratio of infected populations are:

# In[ ]:


df_country_temp['ratio_infected'] = df_country_temp['cumulative_cases'] / df_country_temp['population']
print(f'Countries - Number of persons infected per {factor} capita')
print((df_country_temp.groupby('country')['ratio_infected'].max().sort_values(ascending=False) * factor).head(N))

df_province_temp['ratio_infected'] = df_province_temp['cumulative_cases'] / df_province_temp['population']
print(f'Provinces - Number of persons infected per {factor} capita')
print((df_province_temp.groupby('province')['ratio_infected'].max().sort_values(ascending=False) * factor))


# # Re-aligning the timeseries

# We will now work towards realigning the time series by setting day "0" for each country in the following way::
#         
# The timeseries are realigned by setting the time "0" of each time series to be the day which is the answer to the following question: Which day was the first day that at least 1 in 500.000 in the given country/province was infected with the coronavirus?

# In[ ]:


# Countries - thresholds
first_case_dates = df_country_temp[df_country_temp['cumulative_cases'] > 0].groupby('country')['date'].min().reset_index()
first_m_cases_dates = df_country_temp[df_country_temp['cumulative_cases'] > M].groupby('country')['date'].min().reset_index()
at_least_1_in_gamma_infected_dates = df_country_temp[gamma * df_country_temp['cumulative_cases'] > df_country_temp['population'] ].groupby('country')['date'].min().reset_index()
# rename the columns 
first_case_dates = first_case_dates.rename(columns={'date': 'date_first_case'})
first_m_cases_dates.rename(columns={'date': f'date_first_{M}_case'}, inplace=True)
at_least_1_in_gamma_infected_dates.rename(columns={'date': f'date_1_in_{gamma}_infected'}, inplace=True)

# Merge dates on the dataframe
df_country_temp = df_country_temp.merge(first_case_dates, how='left', on='country')                    .merge(first_m_cases_dates, how='left', on='country')                    .merge(at_least_1_in_gamma_infected_dates, how='left', on='country')                    #.merge(first_death_dates, how='left', on='country')


#####

#print('SOME EXAMPLES')
#print('Date of first case')
#print(first_case_dates[first_case_dates['country'].isin(topNcountries)])
#print(f'First date where number of cases in country exceeded {M} infected people')
#print(first_m_cases_dates[first_m_cases_dates['country'].isin(topNcountries)])
print(f'Date where at least 1 in {gamma} in the country was infected')
print(at_least_1_in_gamma_infected_dates[at_least_1_in_gamma_infected_dates['country'].isin(topNcountries)])
#print(at_least_1_in_gamma_infected_dates[at_least_1_in_gamma_infected_dates['country'].isin(topNcountries)].reset_index(drop=True).to_markdown())
print(f'Number of contries with at least 1 in {gamma} persons infected.')
print(at_least_1_in_gamma_infected_dates.shape[0])


###

# Provinces - thresholds 
first_case_dates = df_province_temp[df_province_temp['cumulative_cases'] > 0].groupby('province')['date'].min().reset_index()
first_m_cases_dates = df_province_temp[df_province_temp['cumulative_cases'] > M].groupby('province')['date'].min().reset_index()
at_least_1_in_gamma_infected_dates = df_province_temp[gamma * df_province_temp['cumulative_cases'] > df_province_temp['population'] ].groupby('province')['date'].min().reset_index()
# rename the columns 
first_case_dates = first_case_dates.rename(columns={'date': 'date_first_case'})
first_m_cases_dates.rename(columns={'date': f'date_first_{M}_case'}, inplace=True)
at_least_1_in_gamma_infected_dates.rename(columns={'date': f'date_1_in_{gamma}_infected'}, inplace=True)

# Merge dates on the dataframe
df_province_temp = df_province_temp.merge(first_case_dates, how='left', on='province')                    .merge(first_m_cases_dates, how='left', on='province')                    .merge(at_least_1_in_gamma_infected_dates, how='left', on='province')                   


# |    | country        | date_1_in_500000_infected   |
# |---:|:---------------|:----------------------------|
# |  0 | France         | 2020-03-02 00:00:00         |
# |  1 | Germany        | 2020-03-03 00:00:00         |
# |  2 | Iran           | 2020-02-27 00:00:00         |
# |  3 | Italy          | 2020-02-23 00:00:00         |
# |  4 | Japan          | 2020-03-01 00:00:00         |
# |  5 | Mainland China | 2020-01-27 00:00:00         |
# |  6 | South Korea    | 2020-02-20 00:00:00         |
# |  7 | Spain          | 2020-03-02 00:00:00         |
# 

# Now we can adjust the starting point of each timeseries by the "starting criteria" above.

# In[ ]:


starting_criteria = 'date_1_in_500000_infected' #  date_first_case	 date_first_50_case	date_1_in_1000000_infected	date_first_death

# Countries
mask = df_country_temp[starting_criteria].notnull()
df_country_in_scope = df_country_temp[mask]
df_country_in_scope = df_country_in_scope.assign(days=df_country_in_scope['date'] - df_country_in_scope[starting_criteria])
df_country_in_scope['days'] = df_country_in_scope['days'].apply(lambda x: x.days)
#print(df_country_in_scope.columns, df_country_in_scope.shape)

# Provinces
mask = df_province_temp[starting_criteria].notnull()
df_province_in_scope = df_province_temp[mask]
df_province_in_scope = df_province_in_scope.assign(days=df_province_in_scope['date'] - df_province_in_scope[starting_criteria])
df_province_in_scope['days'] = df_province_in_scope['days'].apply(lambda x: x.days)
#print(df_province_in_scope.columns, df_province_in_scope.shape)


# After re-adjusting the time-series we can plot the time-series again with the above definition of day "0".

# In[ ]:


# Prepare for plots
fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
plot_evolution(ax=ax, df=df_country_in_scope[df_country_in_scope['days'] > -10], x='days', y='log_adj_cumulative_cases', hue='country', inclusion_list=topNcountries)
#plot_evolution(ax=ax[1], df=df_province_in_scope[df_province_in_scope['days'] > -10], x='days', y='log_adj_cumulative_cases', hue='province', inclusion_list=top5province)
plt.show()


# In[ ]:


#df_country_in_scope[df_country_in_scope['days'] == 0]


# Now stack the provinces into the same dataframe as countries, assuming the provinces are "countries themselves". In addition we add the Wuhan lockdown and the Northern Italy lockdown dates to the plot.

# # Analysis and lockdowns

# In[ ]:


df_in_scope = df_country_in_scope.append(df_province_in_scope.rename(columns={'province': 'country'}))

#print(df_country_in_scope.country.unique())
#print(df_province_in_scope.province.unique())
#print(df_in_scope.country.unique())

# plot
fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
plot_evolution(ax=ax, df=df_in_scope[df_in_scope['days'] > -10], x='days', y='log_adj_cumulative_cases', hue='country', inclusion_list=topNcountries+top5province)

# LOCKDOWNS
# 23/01/2020 - China (Wuhan) - https://www.businessinsider.com/transit-wuhan-china-shut-down-coronavirus-2020-1?r=US&IR=T
# 08/03/2020 - Italy (Milan, Lombardia and more regions in Nothern Italy) - https://www.bloomberg.com/news/articles/2020-03-07/italy-to-impose-virtual-ban-on-entry-to-lombardy-corriere?srnd=premium-europehttps://www.bloomberg.com/news/articles/2020-03-07/italy-to-impose-virtual-ban-on-entry-to-lombardy-corriere?srnd=premium-europe

df_in_scope.loc[df_in_scope['country'] == 'China', 'lockdown'] = pd.to_datetime('23-01-2020', format='%d-%m-%Y')
df_in_scope.loc[df_in_scope['country'] == 'Hubei', 'lockdown'] = pd.to_datetime('23-01-2020', format='%d-%m-%Y')
df_in_scope.loc[df_in_scope['country'] == 'Italy', 'lockdown'] = pd.to_datetime('07-03-2020', format='%d-%m-%Y')
df_in_scope.loc[df_in_scope['country'] == 'China', 'lockdown_text'] = 'Hubei/Wuhan lockdown'
df_in_scope.loc[df_in_scope['country'] == 'Hubei', 'lockdown_text'] = 'Hubei/Wuhan lockdown'
df_in_scope.loc[df_in_scope['country'] == 'Italy', 'lockdown_text'] = 'Northern Italy lockdown'
lockdowns = df_in_scope[df_in_scope['date'] == df_in_scope['lockdown']]
#print(lockdowns)

sns.regplot(x='days', y='log_adj_cumulative_cases', data=lockdowns, ax=ax, fit_reg=False, color='red', marker='^')
for index, row in lockdowns.iterrows():
    ax.annotate('<-' + row['lockdown_text'], (row['days']+0.5, row['log_adj_cumulative_cases']-0.1))




plt.show()


# "There is a striking similarity in the evolution of log-population-normalized cases between Hubei and Italy, Iran and up until recently South Korea. But if we take a look a tad earlier on the graph it also seems that France, Germany and Spain might be, at least for now, on a similar path. France, Germany and Spain are all around day "6" right now whereas Italy has reached day "14". Can we expect to see similar measures in France, Germany and Spain as those taken in Italy if a similar density of coronavirus is reached or exceeded in those countries? If the growth continues with the same speed as Italy then that might happen in about a week or so."
# 
# ---
# 
# The full analysis of the results can be read in this article:
# * https://www.linkedin.com/pulse/covid-19-next-lockdown-goes-nicolaj-schmit/?trackingId=LYcZqVdhRSum9YPUJcGiOA%3D%3D
