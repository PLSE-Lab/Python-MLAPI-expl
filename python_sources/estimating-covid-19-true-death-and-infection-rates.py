#!/usr/bin/env python
# coding: utf-8

# # Estimating COVID-19 True Death and Infection Rates
# 
# In this notebook we try to estimate the true death and true infection rate for each country of COVID-19.  
# We will attempt to do this **based only on reported number of cases, deaths and tests by each country** and also utilize data about the population age distribution of each country.  
# 
# For this purpose we will use a [worldometer snapshots dataset](https://www.kaggle.com/selfishgene/covid19-worldometer-snapshots-since-april-18) I've collected in the past several months that contains, among other things, the most credible and extensive data (that I know of) about testing in all countries
# 
# ![image](https://i.ibb.co/hfSkvhX/worldometer-snapshot-sorted-by-total-deaths.png)
# 
# Following this analsysis we ask wheather the COVID-19 pandemic is over, and for this purpose we utilize the full [Johns Hopkings dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset) (uploaded by [SRK](https://www.kaggle.com/sudalairajkumar) here on kaggle)
# 
# 
# The logical path we will take for the purpose of **estimating true death and infection rates** is:
# * Some countries conduct large amounts of tests and some countries conduct a small amount of tests
# * We will assume that countries that conduct many tests (compared to the number of actual cases they have) are very close to discovering almost all of the infected people among them
# * We will then calculate the CFR for those countries (CFR - case fatality ratio) and just assert that it's close to the true IFR (IFR - infection fatality ratio) for those countries
# * We will average the IFR for the "good testing countries" and is the true IFR of COVID-19
# * We will then assume that the IFR is country depended only through the age distribution of that country
# * Based on population structure of each country (fraction of the population that is older than 65 years old) we will calculate the estimated country specific IFR (e.g. we expect a country with younger population to have a smaller infection fatality rate than a country with a very old population)  
# * We then use that number to lower bound the actual infection in a country (e.g. if IFR is 1% for a given country with a given age structure, and the number of confirmed deaths in that country is 3,456, then we will assume that the total number of infected people in that country is at least 3456*100 = 345,600, even if that country has only reported 56,789 cases)
# * We then divide the resulting number with the population to get the true Infection Rate for each country

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ## Load the main data table and display it

# In[ ]:


dataset_dir = '/kaggle/input/covid19-worldometer-snapshots-since-april-18/'
worldometer_df = pd.read_csv(os.path.join(dataset_dir, 'worldometers_snapshots_April18_to_August1.csv'))
worldometer_df.head(11)


# ## Display a sub-table of a specific country (first and last dates available)

# In[ ]:


country_name = 'USA'

country_df = worldometer_df.loc[worldometer_df['Country'] == country_name, :]
country_df = country_df.reset_index(drop=True)

country_df.head(7).append(country_df.tail(7))


# ## Display a sub-table of a specific date

# In[ ]:


selected_date = datetime.strptime('01/07/2020', '%d/%m/%Y')

selected_date_df = worldometer_df.loc[worldometer_df['Date'] == selected_date.strftime('%Y-%m-%d'), :]
selected_date_df = selected_date_df.reset_index(drop=True)
selected_date_df.head(11)


# ## Lets take the last available date and continue our analysis on it

# In[ ]:


last_date = datetime.strptime('30/07/2020', '%d/%m/%Y')

last_date_df = worldometer_df.loc[worldometer_df['Date'] == last_date.strftime('%Y-%m-%d'), :]
last_date_df = last_date_df.reset_index(drop=True)
last_date_df.head(26)


# ### Remove China from dataset
# China had suspect reporting issues from the get go of this pandemic

# In[ ]:


# remove china as clear outlier and highly suspect reporting
last_date_df = last_date_df.drop(np.nonzero(np.array(last_date_df.loc[:,'Country'] == 'China'))[0], axis=0).reset_index(drop=True)


# ## First, calculate the naive death rate for each country and show histogram

# In[ ]:


last_date_df['Case Fatality Ratio'] = last_date_df['Total Deaths'] / last_date_df['Total Cases']

bins = np.arange(21)
counts, _ = np.histogram(100 * np.array(last_date_df['Case Fatality Ratio']), bins=bins)

plt.figure(figsize=(12,8))
plt.bar(bins[:-1], counts, facecolor='blue', edgecolor='black')
plt.xlabel('Death Rate (%)', fontsize=16)
plt.ylabel('Number of Countries in each bin', fontsize=16)
plt.title('Histogram of Death Rates for various Countries', fontsize=18);


# ### We see a large spread of death rates between countries
# This shouldn't be the case normally, as humans are humans and are likely affected similarly by the disease in various regions of the world  
# ### The question arises: **what can explain this spread?**

# # Filter out countries with small amount of cases
# maybe the spread is due to small samples sizes for countries with small amounts of cases

# In[ ]:


min_number_of_cases = 10000

greatly_affected_df = last_date_df.loc[last_date_df['Total Cases'] > min_number_of_cases,:]

bins = np.arange(21)
counts, _ = np.histogram(100 * np.array(greatly_affected_df['Case Fatality Ratio']), bins=bins)

plt.figure(figsize=(12,8))
plt.bar(bins[:-1], counts, facecolor='blue', edgecolor='black')
plt.xlabel('Death Rate (%)', fontsize=16)
plt.ylabel('Number of Countries in each bin', fontsize=16)
plt.title('Histogram of Death Rates for various Countries', fontsize=18);


# We can see that the spread is somewhat reduced, but is still large and un accounted for

# ## Plot scatter of death rate as function of testing quality
# We know some countries were more responsible regarding their testing strategy and some were less so
# let's plot the death rate as function of testing quality (as mesured by number of tests performed per every positive case)

# In[ ]:


last_date_df['Num Tests per Positive Case'] = last_date_df['Total Tests'] / last_date_df['Total Cases']

greatly_affected_df = last_date_df.loc[last_date_df['Total Cases'] > min_number_of_cases,:]

# limit the x-axis so that the scatter plot will not be empty on the right side
x_axis_limit = 75
random_limit_vec = x_axis_limit + 15 * np.random.rand(greatly_affected_df.shape[0])

death_rate_percent = 100 * np.array(greatly_affected_df['Case Fatality Ratio'])
num_test_per_positive = np.array(greatly_affected_df['Num Tests per Positive Case'])
num_test_per_positive[num_test_per_positive > random_limit_vec] = random_limit_vec[num_test_per_positive > random_limit_vec]
total_num_deaths = np.array(greatly_affected_df['Total Deaths'])
population = np.array(greatly_affected_df['Population'])

plt.figure(figsize=(16,12))
plt.scatter(x=num_test_per_positive, y=death_rate_percent, 
            s=0.5*np.power(np.log(1+population),2), 
            c=np.log10(1+total_num_deaths))
plt.colorbar()
plt.ylabel('Death Rate (%)', fontsize=16)
plt.xlabel('Number of Tests per Positive Case', fontsize=16)
plt.title('Death Rate as function of Testing Quality', fontsize=18)
plt.xlim(-2, x_axis_limit + 30)
plt.ylim(-0.2,17)

# plot on top of the figure the names of the
#countries_to_display = greatly_affected_df['Country'].unique().tolist()
countries_to_display = ['USA', 'Brazil', 'Russia', 'Spain', 'UK', 'Italy', 'France',
                        'Germany', 'India', 'Iran', 'Canada', 'Mexico', 
                        'Belgium', 'Pakistan', 'Netherlands', 'Qatar', 'Ecuador', 
                        'Sweden', 'Singapore', 'Portugal', 'UAE', 'Ireland', 
                        'South Africa', 'Poland', 'Kuwait', 'Ukraine', 'Venezuela',
                        'Romania', 'Egypt', 'Israel', 'Japan', 'Austria', 'Philippines', 
                        'Denmark', 'S. Korea', 'Serbia', 'Afghanistan', 'Ethiopia',
                        'Bahrain', 'Czechia', 'Kazakhstan','Nepal', 'Uzbekistan',
                        'Algeria', 'Australia', 'Moldova', 'Ghana', 'Bulgaria',
                        'Armenia', 'Bolivia', 'Cameroon', 'Iraq', 'Azerbaijan','Morocco']

for country_name in countries_to_display:
    country_index = greatly_affected_df.index[greatly_affected_df['Country'] == country_name]
    plt.text(x=num_test_per_positive[country_index] + 0.5,
             y=death_rate_percent[country_index] + 0.2,
             s=country_name, fontsize=10);


# We can see that the better the testing, the lower the variability of the the death between the different countries  
# The color represents the total number of deaths (on a log scale)
# 
# NOTE: countries with more than 75 cases per positive have been clipped in order to be properly displayied on the same graph

# ## Group contriues into testing quality bins and show mean +/- stdev

# In[ ]:


tests_per_positive_bins = [[0,30], [30,55], [55,400]]

death_rate_percent    = 100 * np.array(greatly_affected_df['Case Fatality Ratio'])
num_test_per_positive = np.array(greatly_affected_df['Num Tests per Positive Case'])
total_num_deaths      = np.array(greatly_affected_df['Total Deaths'])
total_num_cases       = np.array(greatly_affected_df['Total Cases'])

death_rate_average = []
death_rate_variability = []
death_rate_standard_error = []
for tests_bin in tests_per_positive_bins:
    countries_in_bin = np.logical_and(num_test_per_positive > tests_bin[0], num_test_per_positive <= tests_bin[1])
    
    death_rates_in_bin = death_rate_percent[countries_in_bin]
    death_rate_average.append(death_rates_in_bin.mean())
    death_rate_variability.append(death_rates_in_bin.std())
    death_rate_standard_error.append(death_rates_in_bin.std()/np.sqrt(len(death_rates_in_bin)))

bins_locs = range(len(tests_per_positive_bins))
plt.figure(figsize=(8,8))
plt.bar(bins_locs, death_rate_average, yerr=death_rate_standard_error, 
        facecolor='blue', edgecolor='black')
plt.xticks(bins_locs, [str(x) for x in tests_per_positive_bins])
plt.title('Death Rate as function of Testing Quality', fontsize=18)
plt.ylabel('Death Rate (%)', fontsize=16)
plt.xlabel('Number of Tests per Positive Case', fontsize=16);


# ## Look at data from best testing countries
# Lets decide that the cutoff for "good testing country" is 55 tests per positive case (less than 2% positivity rate)

# In[ ]:


good_testing_threshold = 55
good_testing_df = greatly_affected_df.loc[greatly_affected_df['Num Tests per Positive Case'] > good_testing_threshold,:]
good_testing_df = good_testing_df.reset_index(drop=True)
good_testing_df


# # Lets calculate the Death Rate for those countries

# In[ ]:


estimated_death_rate_percent = 100 * good_testing_df['Total Deaths'].sum() / good_testing_df['Total Cases'].sum()

print('Death Rate only for "good testing countries" is %.2f%s' %(estimated_death_rate_percent,'%'))


# # Note that this value correponds well to the estimated IFR in New York City
# NYC, up to now, is the hardest hit place in the world with COVID-19.  
# Apprixmatley 20% of the population in NYC were estimated to be infected (had COVID-19 antibodies in their blood based on a sersurvey conducetd in late April)  
# 
# NOTE: Most widely used antibody tests have high false positive rates (~1%), and therefore they will result in wrong conclutions if applied to areas with low infection rates. 
# As NYC was the hardest hit place, it's likely to assume this NYC serosurvey is one of the best for the purpuse of IFR estimation
# 
# full details can be found here:
# [https://www.worldometers.info/coronavirus/coronavirus-death-rate/](https://www.worldometers.info/coronavirus/coronavirus-death-rate/)

# ## Lets now examine relationship with age
# We know COVID-19 is especially dengeruous for the older population, let's look if this relationship is repersented in the data

# In[ ]:


population_age_df = pd.read_csv(os.path.join(dataset_dir, 'population_structure_by_age_per_contry.csv'))
population_age_df.head(11)


# ## Merge the two tables

# In[ ]:


pop_older_df = population_age_df.loc[:,['Country','Fraction age 65+ years']]

greatly_affected_age_df = greatly_affected_df.merge(pop_older_df, on='Country')
greatly_affected_age_df.head(16)


# ## Show scatter plot of Death Rate vs Fraction 65+ yo population in each country

# In[ ]:


death_rate_percent = 100 * np.array(greatly_affected_age_df['Case Fatality Ratio'])
percent_older = 100 * np.array(greatly_affected_age_df['Fraction age 65+ years'])
total_num_deaths = np.array(greatly_affected_age_df['Total Deaths'])
population = np.array(greatly_affected_age_df['Population'])

plt.figure(figsize=(16,12))
plt.scatter(x=percent_older, y=death_rate_percent, 
            s=0.5*np.power(np.log(1+population),2), 
            c=np.log10(1+total_num_deaths))
plt.colorbar()
plt.ylabel('Death Rate (%)', fontsize=16)
plt.xlabel('Percent of population 65 years or older', fontsize=16)
plt.title('Death Rate as function of Fraction of Older Population', fontsize=18)
plt.ylim(-0.2,17)
plt.xlim(-0.2,31)

# plot on top of the figure the names of the
#countries_to_display = greatly_affected_df['Country'].unique().tolist()
countries_to_display = ['USA', 'Brazil', 'Russia', 'Spain', 'UK', 'Italy', 'France',
                        'Germany', 'India', 'Iran', 'Canada', 'Mexico', 
                        'Belgium', 'Pakistan', 'Netherlands', 'Qatar', 'Ecuador', 
                        'Belarus', 'Sweden', 'Singapore', 'Portugal', 'UAE', 'Ireland', 
                        'South Africa', 'Poland', 'Kuwait', 'Colombia', 'Ukraine', 'Venezuela',
                        'Romania', 'Egypt', 'Israel', 'Japan', 'Austria', 'Philippines', 
                        'Argentina', 'Denmark', 'S. Korea', 'Serbia', 'Ethiopia',
                        'Bahrain', 'Czechia', 'Kazakhstan','Nepal',
                        'Algeria', 'Australia', 'Moldova', 'Ghana', 'Bulgaria',
                        'Armenia', 'Cameroon', 'Iraq', 'Azerbaijan', 'Morocco']

for country_name in countries_to_display:
    country_index = greatly_affected_df.index[greatly_affected_age_df['Country'] == country_name]
    plt.text(x=percent_older[country_index] + 0.25,
             y=death_rate_percent[country_index] + 0.02,
             s=country_name, fontsize=10)


# ## It does appear there is correlation with age as well
# 
# **Disclaimer**: much of this correlation might just be a consequence of the simple fact that most european countries have been hit first and responded slowly, and that they also happen to have a very old population. Also, the fraction of older population is an indication of a developed country which is a possible confounder in addition to western countries being hit first by the virus

# # Estimate the infection fatality rate (IFR) for each country  
# For this we will assume that for good testing contries the case fatality rate (CFR) is similar to the infection fatality rate (IFR)  
# We've already verified that there is great dependece of CFR due to age of the population, so we will calibrate each countrie's IFR based on the fraction of people older than 65 years old, e.g. we expect the IFR of a country with 5% 65+ year old population to be 4 times smaller than a country with 20% 65+ year old population  
#   
# NOTE: this calibration assumes that only people older than 65 years old are dying from the disease, which is only an approximation, but it's not a bad one (it is estimated that only 5% of COVID-19 deaths are of people younger than 65 years)

# ### First, calculate the mean fraction of 65+ years old peope for our estimated IFR

# In[ ]:


good_testing_age_df = good_testing_df.merge(pop_older_df, on='Country')
good_testing_age_df


# In[ ]:


fraction_older_at_IFR = sum(good_testing_age_df['Total Deaths'] * good_testing_age_df['Fraction age 65+ years']) / sum(good_testing_age_df['Total Deaths'])
estimated_IFR = good_testing_age_df['Total Deaths'].sum() / good_testing_age_df['Total Cases'].sum()

print('Estimtaed IFR for a country with %.1f%s of population older than 65 years old is %.2f%s' %(100 * fraction_older_at_IFR,'%',100 * estimated_IFR,'%'))


# In[ ]:


all_countries_age_df = last_date_df.merge(pop_older_df, on='Country')
all_countries_age_df['Estimated Infection Fatality Ratio'] = (all_countries_age_df['Fraction age 65+ years'] / fraction_older_at_IFR) * estimated_IFR
#all_countries_age_df.head(11)


# ## Estimate the true infection rate for each country
# based on country sepecific IFR estimation we can put up a tighter lower bound on the true number of infected people solely based on the reported number of deaths for that country

# In[ ]:


all_countries_age_df['Total Infected'] = all_countries_age_df['Total Deaths'] / all_countries_age_df['Estimated Infection Fatality Ratio']
all_countries_age_df['Total Infected'] = all_countries_age_df[['Total Infected','Total Cases']].max(axis=1)

non_nan_cols = np.logical_not(np.isnan(all_countries_age_df['Total Infected']))
all_countries_age_df.loc[non_nan_cols,'Total Infected'] = all_countries_age_df.loc[non_nan_cols,'Total Infected'].astype(int)
#all_countries_age_df.head(11)


# ## Display the countries with largest estimated percent of the population that were infected
# #### NOTE: these are lower bounds, true numbers are likely substantially higher in most countries

# In[ ]:


all_countries_age_df['Percent Infected'] = 100 * all_countries_age_df['Total Infected'] / all_countries_age_df['Population']

cols_to_use = ['Country', 'Population', 'Total Infected', 'Total Deaths', 'Estimated Infection Fatality Ratio', 'Fraction age 65+ years', 'Num Tests per Positive Case', 'Percent Infected']

min_population_to_display = 1000000

all_countries_sorted_df = all_countries_age_df.sort_values(by=['Percent Infected'], ascending=False)[cols_to_use]
all_countries_sorted_df = all_countries_sorted_df[all_countries_sorted_df['Population'] > min_population_to_display].reset_index(drop=True)
#all_countries_sorted_df.head(22)


# In[ ]:


all_countries_sorted_clean_df = all_countries_sorted_df.copy()

all_countries_sorted_clean_df['Population (M)']          = all_countries_sorted_df['Population'] / 1000000
all_countries_sorted_clean_df['Total Infected (K)']      = all_countries_sorted_df['Total Infected'] / 1000
all_countries_sorted_clean_df['Total Deaths (K)']        = all_countries_sorted_df['Total Deaths'] / 1000
all_countries_sorted_clean_df['Estimated IFR (%)']       = 100 * all_countries_sorted_df['Estimated Infection Fatality Ratio']
all_countries_sorted_clean_df['Age 65+ years (%)']       = 100 * all_countries_sorted_df['Fraction age 65+ years']
all_countries_sorted_clean_df['Population Infected (%)'] = all_countries_sorted_df['Percent Infected']

num_decimals_to_show = {'Population (M)': 2, 'Total Infected (K)': 2, 
                         'Total Deaths (K)': 2, 'Estimated IFR (%)': 2,
                         'Age 65+ years (%)': 2, 'Num Tests per Positive Case': 2,
                         'Population Infected (%)': 2}

cols_to_display = ['Country', 'Population (M)', 'Total Infected (K)', 'Total Deaths (K)', 'Estimated IFR (%)', 'Age 65+ years (%)', 'Num Tests per Positive Case', 'Population Infected (%)']

all_countries_sorted_clean_df = all_countries_sorted_clean_df.round(num_decimals_to_show)[cols_to_display]

#all_countries_sorted_clean_df.head(31).style.background_gradient(subset=['Total Infected (K)', 'Total Deaths (K)', 'Population Infected (%)'], cmap='jet')
all_countries_sorted_clean_df.head(31).style.background_gradient(subset=['Population Infected (%)'], cmap='jet')


# ## Note that many countries have macroscopic infection rates
# In Peru for example, at least 1 of every 12 people was infected, and at least 1 of every 200 people has died
# In Belgium for example, at least 1 of every 22 people was infected, and at least 1 of every 1200 people has died

# # Is COVID-19 pandemic over?
# For anwering this question, we use also the [dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset) by Johns Hopkins University uploaded by [SRK](https://www.kaggle.com/sudalairajkumar) as it contains all data since the begining of the pandemic and not only since mid april as we have in this dataset

# ### Fist, Plot the daily cases for several selected countries 

# In[ ]:


countries_to_plot = ['Australia', 'Japan', 'Israel', 'India', 'USA', 'Uzbekistan', 'Spain']
num_countries = len(countries_to_plot)

plt.figure(figsize=(15,25))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.15)
for k, country_name in enumerate(countries_to_plot):

    country_df = worldometer_df.loc[worldometer_df['Country'] == country_name, :]
    country_df = country_df.reset_index(drop=True)

    plt.subplot(num_countries,1,k+1); plt.bar(country_df['Date'], country_df['Total Cases'].diff())
    plt.title(country_name, fontsize=20)
    if k < num_countries-1:
        plt.xticks([], [])
    else:
        plt.xticks(rotation=90)
    plt.ylabel('daily new cases')


# # Build a country X time matrix for cases, deaths and tests 

# In[ ]:


min_number_of_cases = 10000
greatly_affected_df = last_date_df.loc[last_date_df['Total Cases'] > min_number_of_cases,:]

list_of_all_countries = greatly_affected_df['Country'].unique().tolist()
list_of_all_dates = worldometer_df['Date'].unique().tolist()

tests_matrix = np.zeros((len(list_of_all_countries), len(list_of_all_dates)))
cases_matrix = np.zeros((len(list_of_all_countries), len(list_of_all_dates)))
deaths_matrix = np.zeros((len(list_of_all_countries), len(list_of_all_dates)))

for k, country_name in enumerate(list_of_all_countries):

    country_df = worldometer_df.loc[worldometer_df['Country'] == country_name, :]
    country_df = country_df.reset_index(drop=True)
    
    tests_matrix[k,:]  = country_df['Total Tests']
    cases_matrix[k,:]  = country_df['Total Cases']
    deaths_matrix[k,:] = country_df['Total Deaths']
    
tests_matrix[np.isnan(tests_matrix)] = 0
cases_matrix[np.isnan(cases_matrix)] = 0
deaths_matrix[np.isnan(deaths_matrix)] = 0

#print(list_of_all_countries)


# ### Display the normalized daily tests, cases, deaths for majorly affected countries

# In[ ]:


tests_matrix_norm  = tests_matrix  / np.tile(np.maximum(100, tests_matrix.max(axis=1, keepdims=True)), (1, tests_matrix.shape[1]))
cases_matrix_norm  = cases_matrix  / np.tile(np.maximum(100, cases_matrix.max(axis=1, keepdims=True)), (1, cases_matrix.shape[1]))
deaths_matrix_norm = deaths_matrix / np.tile(np.maximum(100, deaths_matrix.max(axis=1, keepdims=True)), (1, deaths_matrix.shape[1]))

daily_tests_matrix_norm = np.diff(tests_matrix_norm, axis=1)
daily_cases_matrix_norm = np.diff(cases_matrix_norm, axis=1)
daily_deaths_matrix_norm = np.diff(deaths_matrix_norm, axis=1)

from scipy import signal

window_std = 4
window_size = 11
filtering_window_tests = signal.gaussian(window_size, std=window_std)[np.newaxis]
filtering_window_tests /= filtering_window_tests.sum()

window_std = 2.5
window_size = 11
filtering_window = signal.gaussian(window_size, std=window_std)[np.newaxis]
filtering_window /= filtering_window.sum()


daily_tests_matrix_norm_smoothed = signal.fftconvolve(daily_tests_matrix_norm, filtering_window_tests, mode='valid')
daily_cases_matrix_norm_smoothed = signal.fftconvolve(daily_cases_matrix_norm, filtering_window, mode='valid')
daily_deaths_matrix_norm_smoothed = signal.fftconvolve(daily_deaths_matrix_norm, filtering_window, mode='valid')

daily_tests_matrix_norm_smoothed[daily_tests_matrix_norm_smoothed > 0.04] = 0.04
daily_cases_matrix_norm_smoothed[daily_cases_matrix_norm_smoothed > 0.04] = 0.04
daily_deaths_matrix_norm_smoothed[daily_deaths_matrix_norm_smoothed > 0.04] = 0.04

daily_tests_matrix_norm_smoothed[daily_tests_matrix_norm_smoothed < 0] = 0
daily_cases_matrix_norm_smoothed[daily_cases_matrix_norm_smoothed < 0] = 0
daily_deaths_matrix_norm_smoothed[daily_deaths_matrix_norm_smoothed < 0] = 0

plt.figure(figsize=(15,15));
plt.subplot(3,1,1); plt.plot(daily_tests_matrix_norm_smoothed.T); plt.title('tests', fontsize=20);
plt.subplot(3,1,2); plt.plot(daily_cases_matrix_norm_smoothed.T); plt.title('cases', fontsize=20);
plt.subplot(3,1,3); plt.plot(daily_deaths_matrix_norm_smoothed.T); plt.title('deaths', fontsize=20);


# ## Show normalized (test, cases, deaths) per country

# In[ ]:


plt.figure(figsize=(15,50));

for k in range(15):
    plt.subplot(15, 1, k + 1); plt.title(list_of_all_countries[k], fontsize=20)
    plt.plot(daily_tests_matrix_norm_smoothed[k,:].T, color='g');
    plt.plot(daily_cases_matrix_norm_smoothed[k,:].T, color='orange');
    plt.plot(daily_deaths_matrix_norm_smoothed[k,:].T, color='r');
    plt.legend(['tests','cases','deaths'])
    

