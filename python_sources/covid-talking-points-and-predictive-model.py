#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Analysis
# 
# COVID-19 is perhaps the biggest global event of the past 20 years. The deaths, the entire shutdown of the global economy, and what are most likely lasting changes in day to day life have had a profound effect on all of us. Through all of this, one of the most dangerous aspects in our stay at home lives is the non-data driven opinions we are being fed. So this notebook is an attempt at addressing some of these points and looking at a data driven view to give some clarity in murkiness that is our new normal. 
# 
# Specifically we will address the following four questions:
# 
#     1. How is the USA dealing with the virus compared to other countries ?
#     2. Is Sweden's approach to not lockdown effective ?
#     3. Does weather have an effect on deaths/cases ?
#     4. Can we build a predictive model?
#    

# # The Data
# 
# For this analysis we will be relying on three major data sources 
# 
# Population data - Provided from the UN
# 
# https://population.un.org/wpp/Download/Standard/CSV/
# 
# Testing data and Cases/Deaths data - Provided from ourworldindata.org (their sources are shown in their github)
# 
# https://github.com/owid/covid-19-data/tree/master/public/data
# 
# Covid cases and weather data - Provided from Kaggle and NOAA
# 
# https://www.kaggle.com/davidbnn92/weather-data-for-covid19-data-analysis
# 
# https://www.kaggle.com/c/covid19-global-forecasting-week-4
# 
# https://www.kaggle.com/noaa/gsod
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#loading loading weather data
weatherAndCases = pd.read_csv('../input/covidandpopulationdata/training_data_with_weather_info_week_4.csv')
weatherAndCases.head()


# In[ ]:


#loading testing data 
covidTestingData = pd.read_csv('../input/covidandpopulationdata/owid-covid-data.csv')
covidTestingData.head()


# In[ ]:


#poplulation data 
populationdata = pd.read_csv('../input/covidandpopulationdata/WPP2019_TotalPopulationBySex.csv')
populationdata.head()


# # Data cleaning
# 
# Obviously in it's current state we can't use any of this data so we'll have to clean it.

# In[ ]:


# all of our data is in 2020 and some columns are superflous 
populationdata = populationdata[populationdata['Time']==2020]
populationdata = populationdata[['Location','Time','PopMale','PopFemale','PopTotal','PopDensity']]
populationdata = populationdata.drop_duplicates()
populationdata.head()


# In[ ]:


#join the population and testing data
populationAndTesting = populationdata.merge(covidTestingData, left_on='Location', right_on='location')
populationAndTesting.head()


# In[ ]:


populationAndTesting.dtypes


# In[ ]:


populationAndTesting.head()


# In[ ]:


#clearly some countries dont match up so lets fix that

print(len(populationdata['Location'].unique()))
print(len(covidTestingData['location'].unique()))
print(len(populationAndTesting['Location'].unique()))
print(set(covidTestingData['location'])-set(populationAndTesting['Location']))


# In[ ]:


#most of these we can disregard since they are pretty small countries we wont use in our analyssi
#but some countries we cant look past
#United States, Hong Kong, Bolivia, Russia, etc.. 
#we will have to do a map and transform on these data points

def changecountryname(x):
    countryMap = {"United States of America":"United States",
             "China, Hong Kong SAR": "Hong Kong",
             "Bolivia (Plurinational State of)":"Bolivia",
             "Russian Federation":"Russia",
             "Viet Nam": "Vietnam",
             "China, Taiwan Province of China":"Taiwan",
             "Iran (Islamic Republic of)":"Iran",
             "Republic of Korea":"South Korea",
             "United Republic of Tanzania": "Tanzania",
             "Czechia":"Czech Republic",
             "Democratic Republic of the Congo":"Democratic Republic of Congo",
             "Syrian Arab Republic":"Syria",
             "Venezuela (Bolivarian Republic of)":"Venezuela"}
    if x in countryMap.keys():
        return countryMap[x]
    else:
        return x
populationdata['New_Country_Code'] = populationdata.Location.apply(lambda x: changecountryname(x))


# In[ ]:


populationdata.head()


# In[ ]:


#join the population and testing data
populationAndTesting = populationdata.merge(covidTestingData, left_on='New_Country_Code', right_on='location')
populationAndTesting.head()


# In[ ]:


#Test new missing set
print(set(covidTestingData['location'])-set(populationAndTesting['New_Country_Code']))


# In[ ]:


populationAndTesting.dtypes


# In[ ]:


#the weather data we're good with but we will be adding some 
#population information to standardize our points
populationandweather = populationdata.merge(weatherAndCases, left_on='New_Country_Code', right_on='Country_Region')
populationandweather.head()


# In[ ]:


#clearly some countries dont match up so lets fix that
print(len(populationdata['Location'].unique()))
print(len(weatherAndCases['Country_Region'].unique()))
print(len(populationandweather['Location'].unique()))
print(set(weatherAndCases['Country_Region'])-set(populationandweather['New_Country_Code']))


# In[ ]:


#diamond princess and MS Zaandam are a cruise liners so we obvioulsy dont need that
#from this list to simplify things we're only going to change the some of the countries

def changecountryname2(x):
    countryMap = {
        "United States":"US",
        "Taiwan":"Taiwan*",
        "South Korea":"Korea, South",
        "Czech Republic":"Czechia"}
    if x in countryMap.keys():
        return countryMap[x]
    else:
        return x
populationdata['New_Country_Code_Weather'] = populationdata.New_Country_Code.apply(lambda x: changecountryname2(x))


# In[ ]:


#the weather data we're good with but we will be adding some 
#population information to standardize our points
populationandweather = populationdata.merge(weatherAndCases, left_on='New_Country_Code_Weather', right_on='Country_Region')
populationandweather.head()


# In[ ]:


#clearly some countries dont match up so lets fix that
print(len(populationdata['Location'].unique()))
print(len(weatherAndCases['Country_Region'].unique()))
print(len(populationandweather['Location'].unique()))
print(set(weatherAndCases['Country_Region'])-set(populationandweather['New_Country_Code_Weather']))


# # Variable Creation
# 
# 

# In[ ]:


from dateutil import parser
populationAndTesting['DTDate'] = populationAndTesting.date.apply(lambda x: parser.parse(x))
populationAndTesting.head()


# In[ ]:


populationAndTesting.dtypes


# In[ ]:


#cleaning up data
populationAndTesting = populationAndTesting[['Location','PopTotal','PopDensity',
                                             'total_cases','total_deaths',
                                             'total_tests','DTDate']]


# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_deaths", hue="Location",
                  data=populationAndTesting)


# Wow that is one ugly graph (Hiding the output for visibility purposes)
# 
# Lets get rid of a lot of the smaller countries for our analysis
# 
# We're only going to take the Top 25 most populous countries and base our anaylsis on those
# 

# In[ ]:


populationAndTesting = populationAndTesting.sort_values(by='PopTotal', ascending=False)
populationAndTesting.head()


# Interesting, we have world data. Lets save that for later

# In[ ]:


populationAndTestingWorld = populationAndTesting[populationAndTesting['Location']=='World']
populationAndTesting = populationAndTesting[populationAndTesting['Location']!='World']
populationAndTesting.head()


# In[ ]:


topCountries = populationAndTesting.Location.unique()[:25]
topCountries = np.concatenate((topCountries,['Sweden', 'Spain']))
topCountries


# In[ ]:


populationAndTestingTop = populationAndTesting[populationAndTesting['Location'].isin(topCountries)]
populationAndTestingTop.head()


# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_deaths", hue="Location",
                  data=populationAndTestingTop)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(ax.get_xticklabels(), fontsize=5)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# It seems like the us has an extreme amount of cases however 
# 
# this is decieving. The us is 3rd in population. But more importantly
# 
# lets look at the testing

# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_tests", hue="Location",
                  data=populationAndTestingTop)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(ax.get_xticklabels(), fontsize=5)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# In[ ]:


#lets standardize total deaths and cases by population 
#population is in 1000
populationAndTestingTop['total_cases_perpopulation'] = populationAndTestingTop['total_cases'] / (1000*populationAndTestingTop['PopTotal'])
populationAndTestingTop['total_tests_perpopulation'] = populationAndTestingTop['total_tests'] / (1000*populationAndTestingTop['PopTotal'])
populationAndTestingTop['total_deaths_perpopulation'] = populationAndTestingTop['total_deaths'] / (1000*populationAndTestingTop['PopTotal'])
populationAndTestingTop.head()


# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_cases_perpopulation", hue="Location",
                  data=populationAndTestingTop.dropna())
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(ax.get_xticklabels(), fontsize=5)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_tests_perpopulation", hue="Location",
                  data=populationAndTestingTop.dropna())
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(ax.get_xticklabels(), fontsize=5)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_deaths_perpopulation", hue="Location",
                  data=populationAndTestingTop)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(ax.get_xticklabels(), fontsize=5)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# It is clear that the US has vastly more tests than any other country.
# 

# In[ ]:


#lets control deaths and cases for tests
populationAndTestingTop['deaths_per_test'] = populationAndTestingTop['total_deaths'] / populationAndTestingTop['total_tests']
populationAndTestingTop.head()
set(populationAndTestingTop.dropna().Location)


# In[ ]:


ax = sns.lineplot(x="DTDate", y="deaths_per_test", hue="Location",
                  data=populationAndTestingTop.dropna())
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(ax.get_xticklabels(), fontsize=5)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# In[ ]:


corr = populationAndTestingTop.corr()
corr 


# In[ ]:


mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# It seems that oddly population density didnt have the same correlation on deaths as we predicted lets look at the full data set to see if this is still the case

# In[ ]:


corr = populationAndTesting.corr()
corr 


# In[ ]:


mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Interesting, population density isnt corelated with cases and deaths as strongly as we would have predicted. 
# This could be a result of testing data scarcity.

# In[ ]:


ax = sns.lineplot(x="PopDensity", y="deaths_per_test",
                  data=populationAndTestingTop.dropna())
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# Does seem oddly random. 

# # USA analysis
# 
# There has been a lot of media attention about how the US has handled the COVID situation. Many people seem to say that the death toll for COVID patients has been significantly higher in the US than in other countries. And while looking at raw numbers one can say that but, if we standardize our data we see a different story. There does seem to be a high degreee in testing data scarcity. Combine that with the fact countries are self reporting all these numbers, we can have a degree of skepticism when looking at this information. That being said we can paint a much different picture with this information than is being portrayed to us. 

# # The Sweden situation
# 
# The way that Sweden has handled the lockdown has been significanltly different from most countries. They haven't shutdown their economy. And though the US president has said they are 'paying heavily' for this decision. It is not obvious from the data that they are. So lets take a deeper dive and do a statistical test to see if Sweden is fairing marketabley different from other countries. 

# In[ ]:


#lets use our standarized deaths measure with the full dataset
populationAndTesting['total_deaths_perpopulation'] = populationAndTesting['total_deaths'] / (1000*populationAndTesting['PopTotal'])
populationAndTesting.head()


# In[ ]:


#lets create a new measure if a country is sweden or its not
populationAndTesting['Is_Sweden'] = populationAndTesting.Location.apply(lambda x: "Sweden" in x)
populationAndTesting.head()


# In[ ]:


set(populationAndTesting.Is_Sweden)


# In[ ]:


# load packages
import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(populationAndTesting[populationAndTesting['Is_Sweden']==True].total_deaths_perpopulation,
                               populationAndTesting[populationAndTesting['Is_Sweden']==False].total_deaths_perpopulation)
print(fvalue, pvalue)


# Populations do look different here but lets take into consideration that Sweden is a highly developed country. So comparing it's healthcare system to all the other countries in the world isn't necessarily fair. We could have a significan difference based purely on the fact that they have a better health care system than the majority of other countries.

# In[ ]:


#lets instead take a subset of some of the most populous top 20 developed countries as based on the HDI index.
#this seems to be a more fair comparison

developed = ['Norway', 'Ireland', 'Germany', 
            'Australia', 'Iceland', 'Sweden',
            'Singapore', 'Netherlands', 'Denmark',
            'Finland', 'Canada', 'New Zealand',
            'United Kingdom', 'United States of America']
populationAndTestingDeveloped = populationAndTesting[populationAndTesting['Location'].isin(developed)]
populationAndTestingDeveloped.head()


# In[ ]:


#test to see if we got them all
set(populationAndTestingDeveloped.Location)-set(populationAndTestingDeveloped.Location)
#cool it worked


# In[ ]:


fvalue, pvalue = stats.f_oneway(populationAndTestingDeveloped[populationAndTestingDeveloped['Is_Sweden']==True].total_deaths_perpopulation,
                               populationAndTestingDeveloped[populationAndTestingDeveloped['Is_Sweden']==False].total_deaths_perpopulation)
print(fvalue, pvalue)


# In[ ]:


ax = sns.lineplot(x="DTDate", y="total_deaths_perpopulation", hue="Is_Sweden",
                  data=populationAndTestingDeveloped.dropna())
plt.setp(ax.get_xticklabels(), fontsize=7)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')


# Sweden does seem to have a different death rate according to our ANOVA and graph than most other developed nations. We can say that we are fairly certain the group mean of Swedens death rate isn't the same as other develped nations. So perhaps their decision to not lockdown wasn't a good one.

# # Anova Assumptions
# 
# Looking at this we have to take into consideration the assumptions of an anova test.
# 
# The samples are independent.
# 
# Each sample is from a normally distributed population.
# 
# The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
# 
# Unfortunately we cannot say that we meet all of these assumptions. However considering the sample size we can look passed some of them

# # Weather
# 
# There is a current theory going around that the weather has an impact on the Coronavirus. That warm and humid places tend make the virus live a shorter period and and therby safer. We're going to test these variables importance on growth of the virus. We're also going to attempt to fit a model based on this. 
# 

# In[ ]:


#we have to address one annoying thing
populationandweather.dtypes


# In[ ]:


populationandweather = populationandweather[['Location','PopTotal','Province_State','Date',
                                             'ConfirmedCases','Fatalities','Lat','Long',
                                             'day_from_jan_first','temp','min',
                                             'max','stp','slp','dewp','rh',
                                             'ah','wdsp','prcp','fog']]
populationandweather.head()


# In[ ]:


set(populationandweather[populationandweather['Province_State'].notnull()].Location)


# In[ ]:


#Some of our data is more modular so we have to use state/province level data for 
#Australia, Canada, China, United States of America. The rest are just islands so we can honestly drop those

#australia we scraped from
#https://www.abs.gov.au/ausstats/abs@.nsf/mediareleasesbyCatalogue/CA1999BAEAA1A86ACA25765100098A47

#canada 
#https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901

#china
#http://data.stats.gov.cn/english/easyquery.htm?cn=E0103

#usa
#https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html
australia = pd.read_csv('../input/covidandpopulationdata/australia.csv')
china = pd.read_csv('../input/covidandpopulationdata/china.csv')
canada = pd.read_csv('../input/covidandpopulationdata/canada.csv')
usa = pd.read_csv('../input/covidandpopulationdata/usa.csv')


# In[ ]:


populationandweather_usa = populationandweather[populationandweather['Location']=='United States of America']
populationandweather_canada = populationandweather[populationandweather['Location']=='Canada']
populationandweather_china = populationandweather[populationandweather['Location']=='China']
populationandweather_australia = populationandweather[populationandweather['Location']=='Australia']


# In[ ]:


populationandweather = populationandweather[populationandweather['Province_State'].isnull()]


# In[ ]:


populationandweather_usa.head()


# In[ ]:


usa.head()


# In[ ]:


#join the population and testing data
populationandweather_usa = populationandweather_usa.merge(usa, left_on='Province_State', right_on='state')
populationandweather_usa.head()


# In[ ]:


populationandweather_usa['PopTotal'] = populationandweather_usa['population']
populationandweather_usa = populationandweather_usa[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',
       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',
       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]
populationandweather_usa.head()


# In[ ]:


#test to see if we got them 
set(populationandweather_usa.Province_State) - set(usa.state)


# In[ ]:


china = china[['State','Population']]
china.head()


# In[ ]:


populationandweather_china.head()


# In[ ]:


#join the population and testing data
populationandweather_china = populationandweather_china.merge(china, left_on='Province_State', right_on='State')
populationandweather_china.head()


# In[ ]:


populationandweather_china['PopTotal'] = populationandweather_china['Population']
populationandweather_china = populationandweather_china[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',
       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',
       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]
populationandweather_china.head()


# In[ ]:


#test to see if we got them 
set(populationandweather_china.Province_State) - set(china.State)


# In[ ]:


australia = australia[['State','population']]
australia.head()


# In[ ]:


print(set(populationandweather_australia.Province_State))
populationandweather_australia.head()


# In[ ]:


#join the population and testing data
populationandweather_australia = populationandweather_australia.merge(australia, left_on='Province_State', right_on='State')
populationandweather_australia.head()


# In[ ]:


populationandweather_australia['PopTotal'] = populationandweather_australia['population']
populationandweather_australia = populationandweather_australia[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',
       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',
       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]
populationandweather_australia.head()


# In[ ]:


#test to see if we got them 
set(populationandweather_australia.Province_State) - set(australia.State)


# In[ ]:


canada = canada[['Province','Population']]
canada.head()


# In[ ]:


#join the population and testing data
populationandweather_canada = populationandweather_canada.merge(canada, left_on='Province_State', right_on='Province')
populationandweather_canada.head()


# In[ ]:


populationandweather_canada['PopTotal'] = populationandweather_canada['Population']
populationandweather_canada = populationandweather_canada[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',
       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',
       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]
populationandweather_canada.head()


# In[ ]:


finalweather = populationandweather.append([populationandweather_canada,populationandweather_usa,
                                           populationandweather_australia,populationandweather_china])
finalweather.head()


# In[ ]:


finalweather['PopTotal']  = finalweather['PopTotal'].replace(',','', regex=True)
finalweather['PopTotal'] = finalweather['PopTotal'].astype(float)
finalweather['ConfirmedCasesPerCapita'] = finalweather['ConfirmedCases']/finalweather['PopTotal']
finalweather['DeathsPerCapita'] = finalweather['Fatalities']/finalweather['PopTotal']


# In[ ]:


finalweather['DTDate'] = finalweather.Date.apply(lambda x: parser.parse(x))


# In[ ]:


finalweather.columns


# In[ ]:


finalweather = finalweather[['PopTotal', 'ConfirmedCases','Fatalities', 
                             'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',
                             'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog',
       'ConfirmedCasesPerCapita', 'DeathsPerCapita']]


# # Impute missing data 

# In[ ]:


#impute missing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[ ]:


finalweather = finalweather.replace([np.inf, -np.inf], np.nan)


# In[ ]:


imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(finalweather)
IterativeImputer(random_state=0)
x = imp.fit_transform(finalweather)
# x.head()
temp = pd.DataFrame(x, columns=finalweather.columns)
temp.head()


# # Scale our data 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
#load scaler
scaler = MinMaxScaler()
scaler.fit(temp)
scaled =scaler.fit_transform(temp) 


# In[ ]:


scaleddf = pd.DataFrame(scaled, columns=finalweather.columns)
scaleddf.head()


# In[ ]:


scaleddf.dtypes


# In[ ]:


for i in scaleddf.columns:
    scaleddf[i] = scaleddf[i].astype(int)


# In[ ]:


scaleddf.dtypes


# # Testing feature importance
# 
# First lets test fatalities per capita

# In[ ]:



from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

y = scaleddf['DeathsPerCapita'] 
X = scaleddf[['ConfirmedCases', 'Lat', 'Long', 
                  'day_from_jan_first', 'temp', 'min',
                  'max','stp', 'slp', 'dewp', 'rh', 'ah',
                  'wdsp', 'prcp', 'fog']]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# ### So obviously the most important feature is days since january first because deaths are going to be higher as time goes on.
# 
# ### Interestingly stp- or maximum temperature reported during the day in Fahrenheit to tenths--time of max temp report varies by country and region, so this will sometimes not be the max for the calendar day. Also seemed important

# In[ ]:


#lets do the same check but for cases
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

y = scaleddf['ConfirmedCases'] 
X = scaleddf[['DeathsPerCapita', 'Lat', 'Long', 
                  'day_from_jan_first', 'temp', 'min',
                  'max','stp', 'slp', 'dewp', 'rh', 'ah',
                  'wdsp', 'prcp', 'fog']]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# ### So again the most important feature is days since january first.
# 
# ### We had the same result as before but stp is significantly less than it was before

# # Fit to a predictive model

# ### Start with simple regression model

# In[ ]:


scaleddf = pd.DataFrame(scaled, columns=finalweather.columns)
scaleddf.head()


# In[ ]:


scaleddf.columns


# In[ ]:


scaleddf.head()


# In[ ]:


#create a testing and training model 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaleddf.drop(['DeathsPerCapita'],axis=1), scaleddf['DeathsPerCapita'], test_size=0.2, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate
import statistics 

print(mean_absolute_error(pred, y_test))
scores = cross_validate(reg, X_test, y_test, cv=10,
                        scoring=('neg_root_mean_squared_error'),
                        return_train_score=True)  
statistics.mean(abs(scores['test_score']))


# ## Lets throw some machine learning in here for fun

# In[ ]:


#multi layer perceptron

from sklearn.neural_network import MLPRegressor
neuralNetwork = MLPRegressor(
    hidden_layer_sizes=(550, 550),
    shuffle=True, activation='relu',
    learning_rate='adaptive')

neuralNetwork.fit(X_train, y_train)

pred_y_test = neuralNetwork.predict(X_test)
pred_y_train = neuralNetwork.predict(X_train)


# In[ ]:


print(mean_absolute_error(pred_y_test, y_test))
scores = cross_validate(neuralNetwork, X_test, y_test, cv=10,
                        scoring=('neg_root_mean_squared_error'),
                        return_train_score=True)  
statistics.mean(abs(scores['test_score']))


# ## Decent performance
# 0.01104385921680685
# average of root mean squared error on testing data after 10 fold cross validation

# ### Lets try tensorflow

# In[ ]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# In[ ]:


def build_model():
  model = keras.Sequential([
    layers.Dense(90, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(90, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[ ]:


model = build_model()
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=50)


# In[ ]:


model.summary()


# In[ ]:


# test_predictions = history.predict(X_test).flatten()
test_predictions = model.predict(X_test).flatten()

print(mean_absolute_error(test_predictions, y_test))


# ## Wow thats pretty accurate

# In[ ]:




