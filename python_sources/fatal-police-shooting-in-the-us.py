#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# In this notebook, I'm going to analyze the fatal fatal policing shooting in the US to see how many factors were related to the race of the victims. I'm also going to build models to predict the race ofthe victims based on other information given in the datasets.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


over_25_completed_hs = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
people_below_poverty = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
median_household_income = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")
share_race_by_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
fatal_police_shooting_us = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")


# # **Processing**

# Drop missing rows and convert string to float64

# In[ ]:


# dropping rows with missing percent_completed_hs
over_25_completed_hs = over_25_completed_hs[over_25_completed_hs['percent_completed_hs'] != '-']
over_25_completed_hs['percent_completed_hs'] = over_25_completed_hs['percent_completed_hs'].astype('float64')

people_below_poverty = people_below_poverty[people_below_poverty['poverty_rate'] != '-']
people_below_poverty['poverty_rate'] = people_below_poverty['poverty_rate'].astype('float64')

median_household_income = median_household_income[median_household_income['Median Income'] != '(X)']
median_household_income = median_household_income[median_household_income['Median Income'] != '-']
median_household_income = median_household_income[median_household_income['Median Income'].notna()]

# drop 250000+ and 2500-
median_household_income = median_household_income[~median_household_income['Median Income'].str.contains('-')]
median_household_income = median_household_income[~median_household_income['Median Income'].str.contains('+', regex=False)]

median_household_income['Median Income'] = median_household_income['Median Income'].astype('float64')

share_race_by_city = share_race_by_city[share_race_by_city['share_white']!='(X)']
share_race_by_city['share_white'] = share_race_by_city['share_white'].astype('float64')

share_race_by_city['share_black'] = share_race_by_city['share_black'].astype('float64')
share_race_by_city['share_native_american'] = share_race_by_city['share_native_american'].astype('float64')
share_race_by_city['share_asian'] = share_race_by_city['share_asian'].astype('float64')
share_race_by_city['share_hispanic'] = share_race_by_city['share_hispanic'].astype('float64')

fatal_police_shooting_us['date'] = pd.to_datetime(fatal_police_shooting_us['date'], format='%d/%m/%y')
fatal_police_shooting_us = fatal_police_shooting_us.dropna()


# Combine over_25_completed_hs, people_below_poverty, median_household_income, share_race_by_city to a demographic dataset

# In[ ]:


a = [people_below_poverty.set_index(['Geographic Area', 'City']), median_household_income.set_index(['Geographic Area', 'City']), share_race_by_city.set_index(['Geographic area', 'City'])]
demographic = over_25_completed_hs.set_index(['Geographic Area', 'City']).join(other=a).reset_index()
demographic.info()


# Trying to combine demographic with fatal_police_shooting to make a single dataset.
# We need to fix the mitmatch in city naming convention between fatal_police_shooting and demographic

# In[ ]:


demographic['City'] = demographic['City'].str.replace(' city.*','')
demographic['City'] = demographic['City'].str.replace(' CDP.*','')
demographic['City'] = demographic['City'].str.replace(' town.*','')
demographic = demographic.rename(columns={'Geographic Area':'state', 'City':'city'})

data = demographic.set_index(['state','city']).join(fatal_police_shooting_us.set_index(['state','city'])).reset_index()
data = data.dropna(axis=0)
data.info()


# # **EDA**

# **Demographics**
# 
# ---
# 
# 

# In[ ]:


demographic.describe()


# *   85.7% of people over 25 completed high school
# *   Average median income is 51177
# *   16.4% of people are in poverty

# In[ ]:


average_share_per_race = pd.DataFrame({'race':['white','black','native_american','asian','hispanic'],
                                       'average_share':[83.190149,6.882655,2.856685	,1.547159,9.203426]})
average_share_per_race = average_share_per_race.sort_values(by='average_share')
plt.figure(figsize=(10,6))
plt.title('Average race share in US', fontdict={'fontsize':15})
sns.barplot(x=average_share_per_race['race'], y=average_share_per_race['average_share'])


# It is clear that white accounts for the majority of the population in the US

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(20,5), sharey=True)

#ax[0].ticklabel_format(style='plain')
sns.distplot(demographic['Median Income'], ax=ax[0], kde=False)
ax[0].set_title('Median Income', fontdict={'fontsize': 15})

sns.distplot(demographic['poverty_rate'], ax=ax[1], kde=False)
ax[1].set_title('Poverty Rate', fontdict={'fontsize': 15})

sns.distplot(demographic['percent_completed_hs'], ax=ax[2], kde=False)
ax[2].set_title('Percent Completed High School', fontdict={'fontsize': 15})


# In[ ]:


plt.title('Poverty rate and Median Income', fontdict={'fontsize':15})
sns.regplot(x = demographic['Median Income'], y = demographic['poverty_rate'], order=10, scatter=False)


# There is a reasonable trend between median income and poverty rate of each city.

# **Fatal police shooting US**
# 
# ---

# In[ ]:


fatal_police_shooting_us['date'].sort_values()


# The data was collected from 2015-01-02 till 2017-07-31

# In[ ]:


grid = sns.FacetGrid(fatal_police_shooting_us,col='flee',col_order=['Foot','Car','Not fleeing','Other'])
grid.map(sns.boxplot, 'age', orient='vertical', color='lavender')


# Age of the victims doesn't seem to have much influence on their fleeing method.

# In[ ]:


a=pd.pivot_table(data=fatal_police_shooting_us, values='age', index='gender', columns='race')
a=a.reindex(columns=['W','A','O','B','N','H'])
plt.title('Average age for each gender and race')
sns.heatmap(a, mask=a.isnull(), linewidth=0.01, linecolor='white')
fatal_police_shooting_us[(fatal_police_shooting_us['gender']=='F') & (fatal_police_shooting_us['race']=='H')]


# !Note:
# 
# *   W: White
# *   B: Black
# *   A: Asian
# *   H: Hispanic
# *   N: Native American
# *   O: Other
# 
# White vitims in average are the oldest, whereas Native American and Hispanic are the youngest.
# For all races except White and Hispanic, Male vitims tend to be younger.
# 
# Female Hispanic are noticably younger than the others. So I tried to select cases in which Female Hispanic were victimes. As a result, there was only 4 cases. The sample is too small to make any inference.

# In[ ]:


a = fatal_police_shooting_us['signs_of_mental_illness'].value_counts()
plt.title('Victims with sign of mental illness',fontdict={'fontsize':15})
sns.barplot(x= a.index, y = a)


# Mental illness of the victim can be a factor that accounts for the decision of the police officers. About 3/8 of the victims had sign of mental illness.

# In[ ]:


a = fatal_police_shooting_us['threat_level'].value_counts()
plt.title('Victims\' status immediately before the fatal shots',fontdict={'fontsize':15})
sns.barplot(x= a.index, y = a)


# About 2/3 of the victims were attacking the police before getting shot. This could also influence the officers' decisions.

# In[ ]:


print('Number of unique weapons armed by the victims: {}'.format(len(fatal_police_shooting_us['armed'].unique())))

count_per_weapon = fatal_police_shooting_us['armed'].value_counts()
plt.figure(figsize=(15,7))
plt.title('Top 10 weapons armed by the victims count',fontdict={'fontsize':17})
sns.barplot(x=count_per_weapon[:10].index, y=count_per_weapon[:10])


# Most of the victims were having gun before getting shot. Only about 200 of them were unarmed. This raises another question: What were the victims doing with those weapons before the incidence?

# In[ ]:


a = fatal_police_shooting_us.groupby(['threat_level','flee']).size()
a = a.reset_index().pivot(index='threat_level',columns='flee',values = 0)
plt.figure(figsize=(10,5))
sns.heatmap(a)


# Most of the victims chose not to flee and attack the officers. This helps explain the decision of the officers.

# In[ ]:


a=fatal_police_shooting_us.groupby('race').size().sort_values()
plt.figure(figsize=(10,5))
sns.barplot(x=a.index, y=a)
plt.title('Total number of people killed',fontdict={'fontsize':20})


# In this chart, we can see that most of the victims were White. However, since the total population of each race in the US is different, especially for White taking a large portion of the US population, a bar plot with the percent people being killed by race would be a better visualization.

# In[ ]:


a = fatal_police_shooting_us.groupby('race').size().sort_values()
percent_death_per_race = list()
# these are the population for each race in 2017 collected from https://en.wikipedia.org/wiki/Demographics_of_the_United_States
total_population_per_race = {
    'White':197277789,
    'NativeAmerican':2098763,
    'Asian':16989540,
    'Hispanic':56510571,
    'Black':39445495,
    'Others':8166727
}

for race, total_death in a.iteritems():
  if race == 'W':
    death_per_race = total_death/total_population_per_race['White']*100
  elif race == 'N':
    death_per_race = total_death/total_population_per_race['NativeAmerican']*100
  elif race == 'A':
    death_per_race = total_death/total_population_per_race['Asian']*100
  elif race == 'H':
    death_per_race = total_death/total_population_per_race['Hispanic']*100
  elif race == 'B':
    death_per_race = total_death/total_population_per_race['Black']*100
  else:
    death_per_race = total_death/total_population_per_race['Others']*100
  percent_death_per_race.append((race, death_per_race))
percent_death_per_race = pd.DataFrame(percent_death_per_race)
plt.figure(figsize=(10,5))
sns.barplot(x=percent_death_per_race[0], y=percent_death_per_race[1])
plt.title('Percent of people per race killed',fontdict={'fontsize':20})


# This chart shows that Black and Native American are much more likely to be shot. The proportion and Black being shot is almost three times higher than White.
# 
# However, this chart still have a caveat: total death per race divided by the total population per race may not be a good representation for the racial bias. 
# 
# Since most of these cases happened when the victims were suspected or alledged to be criminal, taking total death divided by the total population doesn't illustrate the racial bias of the police toward the criminals. Instead, the denominator should be some value that represent the number of crimes of that race.
# 
# Assuming that all of the police making these fatal shots were all properly trained, we can say that these victims were making some act of crimes during the incident.
# 
# In this case, I set the denominator to be total arrest cases for each city in the US. Although death divided arrest doesn't give us any concrete value (people cannot be both shot and arrested), total arrest cases are a good representation of crimes in an area and therefore, the fraction would give us a better estimate about fatal shooting cases and criminals in each city.

# In[ ]:


''' These data represent the total arrest in 2015, 2016, 2017: 
  https://ucr.fbi.gov/crime-in-the-u.s/2015/crime-in-the-u.s.-2015/tables/table-43
  https://ucr.fbi.gov/crime-in-the-u.s/2016/crime-in-the-u.s.-2016/topic-pages/tables/table-21
  https://ucr.fbi.gov/crime-in-the-u.s/2017/crime-in-the-u.s.-2017/tables/table-43
'''
a = fatal_police_shooting_us.groupby('race').size().sort_values()
percent_death_per_race_criminal = list()

total_criminal_cases={
    'White': 5753212*0.816 + 5858330*0.816 + 7/12*0.819*5626140,
    'Black': 2197140*0.816 + 2263112*0.816 + 7/12*0.819*2221697, 
    'Asian': 101064*0.816 + 103244*0.816 + 7/12*0.819*97049,
    'NativeAmerican': 174020*0.816 + 171185*0.816 + 7/12*0.819*196908,
    'Hispanic': 1204862 + 1221066 + 7/12*1190671,
    'Others': 23273*0.816 + 25610*0.816 + 7/12*0.819*21055
}

for race, total_death in a.iteritems():
  if race == 'W':
    death_per_race_criminal = total_death/total_criminal_cases['White']*100
  elif race == 'N':
    death_per_race_criminal = total_death/total_criminal_cases['NativeAmerican']*100
  elif race == 'A':
    death_per_race_criminal = total_death/total_criminal_cases['Asian']*100
  elif race == 'H':
    death_per_race_criminal = total_death/total_criminal_cases['Hispanic']*100
  elif race == 'B':
    death_per_race_criminal = total_death/total_criminal_cases['Black']*100
  else:
    death_per_race_criminal = total_death/total_criminal_cases['Others']*100
  percent_death_per_race_criminal.append((race, death_per_race_criminal))

percent_death_per_race_criminal = pd.DataFrame(percent_death_per_race_criminal)
plt.figure(figsize=(10,5))
sns.barplot(x=percent_death_per_race_criminal[0], y=percent_death_per_race_criminal[1])
plt.title('Percent of criminals per race killed',fontdict={'fontsize':20})


# Calculating steps:
# * Since the datasets provided from FBI:UCR don't have categories for non-Hispanic of each of the other races, I calculated that by multiplying total cases of each race by the percentage of non-Hispanic cases, assuming that each of the other races shares the same proportion of Hispanic/non-Hispanic.
# * Note that the FatalPoliceShootingUS only have records from Jan 2015 to July 2017. However, the datasets from FBI:UCR have the total cases for each year 2015, 2016, and 2017. Therefore, we have to add the total of each of the year to find the total cases during the period. Besides, for 2017, assuming number of cases are not affected by the time of a year, I multiply the total cases in 2017 by 7/12 to get the estimate of total cases in the first 7 months of 2017.
# 
# Results:
# * The percentage of criminals being shot for the Others race is disproportionally high.
# * Among the other races, although there are differences between the proportion of each race being killed, the bias is not significant.

# In[ ]:


fatal_police_shooting_us['month'] = fatal_police_shooting_us['date'].dt.month#.value_counts()
fatal_police_shooting_us['month'] = fatal_police_shooting_us['month'].astype('str')
fatal_police_shooting_us['month'] = fatal_police_shooting_us['month'].map(lambda x: '0'+ x if len(x)==1 else x)

fatal_police_shooting_us['year'] = fatal_police_shooting_us['date'].dt.year

fatal_police_shooting_us['year-month'] = fatal_police_shooting_us['year'].astype('str') + '-' + fatal_police_shooting_us['month'].astype('str')
shooting_per_month = fatal_police_shooting_us.groupby('year-month').size()

fig, ax = plt.subplots(figsize=(30,7))
ax.set_title('Total victims each month from January 2015 to July 2017', fontdict={'fontsize':25})
sns.lineplot(ax=ax, x=shooting_per_month.index, y=shooting_per_month)

###########
shooting_per_month = pd.DataFrame(shooting_per_month)
shooting_per_month['average_per_month'] = shooting_per_month[0].mean()

sns.lineplot(ax=ax, x=shooting_per_month.index, y=shooting_per_month['average_per_month'])
ax.legend(labels=['Victims per month','Average'])


# Time and total death are not correlated. This backs up the assumption made in the previous calculation.

# In[ ]:


a = fatal_police_shooting_us.groupby('state').size().sort_values(ascending=False)
plt.figure(figsize=(20,5))
plt.title('Total fatal shots in each state', fontdict={'fontsize':20})
sns.barplot(x = a.index, y = a)


# CA, TX, FL and have most police fatal shooting cases
# 
# 
# Same as cases per race, for this cases per city, we also need to divide the number by total arrest cases per city.
# 
# 
# https://ucr.fbi.gov/crime-in-the-u.s/2015/crime-in-the-u.s.-2015/tables/table-69
# 
# https://ucr.fbi.gov/crime-in-the-u.s/2016/crime-in-the-u.s.-2016/tables/table-22
# 
# https://ucr.fbi.gov/crime-in-the-u.s/2017/crime-in-the-u.s.-2017/topic-pages/tables/table-69

# In[ ]:


arrest_by_state = pd.read_csv('../input/arrest-by-state/arrest_by_state.csv')

arrest_by_state['2015'] = arrest_by_state['2015'].str.replace(',','')
arrest_by_state['2,016'] = arrest_by_state['2,016'].str.replace(',','')
arrest_by_state['2,017'] = arrest_by_state['2,017'].str.replace(',','')
arrest_by_state['2015'] = arrest_by_state['2015'].astype('int64')
arrest_by_state['2,016'] = arrest_by_state['2,016'].astype('int64')
arrest_by_state['2,017'] = arrest_by_state['2,017'].astype('int64')

arrestfrom2015to2017 = arrest_by_state['2015'] + arrest_by_state['2,016'] + arrest_by_state['2,017']*7/12
arrestfrom2015to2017.index = arrest_by_state['Unnamed: 0'].str.strip()

#https://gist.github.com/rogerallen/1583593

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York6': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

us_state_abbrev = { state.upper(): ab for state, ab in us_state_abbrev.items()}
arrestfrom2015to2017.index = arrestfrom2015to2017.index.map(us_state_abbrev)
death_per_arrest = fatal_police_shooting_us.groupby('state').size().sort_index() / arrestfrom2015to2017.sort_index()
plt.figure(figsize=(20,5))
plt.title('Fatal shot per arrest in each state', fontdict={'fontsize':20})
sns.barplot(x = death_per_arrest.sort_values().index, y = death_per_arrest.sort_values(ascending=False))


# This chart illustrates better the aggressivity of the police officers. Officers in CT, RI, and NJ are most likely to cause fatal shots.

# # **Modeling To Predict Race Of Victims**

# In[ ]:


from sklearn import preprocessing
from sklearn import model_selection
from sklearn import inspection
from sklearn import metrics
from sklearn import ensemble

FEATURE_RACE = ['percent_completed_hs','poverty_rate','Median Income',
               'share_white','share_black','share_asian',
               'share_native_american','share_hispanic','manner_of_death', 
               'armed', 'gender', 'age', 'signs_of_mental_illness', 
               'threat_level']
TARGET_RACE = 'race'

X_race = data[FEATURE_RACE]
y_race = data[TARGET_RACE]


# In[ ]:


object_cols = (X_race.dtypes == 'object')[X_race.dtypes == 'object'].index
X_race[object_cols].describe()

RACE_OH_FEATURES = ['threat_level', 'armed']  # since these variables are nominal
RACE_LABEL_FEATURES = ['manner_of_death', 'gender', 'signs_of_mental_illness'] # since these are binary variables


# In[ ]:


# reduce categories in 'armed' column by removing rarely appeared (<=7) values
s = X_race['armed'].value_counts() 
armed_others = list(s[s<=7].index) 
X_race['armed'] = X_race['armed'].apply(lambda x: 'others' if x in armed_others else x)
X_race['armed'].value_counts()


# In[ ]:


TRAIN_SIZE = 0.8
TEST_SIZE = 0.2

# splitting train/test
X_race_train, X_race_test, y_race_train, y_race_test  = model_selection.train_test_split(X_race,y_race,train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=0)

# One Hot Encoding 
encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)

X_race_OH_train_cols = pd.DataFrame(encoder.fit_transform(X_race_train[RACE_OH_FEATURES]))
X_race_OH_test_cols = pd.DataFrame(encoder.transform(X_race_test[RACE_OH_FEATURES]))

X_race_train = X_race_train.drop(RACE_OH_FEATURES, axis=1)
X_race_test = X_race_test.drop(RACE_OH_FEATURES, axis=1)

X_race_OH_train_cols.columns = encoder.get_feature_names()
X_race_OH_test_cols.columns = encoder.get_feature_names()

X_race_OH_train_cols.index = X_race_train.index
X_race_OH_test_cols.index = X_race_test.index

X_race_train = pd.concat([X_race_train, X_race_OH_train_cols], axis=1)
X_race_test = pd.concat([X_race_test, X_race_OH_test_cols], axis=1)

# Label Encoding
encoder = preprocessing.LabelEncoder()

for col in RACE_LABEL_FEATURES:
  X_race_train[col] = encoder.fit_transform(X_race_train[col])
  X_race_test[col] = encoder.transform(X_race_test[col])


# **KNN Classifier**
# 
# ---
# 
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

race_knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 10, 13]}

gridsearch_race_knn = model_selection.GridSearchCV(estimator=race_knn, 
                                                  param_grid=param_grid,
                                                  cv=5,
                                                  scoring="accuracy")
gridsearch_race_knn.fit(X_race_train, y_race_train)

print('\n\nK Nearest Neighbors CV Accuracy Score: {}'.format(abs(gridsearch_race_knn.best_score_)))
print('Best parameters: {}'.format(gridsearch_race_knn.best_params_))

test_prediction = gridsearch_race_knn.predict(X_race_test)
print('K Nearest Neighbors Test Accuracy Score: {}'.format(metrics.accuracy_score(test_prediction, y_race_test)))
print(metrics.classification_report(y_race_test, test_prediction))


# **ADA Boost Classifier**
# 
# ---
# 
# 

# In[ ]:


race_abc = ensemble.AdaBoostClassifier()
param_grid = {'n_estimators': [40, 50, 60]}

gridsearch_race_abc = model_selection.GridSearchCV(estimator=race_abc, 
                                                  param_grid=param_grid,
                                                  cv=5,
                                                  scoring="accuracy")
gridsearch_race_abc.fit(X_race_train, y_race_train)

print('\n\nADA Boost CV Accuracy Score: {}'.format(abs(gridsearch_race_abc.best_score_)))
print('Best parameters: {}'.format(gridsearch_race_abc.best_params_))

test_prediction = gridsearch_race_abc.predict(X_race_test)
print('ADA Boost Test Accuracy Score: {}'.format(metrics.accuracy_score(y_race_test, test_prediction)))
print(metrics.classification_report(y_race_test, test_prediction))


# **Results**
# 
# ---
# 
# 

# In[ ]:


from sklearn.inspection import permutation_importance

# KNN Boosting
result = permutation_importance(gridsearch_race_knn.best_estimator_, X_race_test,
                                y_race_test, n_repeats=10, random_state=0)
race_knn_feature_importance = pd.Series(result.importances_mean)
race_knn_feature_importance.index = X_race_train.columns
race_knn_feature_importance = race_knn_feature_importance.sort_values()

# ADA Boost Classifier
result = permutation_importance(gridsearch_race_abc.best_estimator_, X_race_test,
                                y_race_test, n_repeats=10, random_state=0)
race_abc_feature_importance =  pd.Series(result.importances_mean)
race_abc_feature_importance.index = X_race_train.columns
race_abc_feature_importance = race_abc_feature_importance.sort_values()

fig, ax = plt.subplots(1,2, figsize=(20,10))
fig.tight_layout(pad=10)
sns.barplot(ax=ax[0],y=race_abc_feature_importance.index, x=race_abc_feature_importance)
ax[0].set_title('ADA Boost Classifier', fontdict={'fontsize': 15})
sns.barplot(ax=ax[1],y=race_knn_feature_importance.index, x=race_knn_feature_importance)
ax[1].set_title('KNN Classifier', fontdict={'fontsize': 15})


# For both models, the importance of most features are too minor compared to the top ones. Moreover, these two models prioritize features differently for predicting the output. For ADA Boost, share of black, white, and black are the most important and Median Income and age has almost no effect. In contrast, KNN Classifier uses Median Income as almost the only feature to making prediction.
# These two models seems unreliable to use.  

# In[ ]:


# accuracy
knn_CV = abs(gridsearch_race_knn.best_score_)
knn_test = abs(gridsearch_race_knn.score(X_race_test, y_race_test))
abc_CV = abs(gridsearch_race_abc.best_score_)
abc_test = abs(gridsearch_race_abc.score(X_race_test, y_race_test))
age_models = pd.DataFrame({'Accuracy': [knn_CV, knn_test, abc_CV, abc_test],
                           'CV/Test':['CV','Test','CV','Test'],
                           'model': ['knn','knn','abc','abc']})
plt.title('Accuracy between KNN and ADABoost')
sns.barplot(x=age_models['model'], y=age_models['Accuracy'], hue=age_models['CV/Test'])


# The ADA Boost Classifier model does suprisingly better in the test set than in the training set. KNN and ADABoost are comparatively equal in making decisions about the victim's race. However, the accuracy for both models are too low, only around 0.5. With these accuracy scores and the different behaviors of the two models discussed above, we can conclude that using these models' prediction are not better than blind guess. As demonstrated below, with the proportion of White victims being 0.497, setting a model to always predict 'W' will also be able to achieve a similar result.

# In[ ]:


# Proportion of White victims
data['race'].value_counts()['W']/len(data['race'])


# # **Conclusion**

# In this report, I analysed the dataset regarding fatal police shooting in the US. Note that this data only includes incidents in which the officers used his or her gun to cause the fatality. This doesn't include other fatal methods.
# 
# These dataset suffers from many shortcomings: 
# 
# *   The demographic data lost its source of origin. Those datasets are for year 2015. Therefore, aggregating them with the fatal_police_shooting dataset, which is from 2015 to 2017, to analyze and build the model doesn't give the most accurate result.
# *   The data collected is not enough to predict the victim's race. Several factors like demographic of the location where the incidents happened, median income, or general status of the victims before getting shot were not indicative enough to make any conclusion about their race. Some other features that may help are information of the officers, criminal history of the victims, for what reason did the officers approached the victims.
# *   The datas are only from 2015 to 2017, therefore, we cannot make any inferences outside of that timeframe.
# 
# With many assumptions made during analyzing these data, some conclusions were able to be made:
# 
# *   White victims tend to be the oldest in the group and Native American victims tend to be the youngest.
# *   Many victims were having sign of mental illness, some of them attacked the officers during the incidents, and the most common weapon the victims were armed with was gun.
# *   Races other than White, Black, Hispanic, Asian, and Native American were a lot more likely to get shot when they commit a crime.
# *   Connecticut, Rhode Island, New Jersey officers were the most likely to make the fatal shots.
# 
# 
# 
