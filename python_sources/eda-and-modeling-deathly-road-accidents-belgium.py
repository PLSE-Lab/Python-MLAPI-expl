#!/usr/bin/env python
# coding: utf-8

# # Deathly road accidents Belgium (2005-2018)  

# ## Data source: https://statbel.fgov.be/en/open-data.<br>
# 
# -License: 'Licentie open data' which is compatible with the Creative Commons Attribution 2.0 license <br>https://creativecommons.org/licenses/by/2.0

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os, re, sys
import time


# In[ ]:


df_accidents = pd.read_csv('/kaggle/input/road-accidents-belgium/road_accidents_2005_2018.csv')


# In[ ]:


df_accidents.shape


# In[ ]:


df_accidents.columns


# #### Will focus on deadly accidents, remove other outcomes

# In[ ]:


df_accidents = df_accidents.drop(['Unnamed: 0','MS_ACCT_WITH_MORY_INJ',
       'MS_ACCT_WITH_SERLY_INJ', 'MS_ACCT_WITH_SLY_INJ','MS_ACCT'],axis=1)


# #### Remove french descriptions (FR columns)

# In[ ]:


cols = [c for c in df_accidents.columns if not c.endswith('FR')]
cols
df_accidents=df_accidents[cols]


# In[ ]:


df_accidents.columns


# In[ ]:


df_accidents.shape


# In[ ]:


df_accidents.head(5)


# In[ ]:


type(df_accidents['DT_DAY'].iloc[1])


# In[ ]:


df_accidents['date'] = pd.to_datetime(df_accidents['DT_DAY'])
df_accidents = df_accidents.drop('DT_DAY',axis=1)


# In[ ]:


type(df_accidents['date'].iloc[1])


# In[ ]:


df_accidents['day'] = df_accidents['date'].apply(lambda date: date.day)
df_accidents['month'] = df_accidents['date'].apply(lambda date: date.month)
df_accidents['year'] = df_accidents['date'].apply(lambda date: date.year)
df_accidents['quarter'] = df_accidents['date'].apply(lambda date: date.quarter)


# In[ ]:


df_accidents.index = pd.DatetimeIndex(df_accidents['date'])


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Distribution of accidents per day', fontsize=16)
plt.tick_params(labelsize=14)
sns.distplot(df_accidents.resample('D').size(), bins=60);


# In[ ]:


accidents_daily = pd.DataFrame(df_accidents.resample('D').size())
accidents_daily['MEAN'] = df_accidents.resample('D').size().mean()
accidents_daily['STD'] = df_accidents.resample('D').size().std()
UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']
LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']

plt.figure(figsize=(15,6))
df_accidents.resample('D').size().plot(label='Accidents per day')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Total accidents per day', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Number of accidents')
plt.tick_params(labelsize=14)
plt.legend(prop={'size':16})


# In[ ]:


plt.figure(figsize=(15,6))
df_accidents.resample('M').size().plot(label='Total per month')
df_accidents.resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')

plt.title('Accidents per month', fontsize=16)
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Distribution of accidents per month', fontsize=16)
plt.tick_params(labelsize=14)
sns.distplot(df_accidents.resample('M').size(), bins=60);


# In[ ]:


accidents_daily = pd.DataFrame(df_accidents.resample('M').size())
accidents_daily['MEAN'] = df_accidents.resample('M').size().mean()
accidents_daily['STD'] = df_accidents.resample('M').size().std()
UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']
LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']

plt.figure(figsize=(15,6))
df_accidents.resample('M').size().plot(label='Accidents per month')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Total accidents per month', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Number of accidents')
plt.tick_params(labelsize=14)
plt.legend(prop={'size':16})


# In[ ]:


accidents_daily = pd.DataFrame(df_accidents.resample('Y').size())
accidents_daily['MEAN'] = df_accidents.resample('Y').size().mean()
accidents_daily['STD'] = df_accidents.resample('Y').size().std()
UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']
LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']

plt.figure(figsize=(15,6))
df_accidents.resample('Y').size().plot(label='Accidents per year')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Total accidents per year', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Number of accidents')
plt.tick_params(labelsize=14)
plt.legend(prop={'size':16})


# In[ ]:


df_accidents.columns


# In[ ]:


### Remove duplicate text attributes


# In[ ]:


df_accidents = df_accidents.drop(['TX_DAY_OF_WEEK_DESCR_NL','TX_BUILD_UP_AREA_DESCR_NL',
                                 'TX_COLL_TYPE_DESCR_NL','TX_LIGHT_COND_DESCR_NL',
                                 'TX_ROAD_TYPE_DESCR_NL','TX_MUNTY_DESCR_NL',
                                 'TX_ADM_DSTR_DESCR_NL','TX_PROV_DESCR_NL',
                                 'TX_RGN_DESCR_NL'],axis=1)


# In[ ]:


df_accidents.columns


# In[ ]:


df_accidents.isnull().sum()


# In[ ]:


(df_accidents.isnull().sum())/len(df_accidents)


# In[ ]:


attributes_missing = [att for att in df_accidents.columns if df_accidents[att].isnull().sum()>0 ]


# In[ ]:


attributes_missing


# In[ ]:


for att in attributes_missing:
    
    print(df_accidents[att].value_counts())


# In[ ]:


attributes_missing


# In[ ]:


#replace nan's 

for att in attributes_missing:
    
    if att == 'CD_BUILD_UP_AREA':
        df_accidents[att].fillna(1.0, inplace = True) 
    elif att == 'CD_COLL_TYPE':
        df_accidents[att].fillna(4.0, inplace = True) 
    elif att == 'CD_LIGHT_COND':
        df_accidents[att].fillna(1.0, inplace = True) 
    elif att == 'CD_ROAD_TYPE':
        df_accidents[att].fillna(2.0, inplace = True) 
    else:
        df_accidents[att].fillna(10000.0, inplace = True)


# In[ ]:


df_accidents.isnull().sum()


# CD_BUILD_UP_AREA (where)
# 
# * 1.0 = Within built-up areas (in residential area)
# * 2.0 = Outside built-up areas
# * 3.0 = Not available 
# 
# CD_COLL_TYPE
# 
# * 1.0 = Chain collision (4 drivers or more)
# * 2.0 = Frontal impact (or when crossing)
# * 3.0 = Along the back (or side by side)
# * 4.0 = Sideways
# * 5.0 = With a pedestrian
# * 6.0 = Against an obstacle on the roadway
# * 7.0 = Against an obstacle outside the roadway
# 
# CD_LIGHT_COND
# 
# * 1.0 # In broad daylight
# * 3.0 # Night, public lighting turned on
# * 5.0 # Not available
# * 2.0 # Dawn - twilight
# * 4.0 # Night, no public lighting
# 
# CD_ROAD_TYPE
# 
# * 2.0 Regional road, provincial road or municipal road
# * 1.0 Motorway
# * 3.0 Unknown
# 
# TX_RGN_DESCR_NL
# 
# * Vlaams Gewest 2000                    
# *  Waals Gewest 3000                     
# * Brussels Hoofdstedelijk Gewest 4000     
# 
# 

# # EDA

# 

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df_accidents.corr().round(2),annot=True)


# In[ ]:


df_accidents.columns


# In[ ]:


# Create a pivot table by crossing the day number by the month and calculate the average number of accidents for each crossing
accidents_pivot_table = df_accidents.pivot_table(values='date', index='day', columns='month', aggfunc=len)
accidents_pivot_table_date_count = df_accidents.pivot_table(values='date', index='day', columns='month', aggfunc=lambda x: len(x.unique()))
accidents_average = accidents_pivot_table/accidents_pivot_table_date_count
accidents_average.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Using seaborn heatmap
plt.figure(figsize=(7,9))
plt.title('Average accidents per day and month', fontsize=14)
sns.heatmap(accidents_average.round(), cmap='coolwarm', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")


# In[ ]:


df_accidents = df_accidents.drop('date',axis=1)


# In[ ]:


for att in df_accidents.columns:
    plt.figure() #this creates a new figure on which your plot will appear
    sns.countplot(df_accidents[att])


# In[ ]:


df_accidents.shape


# In[ ]:


df_accidents = df_accidents[df_accidents['MS_ACCT_WITH_DEAD_30_DAYS']!=2]


# In[ ]:


df_accidents = df_accidents[df_accidents['MS_ACCT_WITH_DEAD']!=2]


# In[ ]:


df_accidents.shape


# #### Explore REFNIS
# 
# https://en.wikipedia.org/wiki/Provinces_of_Belgium
# The country of Belgium is divided into three regions. Two of these regions, the Flemish Region or Flanders, and Walloon Region, or Wallonia, are each subdivided into five provinces. The third region, the Brussels-Capital Region, is not divided into provinces, as it was originally only a small part of a province itself. 
# 
# https://nl.wikipedia.org/wiki/NIS-code
# The NIS code (Dutch: NIS-code, French: code INS) is an alphanumeric code for regional areas of Belgium.
# This code is used for statistical data treatment in Belgium. This code was developed mid-1960s by the Statistics Belgium. It was first used for the census of 1970. 
# 
# 
# https://statbel.fgov.be/nl/open-data/refnis-code
# Belgium is classified according to a hierarchical system that the French introduced in 1796. The administrative division is based on four territorial units: regions, provinces, administrative districts and municipalities. The municipalities are the basic units, which means that each higher level territorial unit consists of different municipalities.
# 
#    Level 1: 3 regions
#     Level 2: 10 provinces
#     Level 3: 43 administrative districts
#     Level 4: 581 municipalities (589 to 31/12/2018) --- sanity check: length of CD_MUNTY_REFNIS also 589
#     Level 5: Boroughs
#     Level 7: Statistical sectors
# 
# https://nl.wikipedia.org/wiki/Arrondissement
# A district is part of the territory of the state, which is divided up for administrative or administrative reasons. Each district is an area of responsibility of colleges and civil servants (working for a government). It is originally from French and literally means completion.
# 
# https://nl.wikipedia.org/wiki/Deelgemeente_(Belgi%C3%AB)
# A sub-municipality in Belgium is the territory of the former municipalities that were still independent before the major municipal redivisions in the 1960s-70s
# 
# Source REFNIS file: https://statbel.fgov.be/nl/over-statbel/methodologie/classificaties/geografie

# In[ ]:


df_accidents.columns


# In[ ]:


df_accidents = df_accidents.rename(index=str, columns={'CD_BUILD_UP_AREA':'where',
                                                      'CD_COLL_TYPE':'how',
                                                      'CD_ROAD_TYPE':'typeofroad',
                                                      'CD_MUNTY_REFNIS':'refnismun',
                                                      'CD_DSTR_REFNIS':'refnisdist',
                                                      'CD_PROV_REFNIS':'refnisprov',
                                                      'CD_RGN_REFNIS':'refnisgew',
                                                      'CD_LIGHT_COND':'illumination',
                                                      'MS_ACCT_WITH_DEAD':'dead',
                                                      'MS_ACCT_WITH_DEAD_30_DAYS':'deadafter30d'})


# In[ ]:


df_accidents.head()


# Cardinality refers to the number of unique values contained in a particular column, or field, of a database. 

# In[ ]:


print(df_accidents['refnismun'].nunique())
print(df_accidents['refnisdist'].nunique())
print(df_accidents['refnisprov'].nunique())
print(df_accidents['refnisgew'].nunique())


# Let's focus on municipalities and remove district, province and region (gewest).

# In[ ]:


df_accidents = df_accidents.drop(['refnisdist','refnisprov','refnisgew'],axis=1)


# In[ ]:


df_accidents.head()


# In[ ]:


df_accidents = df_accidents.reset_index()


# In[ ]:


df_accidents = df_accidents.drop('date',axis=1)


# In[ ]:


df_accidents.head()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df_accidents.corr().round(2),annot=True)


# * deadafter30d and dead are highly correlated. Remove deadafter30d.
# * Quarter and month is also duplicate information. Remove quarter.

# In[ ]:


df_accidents = df_accidents.drop(['deadafter30d','quarter'],axis=1)


# In[ ]:


df_accidents.head()


# In[ ]:


df_accidents.columns


# In[ ]:


df_accidents.columns = ['hour', 'day_of_week', 'where', 'how', 'illumination',
       'roadtype', 'refnismun', 'death', 'day_of_month', 'month', 'year']


# In[ ]:


df_accidents.head()


# In[ ]:


df_accidents['year'] = df_accidents['year']-2000


# In[ ]:


df_accidents.head()


# In[ ]:


df_accidents = df_accidents.astype(int)


# In[ ]:


df_accidents.head()


# In[ ]:


df_accidents['refnismun'].unique()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df_accidents.corr().round(2),annot=True)


# Find most informative features

# In[ ]:


# from sklearn.feature_selection import RFE #recursive feature elimination (RFE)
# from sklearn.svm import SVR # Support Vector Regression
# X = df_accidents.drop('death',axis=1)
# X = X[:7000]
# y = df_accidents['death']
# y = y[:7000]
# print(X.shape, y.shape)


# In[ ]:


# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=5, step=1)


# In[ ]:


# selector = selector.fit(X, y)
# selector.support_


# In[ ]:


#  selector.ranking_


# * ['hour', 'day_of_week', 'where', 'how', 'illumination', 'roadtype','refnismun', 'day_of_month', 'month', 'year']
# 
# 
# * 1000: array([4, 1, 1, 1, 1, 3, 1, 5, 2, 6])
# * 5000: array([5, 1, 1, 3, 1, 1, 4, 1, 2, 6])
# * 7000: array([3, 4, 1, 1, 1, 1, 5, 2, 1, 6])
# * 10000: CPU can't handle
# * 40000:45000 : CPU can't handle?! They're also 5000 instances..
# 
# 
# * Reocurring 1's in all 3: 'where','illumination'. But still I will use all of them to model. Can't rely on these relatively few instances used for RFE.
# * 'year' can be removed, also makes sense that the year can't influence individual instances so much.

# In[ ]:


df_accidents = df_accidents.drop('year',axis=1)


# In[ ]:


df_accidents.head()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df_accidents.corr().round(2),annot=True)


# In[ ]:


df = df_accidents


# # Model 

# In[ ]:


sns.countplot(df['death'])


# In[ ]:


df['death'].value_counts()/len(df)


# With only 1.5% of the dataset being deathly accidents, it will be difficult (impossible) to build an good model.

# ## Training on oversampled datasets

# In[ ]:


X = df.drop('death', axis=1).values
y = df['death'].values


# In[ ]:


from imblearn.over_sampling import RandomOverSampler

# define oversampling strategy

oversample = RandomOverSampler(sampling_strategy='minority')

# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)

from collections import Counter


# In[ ]:


# summarize class distribution

print(Counter(y))

# summarize class distribution

print(Counter(y_over))


# In[ ]:


fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.countplot(df['death'], ax=ax1).set_title('Before')
sns.countplot((y_over), ax=ax2).set_title('After')


# In[ ]:


X = X_over
y = y_over

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# #### Fit

# In[ ]:


logmodel_over = LogisticRegression()
logmodel_over.fit(X_train, y_train)


# In[ ]:


rfc_over = RandomForestClassifier(n_estimators=100, verbose=100)
rfc_over.fit(X_train,y_train)


# #### Predictions

# In[ ]:


#Predictions and Evaluations

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# ### Logistic regression

# In[ ]:


#Model performance on train dataset

training_predictions = logmodel_over.predict(X_train)

print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))

plot_confusion_matrix(logmodel_over, X_train, y_train)


# In[ ]:


plot_confusion_matrix(logmodel_over, X_train, y_train,normalize='true')


# In[ ]:


#Model performance on test dataset

test_predictions = logmodel_over.predict(X_test)

print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))

plot_confusion_matrix(logmodel_over, X_test, y_test)


# In[ ]:


plot_confusion_matrix(logmodel_over, X_test, y_test,normalize='true')


# ### RF

# In[ ]:


#Model performance on train dataset

training_predictions = rfc_over.predict(X_train)

print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))

plot_confusion_matrix(rfc_over, X_train, y_train)


# In[ ]:


#Model performance on test dataset

test_predictions = rfc_over.predict(X_test)

print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))

plot_confusion_matrix(rfc_over, X_test, y_test)


# # Test oversampled models on entire dataset

# In[ ]:


X_test = df_accidents.drop('death', axis=1).values
y_test = df_accidents['death'].values


# In[ ]:


predictions_log_reg = logmodel_over.predict(X_test)
predictions_rf = rfc_over.predict(X_test)


# Logistic reg
# 

# In[ ]:


print(classification_report(y_test, predictions_log_reg))

print(confusion_matrix(y_test, predictions_log_reg))

plot_confusion_matrix(logmodel_over, X_test, y_test)


# RF

# In[ ]:


print(classification_report(y_test, predictions_rf))

print(confusion_matrix(y_test, predictions_rf))

plot_confusion_matrix(rfc_over, X_test, y_test)


# ## Training on undersampled datasets

# In[ ]:


X = df.drop('death', axis=1).values
y = df['death'].values


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler

# define undersampling strategy

oversample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)

from collections import Counter


# In[ ]:


# summarize class distribution

print(Counter(y))

# summarize class distribution

print(Counter(y_over))


# In[ ]:


fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.countplot(df['death'], ax=ax1).set_title('Before')
sns.countplot((y_over), ax=ax2).set_title('After')


# In[ ]:


X = X_over
y = y_over

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ### Fit

# In[ ]:


logmodel_under = LogisticRegression()
logmodel_under.fit(X_train, y_train)

rfc_under = RandomForestClassifier(n_estimators=100,verbose=100)
rfc_under.fit(X_train,y_train)


# ### Predictions

# Logistic regression

# In[ ]:


#Model performance on train dataset

training_predictions = logmodel_under.predict(X_train)

print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))

plot_confusion_matrix(logmodel_under, X_train, y_train)


# In[ ]:


#Model performance on test dataset

test_predictions = logmodel_under.predict(X_test)

print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))

plot_confusion_matrix(logmodel_under, X_test, y_test)


# RF

# In[ ]:


#Model performance on train dataset

training_predictions = rfc_under.predict(X_train)

print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))

plot_confusion_matrix(rfc_under, X_train, y_train)


# In[ ]:


#Model performance on test dataset

test_predictions = rfc_under.predict(X_test)

print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))

plot_confusion_matrix(rfc_under, X_test, y_test)


# # Test undersampled models on entire dataset

# In[ ]:


X_test = df_accidents.drop('death', axis=1).values
y_test = df_accidents['death'].values


# In[ ]:


predictions_log_reg = logmodel_under.predict(X_test)
predictions_rf = rfc_under.predict(X_test)


# Logistic reg

# In[ ]:


print(classification_report(y_test, predictions_log_reg))


# In[ ]:


print(confusion_matrix(y_test, predictions_log_reg))


# In[ ]:


plot_confusion_matrix(logmodel_under, X_test, y_test)


# RF

# In[ ]:


print(classification_report(y_test, predictions_rf))


# In[ ]:


print(confusion_matrix(y_test, predictions_rf))


# In[ ]:



plot_confusion_matrix(rfc_under, X_test, y_test)

