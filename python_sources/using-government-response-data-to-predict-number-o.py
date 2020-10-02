#!/usr/bin/env python
# coding: utf-8

# # Using government response data to predict number of cases
# 
# Governments across the world are applying social distancing measures in the hope of reducing the dissemination of COVID-19. Those measures, however, come with a huge economic burden, which, if too large, may actually bring even more severe consequences than the virus. This is a new situation, and we are not sure on what should be done.
# 
# There is, therefore, an interest on defining quantitatively the effectiveness of measures such as school closings, public transport restictions, etc. This could help policy-makers define when to start such meaures and when it's safe to stop them. 
# A recent study made by an Oxford team has released a dataset containing the different measures that were carried out by different countries around the world. More details here:
# 
# https://www.bsg.ox.ac.uk/research/research-projects/oxford-covid-19-government-response-tracker
# 
# This notebook is divided in 4 sections:
# 
# 1) Generating a dataset: the data that was made available by the Oxford group is not ready for applying models. I have thus organized it and shared on the link below in case anyone wishes to make further tests. 
# 
# https://www.kaggle.com/rhfall/government-response-to-covid19-worldwide
# 
# 2) Applying ML models and other statistical approaches to define importance of each of the measures
# 
# 3) Building a SEIR model (a mathematical model used to describe the spread of diseases) and use the government response data to define the parameters of the model. 
# 
# 4) Discussion topics
# 
# 
# This is still a work in its early beginnings, so there still a lot of flaws and things missing, but I will try to have this all ready as soon as I can. However, I thought it would be a good idea to already share what I have until now. Please, I would love to hear your opinions.

# ## 1) Generating Dataset

# In[ ]:


import pandas as pd
import datetime
import math
import numpy as np


# In[ ]:


oxford = pd.read_csv("../input/government-response-to-covid19-worldwide/OxCGRT_Download_29_03.csv", sep=';')


# In[ ]:


oxford.head()


# ### Dropping columns
# 
# This dataset is is composed by 11 indexes, such as Schoolclosing, Workingclosing etc. Each of those indexes has three columns. The first is the actual value, whose meaning depends on the index but normally means that 0 is a measure not taken, 1 is a measure recommended and 2 is a measure enforced. The second column is whether this measure was applied to whole national territory and the third is general notes. 
# 
# Indexes S1 to S7 are social distancing measures whereas indexes S8 to S11 are economic ones. We will therefore only use the S1 to S7. We will also drop the IsGeneral columns as they are mostly empty and also the Notes as they have no interest for a quantitative analysis. 
# 
# For more information about thoses indexes refer to the original Oxford article.

# In[ ]:


oxford['Date'] = pd.to_datetime(oxford['Date'], format='%Y%m%d')


# In[ ]:


oxford.drop(['CountryCode'], axis=1,inplace=True)


# In[ ]:


new_col = [x for x in oxford.columns if ('_Notes' not in x) and ('IsGeneral' not in x)] 


# In[ ]:


new_oxford = oxford[new_col]


# In[ ]:


new_oxford.drop(['S8_Fiscal measures', 'S9_Monetary measures', 'S10_Emergency investment in health care',
                           'S11_Investment in Vaccines', 'Unnamed: 34'], axis=1,inplace=True)


# ### Fixing NaNs
# 
# Ok, so we have already droped the columns we are not interested in. Let's analyse how is the data for a single country, such as Argentina and Norway, below. We note a couple of things:
# 
# 1) For Argentina, before the first Case and first Death, the values are NaN. This happens for several countries and we can fill it with 0. 
# 
# 2) For Norway, for the rows in the beginning of the year, the actions are all NaN and we can assume that those actions were not taken, changing it to zero.
# 
# 3) For both countries, the most recent information about the measures are NaN, but we can assume that they remained the same and we simply use the last not NaN value.
# 
# ATTENTION: This assumption may be valid for this two countries, but there are countries, such as Brazil, where we don't have the information of the last 10 days, and where actually  virtually all measureas have been taken in the last 10 days, so assuming a situation as it was in 16/03 (the last day with data) is completely unrealistic. This data should thus be dealt with care. More on that later
# 
# 4) For Argentina,there are some recent NaN values. This should soon be fixed and, as those are quite rare, we will simply drop them

# In[ ]:


new_oxford[new_oxford['CountryName']=='Argentina']


# In[ ]:


new_oxford[new_oxford['CountryName']=='Norway']


# In[ ]:


countries = set(new_oxford['CountryName'].values)


# In[ ]:


oxford_fixed = pd.DataFrame(columns = new_oxford.columns)


# In[ ]:


for country in countries:
    df = new_oxford[new_oxford['CountryName'] == country].copy()
    #Cases
    for j in range(9,11):
        for i in range(df.shape[0]):
            if math.isnan(df.iloc[i, j]):
                df.iloc[i, j] = 0
            elif df.iloc[i, j] > 0:
                break
    #Measures
    for j in range(2,9):
        started = False
        for i in range(df.shape[0]):
            if started == False:
                if math.isnan(df.iloc[i, j]):
                    df.iloc[i, j] = 0
                elif df.iloc[i, j] > 0:
                    started = True
                    value = df.iloc[i, j]
            else:
                if math.isnan(df.iloc[i, j]):
                    df.iloc[i, j] = value
                else:
                    value = df.iloc[i, j]
    #Index
    j=11                
    started = False
    for i in range(df.shape[0]):
        if started == False:
            if math.isnan(df.iloc[i, j]):
                df.iloc[i, j] = 0
            elif df.iloc[i, j] > 0:
                started = True
                value = df.iloc[i, j]
        else:
            if math.isnan(df.iloc[i, j]):
                df.iloc[i, j] = value
            else:
                value = df.iloc[i, j]
    oxford_fixed = oxford_fixed.append(df)

oxford_fixed = pd.get_dummies(oxford_fixed, columns = new_oxford.columns[2:9],drop_first=True)


# We will also drop the StrigencyIndex column, which is simply an index the researchers have used to consolidate all measures

# In[ ]:


strigency = oxford_fixed[['StringencyIndex']] #saving for eventual use


# In[ ]:


oxford_fixed=oxford_fixed.drop(['StringencyIndex'],axis=1)


# In[ ]:


oxford_fixed.head()


# ### Creating new features
# 
# 1) The effect of the measures is not imediate, you can only see it in the future and it changes with time. So, instead of simply using the features as booleans, I will create columns saying how many days have passed since the measures were taken. Sometimes countries stop some measures and this is not taken into account here, which may be a flaw of this approach
# 
# 2) I am doing the same with the number of cases and deaths, creating a column showing the number of days since the first case/death. Hopefully, this would give us a way of normalizing the data of the countries across time, as the virus has reached the countries in different points in time. However, I have analyzed that the first cases seem quite random, and a real pattern only starts to emerge after 100 cases, so I have also added a DaysSince100Cases
# 
# 3) The problem of predicting the number of cases is that is is highly dependent on the number of cases of the previous day, making it somehow complicated to use data from other countries to predict the number of cases of another country on a given day. The rate of increase, however, is more easily generalizable across countries. The problem is that it shows a too wild variation, specially on the beginning of the crisis. I apply thus a smoothing factor, doing an average of this value on the last 4 days. 

# In[ ]:


list_days = ['Days_since_' + x for x in list(oxford_fixed.columns[2:])]


# In[ ]:


list_days


# In[ ]:


oxford_days = pd.DataFrame(columns=list_days, index=oxford_fixed.index).fillna(0)


# In[ ]:


list_perc = ['%ConfirmedCases', '%ConfirmedCases_smooth', '%ConfirmedDeaths', '%ConfirmedDeaths_smooth', 'Days', 'Days_since_100Cases']
oxford_perc = pd.DataFrame(columns=list_perc, index=oxford_fixed.index).fillna(0)


# In[ ]:


oxford_complete = pd.concat([oxford_fixed,oxford_days, oxford_perc],axis=1)


# In[ ]:


oxford_complete.head()


# In[ ]:


oxford_complete_days = pd.DataFrame(columns = oxford_complete.columns)
smooth = 4
for country in countries:
    df = oxford_complete[oxford_complete['CountryName'] == country].copy()
    #Measures
    for j in range(2,18):
        value = 0
        for i in range(df.shape[0]):
            df.iloc[i, j+16] = value
            if df.iloc[i, j] > 0:
                value = value + 1
    valueconf = 0
    valuedeath = 0
    days = 0
    days_100 = 0
    for i in range(df.shape[0]):
        
        if df.iloc[i, 2] > 0:
            valueconf = valueconf + 1
        if df.iloc[i, 3] > 0:
            valuedeath = valuedeath + 1
            
        if valueconf > 1:
            df.iloc[i,34] = (df.iloc[i,2] - df.iloc[i-1,2])/df.iloc[i-1,2]
        if valuedeath > 1:
            df.iloc[i,36] = (df.iloc[i,3] - df.iloc[i-1,3])/df.iloc[i-1,3]
            
        if valueconf >= smooth:
            df.iloc[i,35] = df.iloc[i-smooth+1:i+1,34].sum()/smooth
        if valuedeath >= smooth:
            df.iloc[i,37] = df.iloc[i-smooth+1:i+1,36].sum()/smooth 
        
        df.iloc[i,38] = days
        days = days + 1
        
        
        if df.iloc[i, 2] > 100:
            days_100 += days_100
        df.iloc[i,39] = days_100
        
        
    oxford_complete_days = oxford_complete_days.append(df)


# Here I wll drop the recent NaN values of the number of cases and deaths

# In[ ]:


oxford_complete_days.dropna(inplace=True) 


# In[ ]:


oxford_complete_days.head()


# In[ ]:


oxford_fixed.columns


# This dataset ranges across a long period of time, since 1st January, whereas the virus has reached most countries only on March. We have, therefore, many rows where nothing has happened, neither government measures nor cases or deaths, being thus useless for our models. We will drop those all-zeros rows

# In[ ]:


rows_not_zero = [not x for x in oxford_complete_days[oxford_fixed.columns[2:]].sum(axis=1)==0]


# In[ ]:


oxford_complete_final = oxford_complete_days[rows_not_zero]


# In[ ]:


oxford_complete_final.head()


# Dropping the boolean variables and leaving only the DaysSince ones

# In[ ]:


oxford_complete_final = oxford_complete_final.drop(oxford_complete_final.columns[4:18],axis=1)


# In[ ]:


oxford_complete_final.head()


# So, this is our final dataset, containing all countries

# ### Dropping countries without enough government measures information
# 
# As discussed previously, some assumptions were made to fill the NaN values, which may be unrealistic for countries which had too few information. In this section, we will create a reduced dataset, using only the countries which have enough information. In order to do that, we will use the StrigencyIndex. This is an index created by the Oxford researchers to consolidate the value of all other indexes. When all the other indexes are NaN, than the StrigencyIndex also is, so we will use it to see how many rows each country have of measures information

# In[ ]:


strin = [not math.isnan(x) for x in new_oxford['StringencyIndex'].values]


# In[ ]:


new_oxford_drop_strin = new_oxford[strin]


# In[ ]:


new_oxford_drop_strin['CountryName'].value_counts().iloc[-40:]


# In[ ]:


new_oxford[new_oxford['CountryName'] == 'Iran'].iloc[-40:,:]


# In[ ]:


new_oxford[new_oxford['CountryName'] == 'Israel'].iloc[-40:,:]


# We see that Iran doesn't have any data on March and Israel doesn't have data from 16/03 on, as it's the case of many other countries. So we choose the boundary of using only countries which have more that 60 rows which are not NaN. 
# 
# We must, however rememeber that many of those countries that are left have unrealistic values for the most recent data. This may not be so worrying due to the fact that those measure don't have an immediate result but we still must take care with this. 
# so what I do in the training section is to only train my model using data until 16/03, using the rest of the data for test and focusing on testing with countries that do have recent data. 

# In[ ]:


list_countries = new_oxford_drop_strin['CountryName'].value_counts()>60


# In[ ]:


list_restricted = [x in list_countries.index[list_countries] for x in oxford_complete_final['CountryName'].values]


# In[ ]:


#list_test = ['Italy','United States','Spain','France','Germany']
#list_restricted = [x in list_test for x in oxford_complete_final['CountryName'].values]


# In[ ]:


oxford_restricted_final = oxford_complete_final[list_restricted]


# In[ ]:


oxford_restricted_final.shape


# In[ ]:


oxford_restricted_final.head()


# In[ ]:


#oxford_restricted_final.to_csv(r'~\COVID_gov_restricted_28_03.csv')


# In[ ]:


#oxford_complete_final.to_csv(r'~\COVID_gov_complete_28_03.csv')


# # Applying ML models

# ## Preparing Data

# In[ ]:


date = datetime.datetime(2020,3,16,0,0) #Limit data that we will use to train the model


# In[ ]:


cols_y = ['ConfirmedCases', 'ConfirmedDeaths', '%ConfirmedCases', '%ConfirmedCases_smooth', '%ConfirmedDeaths', '%ConfirmedDeaths_smooth']


# In[ ]:


X = oxford_restricted_final.drop(cols_y, axis=1)
#X = oxford_restricted_final[['CountryName','Date','Days_since_ConfirmedCases', 'Days_since_ConfirmedDeaths','Days', 'Days_since_100Cases']]
y = oxford_restricted_final[cols_y+['Date']]
X_countries = X #Used later for ploting


# In[ ]:


X = pd.get_dummies(X, columns = ['CountryName'],drop_first=True)


# In[ ]:


X_train = X[X['Date']<date]
X_test = X[X['Date']>=date]

X_train_countries = X_countries[X_countries['Date']<date]
X_test_countries = X_countries[X_countries['Date']>=date]

X_train.drop(['Date'], axis=1,inplace=True)
X_test.drop(['Date'], axis=1,inplace=True)


# We will create 4 different models, using those different variables as our y:

# In[ ]:


y_train_death = y[y['Date']<date][['ConfirmedDeaths']]
y_test_death = y[y['Date']>=date][['ConfirmedDeaths']]

y_train_cases = y[y['Date']<date][['ConfirmedCases']]
y_test_cases = y[y['Date']>=date][['ConfirmedCases']]

y_train_death_rate = y[y['Date']<date][['%ConfirmedDeaths_smooth']]
y_test_death_rate = y[y['Date']>=date][['%ConfirmedDeaths_smooth']]

y_train_cases_rate = y[y['Date']<date][['%ConfirmedCases_smooth']]
y_test_cases_rate = y[y['Date']>=date][['%ConfirmedCases_smooth']]


# ## Predicting

# In[ ]:


#Imports de modelagem
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, matthews_corrcoef
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, RandomizedSearchCV
from math import sqrt

#Algorithms
from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[ ]:


def metrics_report(X_test, y_test, reg):
    y_pred = reg.predict(X_test)
    return {'r2_score': r2_score(y_test, y_pred), 
          'mean_absolute_error': mean_absolute_error(y_test, y_pred),
          'mean_squared_error': mean_squared_error(y_test, y_pred),
         'median_absolute_error': median_absolute_error(y_test, y_pred),
           'RMSE': sqrt(mean_absolute_error(y_test, y_pred))}


# ### Random Forest

# Let's start by a Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# #### Cases

# In[ ]:


rfr_cases = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)


# In[ ]:


rfr_cases.fit(X_train,y_train_cases)


# In[ ]:


metrics_report(X_test, y_test_cases, rfr_cases)


# In[ ]:


metrics_report(X_train, y_train_cases, rfr_cases)


# Ok, it seems that we have greatly overfit the data, let's plot the predictions to see whats happening

# In[ ]:


pred = rfr_cases.predict(X_test)

pred_train = rfr_cases.predict(X_train)

y_pred_cases = pd.DataFrame(data = pred, index=y_test_cases.index, columns = y_test_cases.columns)

y_pred_cases_train = pd.DataFrame(data = pred_train, index=y_train_cases.index, columns = y_train_cases.columns)


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[ ]:


#countries = set(oxford_restricted_final['CountryName'])
countries = ['Italy','United States','Spain','France','Germany'] #Let's just plot some to make the visualization easier


# Let's first see how we do on the test set

# In[ ]:


fig = go.Figure()
for country in countries:
    df = X_test_countries[X_test_countries['CountryName']==country]
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_test_cases.loc[df.index,'ConfirmedCases'].values,
            mode='lines+markers',
            name=country))
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases.loc[df.index,'ConfirmedCases'].values,
            mode='markers',
            marker_symbol='x',                     
            name=country + ' predicted'))
fig.update_layout(
    title="Predicted vs Real cases for test set",
    xaxis_title="Number of days since first confirmed case",
    yaxis_title="Number of cases",
)
fig.show()


# And now on the training set

# In[ ]:


fig = go.Figure()
for country in countries:
    df = X_train_countries[X_train_countries['CountryName']==country]
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_train_cases.loc[df.index,'ConfirmedCases'].values,
            mode='lines+markers',
            name=country))
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases_train.loc[df.index,'ConfirmedCases'].values,
            mode='markers',
            marker_symbol='x',                     
            name=country + ' predicted'))
fig.update_layout(
    title="Predicted vs Real cases for training set",
    xaxis_title="Number of days since first confirmed case",
    yaxis_title="Number of cases",
)
fig.show()


# Yes, definetely there is overfit. I have tried changing the model parameters and it didn't work either. 
# Apparently for the test set, it is simply using the last value it had on the training set and increasing it slightly (the only exception is France).
# 
# Can we actually see how it is doing those predictions? There is a great lib for that called treeinterpreter, let's see if we can get some insights from that

# In[ ]:


get_ipython().system('pip install treeinterpreter')


# In[ ]:


from treeinterpreter import treeinterpreter as ti


# In[ ]:


prediction, bias, contributions = ti.predict(rfr_cases, X_test)


# Let's investigate how the predictions for Italy are being made

# In[ ]:


X_test_countries_italy = X_test_countries.reset_index()


# In[ ]:


X_test_countries_italy[X_test_countries_italy['CountryName']=='Italy']


# In[ ]:


italy_indexes = list(X_test_countries_italy[X_test_countries_italy['CountryName']=='Italy'].index)


# In[ ]:


for i in italy_indexes:
    print("Instance", i)
    print("Prediction: ", prediction[i])
    print("Bias (trainset mean)", bias[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 X_test.columns), 
                             key=lambda x: -abs(x[0]))[0:10]:
        print(feature, round(c, 2))
    print("-"*20)


# Apparently, the measures have a much bigger effect on the prediction than the actual period in time being analyzed. Therefore, since the measures have not changed, it makes sense that it is leaving the prediction constant. However, interestingly, on the two last predictions DaysSinceFirstDeath gains a sudden importance, and the prediction start to inscrease. Why at this point and why days since death and not since first case? 

# Let's now see some of the last predictions done for Italy on the training set:

# In[ ]:


X_train_Italy = X_train[X_train['CountryName_Italy']==1]


# In[ ]:


prediction_train, bias_train, contributions_train = ti.predict(rfr_cases, X_train_Italy)


# In[ ]:


for i in range(50,53):
    print("Instance", i)
    print("Prediction: ", prediction_train[i])
    print("Bias (trainset mean)", bias_train[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions_train[i], 
                                 X_train.columns), 
                             key=lambda x: -abs(x[0]))[:10]:
        print(feature, round(c, 2))
    print("-"*20)


# Apparently the usage of the features is similar to the one used on the training set. However, althought the measures remained constant, the model has somehow found internal rules to make it follow the training curve anyway, even without using the variable of the country

# I am not sure if this is worth anything, having in mind the poor results we had on the test set, but below you can find the features importance. Maybe making some changes on the model we can get something more accurate:

# In[ ]:


#Feature importances
feature_importances = pd.DataFrame(rfr_cases.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_importances.iloc[:10]


# ### Cases rate smooth

# Well, let's see if we get anything better using the rate of change

# In[ ]:


rfr_cases_rate = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)


# In[ ]:


rfr_cases_rate.fit(X_train,y_train_cases_rate)


# In[ ]:


metrics_report(X_test, y_test_cases_rate, rfr_cases_rate)


# In[ ]:


metrics_report(X_train, y_train_cases_rate, rfr_cases_rate)


# In[ ]:


pred_rate = rfr_cases_rate.predict(X_test)

pred_train_rate = rfr_cases_rate.predict(X_train)

y_pred_cases_rate = pd.DataFrame(data = pred_rate, index=y_test_cases_rate.index, columns = y_test_cases_rate.columns)

y_pred_cases_train_rate = pd.DataFrame(data = pred_train_rate, index=y_train_cases_rate.index, columns = y_train_cases_rate.columns)


# In[ ]:


fig = go.Figure()
for country in countries:
    df = X_test_countries[X_test_countries['CountryName']==country]
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_test_cases_rate.loc[df.index,'%ConfirmedCases_smooth'].values,
            mode='lines+markers',
            name=country))
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases_rate.loc[df.index,'%ConfirmedCases_smooth'].values,
            mode='markers',
            marker_symbol='x',                     
            name=country + ' predicted'))
fig.update_layout(
    title="Predicted vs Real cases rate for test set",
    xaxis_title="Number of days since first confirmed case",
    yaxis_title="Rate of change of the number of cases",
)
fig.show()


# In[ ]:


fig = go.Figure()
for country in countries:
    df = X_train_countries[X_train_countries['CountryName']==country]
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_train_cases_rate.loc[df.index,'%ConfirmedCases_smooth'].values,
            mode='lines+markers',
            name=country))
    fig.add_trace(go.Scatter(x=df['Days_since_ConfirmedCases'].values, y=y_pred_cases_train_rate.loc[df.index,'%ConfirmedCases_smooth'].values,
            mode='markers',
            marker_symbol='x',                     
            name=country + ' predicted'))
fig.update_layout(
    title="Predicted vs Real cases rate for train set",
    xaxis_title="Number of days since first confirmed case",
    yaxis_title="Rate of change of the number of cases",
)
fig.show()


# In[ ]:


prediction_rate, bias_rate, contributions_rate = ti.predict(rfr_cases_rate, X_test)


# In[ ]:


for i in italy_indexes:
    print("Instance", i)
    print("Prediction: ", prediction_rate[i])
    print("Bias (trainset mean)", bias_rate[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions_rate[i], 
                                 X_test.columns), 
                             key=lambda x: -abs(x[0]))[0:10]:
        print(feature, round(c, 2))
    print("-"*20)


# In[ ]:


prediction_train_rate, bias_train_rate, contributions_train_rate = ti.predict(rfr_cases_rate, X_train_Italy)


# In[ ]:


#Anlyzing form the beginning of the peak till the end of it
for i in range(28,45):
    print("Instance", i)
    print("Prediction: ", prediction_train_rate[i])
    print("Bias (trainset mean)", bias_train_rate[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions_train_rate[i], 
                                 X_train.columns), 
                             key=lambda x: -abs(x[0]))[:10]:
        print(feature, round(c, 2))
    print("-"*20)


# In[ ]:


#Feature importances
feature_importances = pd.DataFrame(rfr_cases_rate.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_importances.iloc[:10]


# So, the result is still quite bad, R2 of -0.45... However, it's interesting to see that it was able to understand the overall behaviour for some countries, whereas for others it has predicted the exact oposite behaviour.
# More interesting is that the importance of the days is much bigger for this model than for the cases model. The country also plays an important role, which was not seen in the previous model, which means that the predictions are tailored for each country and not simply made generically based on the measures. If we can improve the quality of this model, we can simply use it's predictions of rate of change to get the actual values of the number of cases.
# 
# There is another important advantage of using the rate of change instead of the cases, specifically for tree based models. The value of the cases is increasing exponencially, always increasing, however the trees are trained with the beginning of data, with low values. Therefore, the future predictions are out of the region in which the model can make predictions because it will never make a prediction bigger than a value to which it was trained. This problem does not exist when you work with the rate. If you want to use the actual cases a different model should be used, such as Linear Regressionor or a Neural Network. 

# ### Deaths

# In[ ]:


rfr_deaths = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)


# In[ ]:


rfr_deaths.fit(X_train,y_train_death)


# In[ ]:


metrics_report(X_test, y_test_death, rfr_deaths)


# In[ ]:


metrics_report(X_train, y_train_death, rfr_deaths)


# In[ ]:


#Feature importances
feature_importances = pd.DataFrame(rfr_deaths.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_importances.iloc[:10]


# ### Other models
# I didn't have yet the time but I would like to test other models more appropriate for time-series prediction such as ARIMA or RNN. I would love to test an RNN but I guess that I have too few data, maybe if I use a small network?
# 
# The problem of those methods is that I would not have the insights of feature importance. Maybe I could use some correlation metrics, such as Spearman or PhiK?
# 
# Another thing I could do is, knowing that I have an exponential shape, to apply a log transformation and apply a simple linear regression and analyze the coefficients to have an idea of it's importances. This would fit well to the actual prediction of cases, without the need of predicting the rate, but I fear to loose some important, non-linear behaviour. Moreover, the shape of the curves is exponencial for now, but it will soon not be anymore and we won't be able to make predictions after this point.
# 
# Having all this in mind, I think the best approach would be what I describe in the next section

# ## 3) Using the SEIR model

# SEIR is a mathematical model used for modeling the spread of an infectuous disease. It was a model used recently by a group of researchers from Imperial College London to predict the spread of the disease in the UK depending on the various strategies of the government:
# https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
# 
# For you to get a feel of how the model works there is also this website, developed by some Harvard researchers:
# https://alhill.shinyapps.io/COVID19seir/
# 
# And a notebook for the same model:
# https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
# 
# The problem is, however that this model depends on several parameters, which can be roughly estimated depending on historic data of other countries, the country's characterestics such as demographics and weather, the healthcare conditions and the government measures. Withot a good estimation of such values the model would be quite inacurate.
# 
# As we already have a feeling of the overall shape of the spread of the disease, my bet is thus that the best possible approach would not be to depart straight with a Machine Learning model, but rather use all the data we have to predict the values of the parameters that best fit the SEIR model to the actual curves at any given point in time. And by that we can actually see the effect of the multiple parameters on the spread of the disease.
# 
# I am currently working on that, as soon as I have anything worth sharing I update the notebook here.

# ## 4) Discussions

# I would love to hear your opinions and feedback on the several things I have presented along this notebook and on how I could better prepare the Dataset and make better models.
# 
# Moreover if you have any more datasets similar to this one, with the responses of the government, please let me know.
# It would be great if we could fill those blanks with actual data, and expand for new countries. It would be also great if we could get those data on a more granular level (states or cities) because often the measures of a state are even more important than those applied nation-wise.
# 
# However, to make such a dataset would be extremely time-consuming. What I think would be the best approach would be to make a crowd-sourcing project, allowing people from around the world to contribute. Do you know if there is anything like that happening? 
