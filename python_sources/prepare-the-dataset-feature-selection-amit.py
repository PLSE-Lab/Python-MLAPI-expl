#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
print(train.shape)
train.head()


# In[ ]:


physician_availibility = pd.read_csv('../input/uncover/world_bank/physicians-per-1-000-people.csv')
print(physician_availibility.shape)
physician_availibility.head()


# In[ ]:


physician_availibility['physicians_per_1000'] = physician_availibility.drop(['country_name','country_code','indicator_name','indicator_code'], axis=1).mean(axis=1)
physician_availibility = physician_availibility[['country_name','country_code','physicians_per_1000']]
physician_availibility.head()


# In[ ]:


handwashing_facilities = pd.read_csv('../input/uncover/world_bank/people-with-basic-handwashing-facilities-including-soap-and-water-of-population.csv')
handwashing_facilities.head()


# In[ ]:


handwashing_facilities['indicator_name'].iloc[1]


# In[ ]:


handwashing_facilities['handwashing_facility_pct'] = handwashing_facilities.drop(['country_name','country_code','indicator_name','indicator_code'], axis=1).mean(axis=1)
handwashing_facilities = handwashing_facilities[['country_name','country_code','handwashing_facility_pct']]
handwashing_facilities.head()


# In[ ]:


international_debt = pd.read_csv('../input/uncover/world_bank/international-debt.csv')
international_debt.head()


# In[ ]:


international_debt.drop_duplicates(inplace=True)
international_debt.head()


# In[ ]:


international_debt = international_debt.groupby(['country_name','country_code']).value.sum().to_frame()
international_debt.head()


# In[ ]:


international_debt.rename(columns={"value": "international_debt_USD"}, inplace=True)
print(international_debt.shape)
international_debt.head()


# In[ ]:


hospital_beds = pd.read_csv('../input/uncover/world_bank/hospital-beds-per-1-000-people.csv')
hospital_beds.head()


# In[ ]:


hospital_beds['hospital_beds_per_1000'] = hospital_beds.drop(['country_name','country_code','indicator_name','indicator_code'], axis=1).mean(axis=1)
hospital_beds = hospital_beds[['country_name','country_code','hospital_beds_per_1000']]
hospital_beds.head()


# In[ ]:


specialist_doctors = pd.read_csv('../input/uncover/world_bank/specialist-surgical-workforce-per-100-000-population.csv')
specialist_doctors.head()


# In[ ]:


specialist_doctors['specialist_doctors_per_1000'] = (specialist_doctors.drop(['country_name','country_code','indicator_name','indicator_code'], axis=1).mean(axis=1))/100
specialist_doctors = specialist_doctors[['country_name','country_code','specialist_doctors_per_1000']]
specialist_doctors.head()


# In[ ]:


glbl_population = pd.read_csv('../input/uncover/world_bank/global-population.csv')
glbl_population.head()


# In[ ]:


glbl_population['population'] = glbl_population.drop(['country','country_code'], axis=1).mean(axis=1)
glbl_population = glbl_population[['country','country_code','population']]
glbl_population.head()


# In[ ]:


health_workers = pd.read_csv('../input/uncover/world_bank/community-health-workers-per-1-000-people.csv')
health_workers.head()


# In[ ]:


health_workers['health_workers_per_1000'] = health_workers.drop(['country_name','country_code','indicator_name','indicator_code'], axis=1).mean(axis=1)
health_workers = health_workers[['country_name','country_code','health_workers_per_1000']]
health_workers.head()


# In[ ]:


nurses = pd.read_csv('../input/uncover/world_bank/nurses-and-midwives-per-1-000-people.csv')
nurses.head()


# In[ ]:


nurses['nurses_per_1000'] = nurses.drop(['country_name','country_code','indicator_name','indicator_code'], axis=1).mean(axis=1)
nurses = nurses[['country_name','country_code','nurses_per_1000']]
nurses.head()


# In[ ]:


health_factors = pd.merge(physician_availibility, specialist_doctors, on=['country_name','country_code'])
health_factors = pd.merge(health_factors, nurses, on=['country_name','country_code'])
health_factors = pd.merge(health_factors, health_workers, on=['country_name','country_code'])
health_factors = pd.merge(health_factors, hospital_beds, on=['country_name','country_code'])
health_factors = pd.merge(health_factors, handwashing_facilities, on=['country_name','country_code'])
health_factors = pd.merge(health_factors, international_debt, how='left', on=['country_name','country_code'])
health_factors = pd.merge(health_factors, glbl_population, left_on=['country_name'], right_on=['country'])
health_factors = health_factors.drop(['country','country_code_x','country_code_y'],axis=1)
print(health_factors.shape)
health_factors.head()


# In[ ]:


health_factors['international_debt_USD_per_1000'] = (health_factors['international_debt_USD'] / health_factors['population'])*1000
health_factors.head()


# In[ ]:


health_factors['handwashing_facility_pct'] = health_factors['handwashing_facility_pct'] / 100
health_factors.head()


# In[ ]:


t1 = train.groupby(['Country_Region']).ConfirmedCases.sum().to_frame()


# In[ ]:


t2 = train.groupby(['Country_Region']).Fatalities.sum().to_frame()


# In[ ]:


temp = pd.merge(t1, t2, on=['Country_Region'])
print(temp.shape)
temp.head()


# In[ ]:


health_factors = pd.merge(health_factors, temp, how='left', left_on=['country_name'], right_on=['Country_Region'])
print(health_factors.shape)
health_factors.head()


# In[ ]:


health_factors['confirmed_cases_per_1000'] = (health_factors['ConfirmedCases'] / health_factors['population'])*1000
health_factors.head()


# In[ ]:


health_factors['fatalities_per_1000'] = (health_factors['Fatalities'] / health_factors['population'])*1000
health_factors.head()


# In[ ]:


health_factors.columns


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = health_factors[['physicians_per_1000','specialist_doctors_per_1000'
                     ,'nurses_per_1000','health_workers_per_1000'
                     ,'hospital_beds_per_1000','handwashing_facility_pct'
                     ,'international_debt_USD_per_1000'
                      ,'confirmed_cases_per_1000']].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["confirmed_cases_per_1000"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features


# In[ ]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = health_factors[['physicians_per_1000','specialist_doctors_per_1000'
                     ,'nurses_per_1000','health_workers_per_1000'
                     ,'hospital_beds_per_1000','handwashing_facility_pct'
                     ,'international_debt_USD_per_1000'
                      ,'fatalities_per_1000']].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["fatalities_per_1000"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.4]
relevant_features


# In[ ]:


# check for missing values
health_factors.isnull().any().sum()
health_factors.fillna(health_factors.mean(), inplace=True)


# In[ ]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
X,y = health_factors[['physicians_per_1000','specialist_doctors_per_1000'
                     ,'nurses_per_1000','health_workers_per_1000'
                     ,'hospital_beds_per_1000','handwashing_facility_pct'
                     ,'international_debt_USD_per_1000']], health_factors.confirmed_cases_per_1000


# In[ ]:


#Backward Elimination
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues


# In[ ]:


#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:


#Recursive Feature Elimination
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


#no of features
nof_list=np.arange(1,7)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[ ]:


cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 2)            
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# In[ ]:


#Lasso Regularization
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:


X,y = health_factors[['physicians_per_1000','specialist_doctors_per_1000'
                     ,'nurses_per_1000','health_workers_per_1000'
                     ,'hospital_beds_per_1000','handwashing_facility_pct'
                     ,'international_debt_USD_per_1000']], health_factors.fatalities_per_1000


# In[ ]:


#Backward Elimination
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues


# In[ ]:


#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:


#Recursive Feature Elimination
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


#no of features
nof_list=np.arange(1,7)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[ ]:


cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 5)           
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# In[ ]:


#Lasso Regularization
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

