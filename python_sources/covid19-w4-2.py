#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/bin/bash
set -e
 
usage() {
cat << EOF
Usage: $0 [OPTIONS] [LABEL]
Push a newly-built image with the given LABEL to gcr.io and DockerHub.
 
Options:
    -g, --gpu                   Push the image with GPU support.
    -s, --source-image IMAGE    Tag for the source image. 
EOF
}
 
SOURCE_IMAGE_TAG='kaggle/python-build:latest'
SOURCE_IMAGE_TAG_OVERRIDE=''
TARGET_IMAGE='gcr.io/kaggle-images/python'
 
while :; do
    case "$1" in 
        -h|--help)
            usage
            exit
            (";")
        -g|--gpu)
            SOURCE_IMAGE_TAG='kaggle/python-gpu-build:latest'
            TARGET_IMAGE='gcr.io/kaggle-private-byod/python'
            (";")
        -s|--source-image)
            if [[ -z $2 ]]; then
                usage
                printf 'ERROR: No IMAGE specified after the %s flag.\n' "$1" >&2
                exit
            fi
            SOURCE_IMAGE_TAG_OVERRIDE=$2
            shift # skip the flag value
            (";")
        -?*)
            usage
            printf 'ERROR: Unknown option: %s\n' "$1" >&2
            exit
            (";")
        *)            
            break
    esac
 
    shift
done
 
LABEL=${1:-testing}
 
if [[ -n "$SOURCE_IMAGE_TAG_OVERRIDE" ]]; then
    SOURCE_IMAGE_TAG="$SOURCE_IMAGE_TAG_OVERRIDE"
fi
 
readonly SOURCE_IMAGE_TAG
readonly TARGET_IMAGE
readonly LABEL
 
set -x
docker tag "${SOURCE_IMAGE_TAG}" "${TARGET_IMAGE}:${LABEL}"
gcloud docker -- push "${TARGET_IMAGE}:${LABEL}"
 


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Data input**

# In[ ]:


p1 = "/kaggle/input/covid19-global-forecasting-week-4/"
p2 = "/kaggle/input/world-bank-wdi-212-health-systems/"
p3 = "/kaggle/input/covid19inf/"
train = pd.read_csv(p1 + "train.csv")
test =  pd.read_csv(p1 + "test.csv")
submission =  pd.read_csv(p1 + "submission.csv")
health = pd.read_csv(p2 + "2.12_Health_systems.csv")
country = pd.read_csv(p3 + "covid19countryinfo.csv")
pollution = pd.read_csv(p3 + "region_pollution.csv")


# Some data munging in the contry dataset

# In[ ]:


country.drop(columns= country.columns[range(22,54)], inplace=True)
country["pop"] = country["pop"].str.replace(",", "").astype('float64')
country["quarantine"] = pd.to_datetime(country.quarantine)
country["schools"] = pd.to_datetime(country.schools)
country["restrictions"] = pd.to_datetime(country.restrictions)


# Data munging and merging

# In[ ]:


train["Date"] = pd.to_datetime(train.Date)
train["country_province"] = train["Province_State"]
train.country_province.fillna(train["Country_Region"], inplace=True)
test["Date"] = pd.to_datetime(test.Date)
test["country_province"] = test["Province_State"]
test.country_province.fillna(test["Country_Region"], inplace=True)
train = train.merge(country, how='left', left_on = ["country_province"], right_on = ["country"])
train = train.merge(pollution, how='left', left_on = ["country_province"], right_on = ["Region"])
train = train.merge(health, how='left', left_on = ["Country_Region", "Province_State"], right_on = ["Country_Region", "Province_State"])
test = test.merge(country, how='left', left_on = ["country_province"], right_on = ["country"])
test = test.merge(pollution, how='left', left_on = ["country_province"], right_on = ["Region"])
test = test.merge(health, how='left', left_on = ["Country_Region", "Province_State"], right_on = ["Country_Region", "Province_State"])
train["days"] = (train.Date - train.Date[0]).dt.days
test["days"] = (test.Date - train.Date[0]).dt.days

#The columns with region/state names in the different csvs are not longer needed
columns_to_drop = ["country", "Region", "Province_State", "Country_Region", "World_Bank_Name"]


# More data preparation, this time when it is in quarnatine, school stops and restriction. Those data are not complete and this is a pity but they are valuable.

# In[ ]:


train["in_quarantine"] = 0
train["in_schools"] = 0
train["in_restrictions"] = 0
test["in_quarantine"] = 0
test["in_schools"] = 0
test["in_restrictions"] = 0
for cp in train.country_province.unique():
    quarantine = country.loc[country.country == cp, "quarantine"]
    schools = country.loc[country.country == cp, "schools"]
    restrictions = country.loc[country.country == cp, "restrictions"]
    if (len(quarantine) > 0) and (quarantine.values[0] is not np.nan):
        date1 = pd.to_datetime(quarantine.values[0])
        train.loc[(train.country_province == cp) & (train.Date > date1), "in_quarantine"] = (train.Date - date1).dt.days
        test.loc[(test.country_province == cp) & (test.Date > date1), "in_quarantine"] = (test.Date - date1).dt.days
        
    if (len(schools) > 0) and (schools.values[0] is not np.nan):
        date1 = pd.to_datetime(schools.values[0])
        train.loc[(train.country_province == cp) & (train.Date > date1), "in_schools"] = (train.Date - date1).dt.days
        test.loc[(test.country_province == cp) & (test.Date > date1), "in_schools"] = (test.Date - date1).dt.days

    if (len(restrictions) > 0) and (restrictions.values[0] is not np.nan):
        date1 = pd.to_datetime(restrictions.values[0])
        train.loc[(train.country_province == cp) & (train.Date > date1), "in_restrictions"] = (train.Date - date1).dt.days
        test.loc[(test.country_province == cp) & (test.Date > date1), "in_restrictions"] = (test.Date - date1).dt.days

columns_to_drop += ["quarantine", "schools", "restrictions"]


# Label encode the countries and/or regions

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train["country_province"] = lb.fit_transform(train.country_province)
test["country_province"] = lb.transform(test.country_province)


# Generate three new variables, days from first death, first reported case and 100th reported case. The virus arrives one place by travelling that is not easy to track for the model.

# In[ ]:


train["days_from_first_case"] = 0
test["days_from_first_case"] = 0
train["days_from_first_death"] = 0
train["days_from_case_100"] = 0
test["days_from_case_100"] = 0
test["days_from_first_death"] = 0

dates = list(train.Date.unique())
for province in train.country_province.unique():
    #print(province)
    mask1 = train.country_province == province
    mask2 = train.ConfirmedCases > 1.0
    mask3 = train.ConfirmedCases > 100.0
    mask4 = train.Fatalities > 1.0
    try:
        idx1 = train.loc[mask1 & mask2 ,["ConfirmedCases"]].idxmin()[0]
        dateidx1 = train.iloc[idx1]["Date"]
    except:
        dateidx1 = test.Date.max()
        pass
    #print(dateidx1)
    train.loc[mask1 & (train.Date >= dateidx1), "days_from_first_case"] = (train.Date - dateidx1).dt.days
    test.loc[mask1 & (test.Date >= dateidx1), "days_from_first_case"] = (test.Date - dateidx1).dt.days
    
    try:
        idx1 = train.loc[mask1 & mask3 ,["ConfirmedCases"]].idxmin()[0]
        dateidx1 = train.iloc[idx1]["Date"]
    except:
        dateidx1 = test.Date.max()
        pass
    train.loc[mask1 & (train.Date >= dateidx1), "days_from_case_100"] = (train.Date - dateidx1).dt.days
    test.loc[mask1 & (test.Date >= dateidx1), "days_from_case_100"] = (test.Date - dateidx1).dt.days    

        
    try:
        idx1 = train.loc[mask1 & mask4 ,["Fatalities"]].idxmin()[0]
        dateidx1 = train.iloc[idx1]["Date"]
    except:
        dateidx1 = test.Date.max()
        pass
    train.loc[mask1 & (train.Date >= dateidx1), "days_from_first_death"] = (train.Date - dateidx1).dt.days
    test.loc[mask1 & (test.Date >= dateidx1), "days_from_first_death"] = (test.Date - dateidx1).dt.days    

train.fillna(value = 0, inplace = True)
test.fillna(value = 0, inplace = True)  


# Definition of lagged variables

# In[ ]:


#Construction of laged variables 
lag_number = 3
for lag in range(1, lag_number + 1):
    var_name = "cases_lag%d" % lag
    train[var_name] = train.ConfirmedCases.shift(periods = lag)
    train.loc[train.Date <= train.Date[lag - 1] , var_name] = 0
    var_name = "fatalities_lag%d" % lag
    train[var_name] = train.Fatalities.shift(periods = 1)
    train.loc[train.Date <= train.Date[lag - 1] , var_name] = 0


# Finally the train and test data can be prepared for the models eliminating the object variables and repited ones. From train data a small sample is taken for validation, the days that coincide in train and test data.

# In[ ]:


#Days that coincide in train and test
print(train.loc[train.Date.isin(test.Date.unique()), "Date"].unique())
# The smallest of those days will be the separation between train and validation
sep_date = train.loc[train.Date.isin(test.Date.unique()), "Date"].unique().min()

result_columns = ["ConfirmedCases", "Fatalities"]
X_train = train.loc[(train.Date<sep_date),].drop(columns = columns_to_drop + ["Id", "Date"] + result_columns)
y_train_cases = train.loc[(train.Date<sep_date),].ConfirmedCases
y_train_fatalities = train.loc[(train.Date<sep_date),].Fatalities

X_val = train.loc[(train.Date>=sep_date),].drop(columns = columns_to_drop + ["Id", "Date"] + result_columns)
y_val_cases = train.loc[(train.Date>=sep_date),].ConfirmedCases
y_val_fatalities = train.loc[(train.Date>=sep_date),].Fatalities

X_test = test.drop(columns = columns_to_drop + ["ForecastId", "Date"])


# Before starting the models a function is defined to reduce code to check errors

# In[ ]:


from sklearn.metrics import mean_squared_error
def validate_models(model_cases, model_fatalities):
    predict_train_cases = model_cases.predict(X_train)
    predict_val_cases = model_cases.predict(X_val)
    print("RMSE in train detected cases: ", np.sqrt(mean_squared_error(y_train_cases, predict_train_cases)))
    print("RMSE in validation detected cases: ", np.sqrt(mean_squared_error(y_val_cases, predict_val_cases)))
    predict_train_fatalities = model_fatalities.predict(X_train)
    predict_val_fatalities = model_fatalities.predict(X_val)
    print("RMSE in train fatalities: ", np.sqrt(mean_squared_error(y_train_fatalities, predict_train_fatalities)))
    print("RMSE in validation fatalities: ", np.sqrt(mean_squared_error(y_val_fatalities, predict_val_fatalities)))


# Now the models, let's start with a Random Forest and after a lightgbm

# In[ ]:


from sklearn.linear_model import LinearRegression
lm_cases = LinearRegression()
lm_cases.fit(X_train, y_train_cases)

lm_fatalities = LinearRegression()
lm_fatalities.fit(X_train, y_train_fatalities)

validate_models(lm_cases, lm_fatalities)


# In[ ]:


#RandomForest
from sklearn.ensemble import RandomForestRegressor
rf_cases = RandomForestRegressor(n_estimators= 400, max_depth=6, random_state=0, verbose=0, n_jobs=-1)
rf_cases.fit(X_train, y_train_cases)

rf_fatalities = RandomForestRegressor(n_estimators= 400, max_depth=6, random_state=0, verbose=0, n_jobs=-1)
rf_fatalities.fit(X_train, y_train_fatalities)

validate_models(rf_cases, rf_fatalities)


# In[ ]:


import lightgbm as lgb
lgb_params = {
               'feature_fraction': 0.8,
               'metric': 'rmse',
               'nthread':-1, 
               'min_data_in_leaf': 2**4,
               'bagging_fraction': 0.75, 
               'learning_rate': 0.5, 
               'objective': 'mse', 
               'bagging_seed': 2**5, 
               'num_leaves': 2**6,
               'bagging_freq':1,
               'verbose':1 
              }
lgbm_cases = lgb.train(lgb_params, 
                       train_set=lgb.Dataset(X_train, label=y_train_cases), 
                       valid_sets=lgb.Dataset(X_val, label=y_val_cases), 
                       num_boost_round=500)
lgbm_fatalities = lgb.train(lgb_params, 
                            train_set=lgb.Dataset(X_train, label=y_train_fatalities), 
                            valid_sets=lgb.Dataset(X_val, label=y_val_fatalities), 
                            num_boost_round=500)
validate_models(lgbm_cases, lgbm_fatalities)


# Both model worked well in train set but much worse in validation, seems overfitting. Let's see the features importances

# In[ ]:


lgb.plot_importance(lgbm_fatalities)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


# Finally the calculation for the test set requires a rolling forecast, going one by one to calculate the lags with the previous values

# In[ ]:


lags = {}
predict_test_cases, predict_test_fatalities = [] , []
test_min_day = test.days.min()
for i in range(1, lag_number + 1):
    lags["caseslag%d" % i] = 0
    lags["fatalitieslag%d" % i] = 0
    
for ind in tqdm_notebook(range(len(X_test))):
    #print("case: {} of {}".format(ind, len(X_test)))
    #First lag data are obtained either from previous calculations or train data
    if X_test.iloc[ind].days == test_min_day:
        #print(test_min_day)
        for i in range(1, lag_number + 1):
            mask1 = train.days == (test_min_day - i)
            mask2 = train.country_province == X_test.iloc[ind].country_province
            lags["caseslag%d" % i] = train.loc[mask1 & mask2, "ConfirmedCases"].values[0]
            lags["fatalitieslag%d" % i] = train.loc[mask1 & mask2, "Fatalities"].values[0]
    else:
        lags["caseslag1"] = pred_cases
        lags["fatalitieslag1"] = pred_fatalities
        for i in range(2, lag_number + 1):
            lags["caseslag%d" % i] = lags["caseslag%d" % (i-1)]
            lags["fatalitieslag%d" % i] = lags["fatalitieslag%d" % (i-1)]
    x_test = X_test.iloc[ind].copy()
    x_test =pd.DataFrame(x_test).transpose()
    for i in range(1, lag_number + 1):
        x_test["cases_lag%d" % i] = lags["caseslag%d" % i]
        x_test["fatalities_lag%d" % i] = lags["fatalitieslag%d" % i]
        
    pred_cases = rf_cases.predict(x_test)[0]
    pred_fatalities = rf_fatalities.predict(x_test)[0]
    predict_test_cases.append(pred_cases)
    predict_test_fatalities.append(pred_fatalities)
    


# submission

# In[ ]:


submission.ConfirmedCases = predict_test_cases
submission.Fatalities = predict_test_fatalities
submission.to_csv("/kaggle/working/submission.csv", index = False)

