#!/usr/bin/env python
# coding: utf-8

# **LearnX Sales Forecasting**
# 
# https://datahack.analyticsvidhya.com/contest/women-in-the-loop-a-data-science-hackathon-by-bain/
# 
# LearnX is an online learning platform aimed at professionals and students. LearnX serves as a market place that allows instructors to build online courses on topics of their expertise which is later published after due diligence by the LearnX team. The platform covers a wide variety of topics including Development, Business, Finance & Accounting & Software Marketing and so on
# 
# Effective forecasting for course sales gives essential insight into upcoming cash flow meaning business can more accurately plan the budget to pay instructors and other operational costs and invest in the expansion of the business.
# 
# Sales data for more than 2 years from 600 courses of LearnX's top domains is available along with information on
# 
#     Competition in the market for each course
#     Course Type (Course/Program/Degree)
#     Holiday Information for each day
#     User Traffic on Course Page for each day
# 
# Your task is to predict the course sales for each course in the test set for the next 60 days
# 
# **Data Dictionary**
# 
# **Train (Historical Sales Data)**
# 
# **Variable Definition**
# * ID 	Unique Identifier for a row
# * Day_No 	Day Number
# * Course_ID Unique ID for a course
# * Course_Domain Course Domain (Development, Finance etc.)
# * Course_Type Course/Program/Degree
# * Short_Promotion Whether Short Term Promotion is Live
# * Public_Holiday Regional/Public Holiday
# * Long_Promotion Whether Long Term Promotion is Live for the course
# * User_Traffic Number of customers landing on the course page
# * Competition_Metric 	A metric defining the strength of competition 
# * Sales (Target) Total Course Sales
# 
# **Test (Next 60 Days)**
# 
# This file contains the store and day number for which the participant needs to submit predictions/forecasts
# 
# **Variable Definition**
# * ID 	Unique Identifier for a row
# * Day_No 	Day Number
# * Course_ID 	Unique ID for a course
# * Course_Domain 	Course Domain (Development, Finance etc.)
# * Course_Type 	Course/Program/Degree
# * Short_Promotion 	Whether Short Term Promotion is Live
# * Public_Holiday 	Regional/Public Holiday
# * Long_Promotion 	Whether Long Term Promotion is Live for the course
# * Competition_Metric 	A metric defining the strength of competition
#  
# **Sample Submission**
# 
# This file contains the exact submission format for the forecasts. Please submit csv file only.
# 
# Variable Definition
# ID 	Unique Identifier for a row
# Sales 	(Target) Total Course Sales predicted from the test set
# 
# **Evaluation**
# 
# The evaluation metric for this competition is 1000*RMSLE where RMSLE is Root of Mean Squared Logarithmic Error across all entries in the test set.
# 
# **Public and Private Split**
# 
# Test data is further divided into Public (First 20 Days) and Private (Next 40 Days)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import lognorm, gamma
import collections
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/train.csv')
test = pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/test_QkPvNLx.csv')
sample_submission = pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/sample_submission_pn2DrMq.csv')


# In[ ]:


print("Course_Domain: ", train['Course_Domain'].unique(), "Course_Type: ", train['Course_Type'].unique())


# In[ ]:


def encode_features(dataset):
    course_domain_le = preprocessing.LabelEncoder()
    course_domain_le.fit(dataset['Course_Domain'].unique())
    dataset['Course_Domain'] = course_domain_le.transform(dataset['Course_Domain'])
    
    course_type_le = preprocessing.LabelEncoder()
    course_type_le.fit(dataset['Course_Type'].unique())
    dataset['Course_Type'] = course_type_le.transform(dataset['Course_Type'])

encode_features(train)
encode_features(test)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


from datetime import date, timedelta

def day_to_date(dataset):
    start = date(2016,12,31)
    dataset['Date'] = dataset['Day_No'].apply(lambda x: start + timedelta(x)) 
   
day_to_date(train)
day_to_date(test)

def day_month_year(dataset): 
    dataset['Day'] = dataset['Date'].apply(lambda x: x.day)
    dataset['Month'] = dataset['Date'].apply(lambda x: x.month)
    dataset['Year'] = dataset['Date'].apply(lambda x: x.year)

day_month_year(train)
day_month_year(test)

train = train[['ID', 'Day_No', 'Date', 'Day', 'Month', 'Year', 'Course_ID', 'Course_Domain', 'Course_Type',
       'Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'User_Traffic', 'Competition_Metric', 'Sales']]

test = test[['ID', 'Day_No', 'Date', 'Day', 'Month', 'Year', 'Course_ID', 'Course_Domain', 'Course_Type',
             'Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'Competition_Metric']]


# In[ ]:


test.describe()


# In[ ]:


ax = sns.distplot(train['Sales'])
plt.show()


# In[ ]:


train['Course_Domain'].value_counts()


# In[ ]:


test['Course_Domain'].value_counts()


# In[ ]:


print("Train: ", train.shape, "Test: ", test.shape)


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


plt.figure(figsize=(15, 15))
sns.heatmap(train.corr(), vmin=-1, center=0, vmax=1, annot=True, square=True)
plt.show()


# In[ ]:


X_train = train.drop(['Sales', 'ID', 'User_Traffic', 'Date', 'Day_No'], axis=1)
y_train = train['Sales'].values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 890)

X_test = test.drop(['ID', 'Date', 'Day_No'], axis=1)

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val)


# from hyperopt import STATUS_OK
# from hyperopt import hp
# from hyperopt import tpe
# from hyperopt import fmin
# from hyperopt import Trials
# 
# N_FOLDS = 5
# 
# def rmsle(preds, lgb_train):
#     eval_name = 'rmsle'
#     eval_result = np.sqrt(mean_squared_log_error(preds, lgb_train.get_label()))
#     return (eval_name, eval_result*1000, False)
# 
# def objective(params, n_folds = N_FOLDS):
#     cv_results = lgb.cv(params, lgb_train, num_boost_round = 1000, nfold = 5, feval = rmsle, early_stopping_rounds = 10, seed = 50)
#     best_score = min(cv_results['rmsle-mean'])
#     return {'loss': best_score, 'params': params, 'status': STATUS_OK}
# 
# space = { 
#             'task': hp.choice('task', ['train']),
#             'objective': hp.choice('objective', ['gamma']),
#             'metric' : hp.choice('metric', ['None']),
#             'boosting': hp.choice('boosting', ['gbdt']),
#             'learning_rate': hp.loguniform('learning_rate',np.log(0.003), np.log(0.5)),
#             'num_leaves': hp.choice('num_leaves', range(2, 100, 5)),
#             'max_depth': hp.choice('max_depth', range(1, 30, 5)),
#             'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 1.0),
#             'bagging_freq': hp.choice('bagging_freq', range(1, 10, 1)),
#             'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0),
#             'max_bin': hp.choice('max_bin', range(200, 256, 5)),
#             'min_data_in_leaf': hp.choice('min_data_in_leaf', range(10, 1000, 1)),
#             'subsample': hp.uniform('subsample', 0.1, 1.0),
#             'bagging_seed': hp.choice('bagging_seed', range(1, 10, 1)),
#             'feature_fraction_seed': hp.choice('feature_fraction_seed', range(1, 10, 1)),
#         }
# 
# trials = Trials()
# 
# best = fmin(fn = objective, space = space, trials=trials, algo=tpe.suggest, max_evals = 80)
# 
# print(best)

# {'bagging_fraction': 0.9411426522615599,
#  'bagging_freq': 6,
#  'bagging_seed': 3,
#  'boosting': 0,
#  'feature_fraction': 0.8148569473842407,
#  'feature_fraction_seed': 6,
#  'learning_rate': 0.1818641793327766,
#  'max_bin': 2,
#  'max_depth': 2,
#  'metric': 0,
#  'min_data_in_leaf': 75,
#  'num_leaves': 15,
#  'objective': 0,
#  'subsample': 0.535865899859516,
#  'task': 0}

# In[ ]:


evals_result = {} 

params = {
        'task': 'train',
        'objective': 'gamma',
        'metric' : 'None',
        'boosting': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 100,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'n_estimators': 1000,
    }

def rmsle(preds, lgb_train):
    eval_name = "rmsle"
    eval_result = np.sqrt(mean_squared_log_error(preds, lgb_train.get_label()))
    return (eval_name, eval_result*1000, False)


cv_results = lgb.cv(params, lgb_train, num_boost_round = 1000, nfold = 5, feval = rmsle, early_stopping_rounds = 10, verbose_eval = 100, seed = 50)

lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, feval = rmsle,  evals_result = evals_result, verbose_eval = 100)


# In[ ]:


# plot cv metric
od = collections.OrderedDict()
d = {}
results = cv_results['rmsle-mean']
od['rmsle'] = results
d['cv'] = od

ax = lgb.plot_metric(d,title='Metric during cross-validation', metric='rmsle')
plt.show()

print("CV best score: " + str(min(cv_results['rmsle-mean'])))


# In[ ]:


# plot train metric 
ax = lgb.plot_metric(evals_result, metric='rmsle')
plt.show()

print("Train best score: " + str(min(evals_result['valid_0']['rmsle'])))

# plot feature importance
lgb.plot_importance(lgbm_model)


# In[ ]:


predictions = lgbm_model.predict(X_test)

# plot predictions
ax = sns.distplot(predictions)
plt.show()

# Writing output to file
subm = pd.DataFrame()
subm['ID'] = test['ID']
subm['Sales'] = predictions

subm.to_csv("/kaggle/working/" + 'submission.csv', index=False)
subm

