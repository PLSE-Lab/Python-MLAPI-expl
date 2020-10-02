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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
print(df_train.shape)
print(df_test.shape)


# import pandas_profiling
# pandas_profiling.ProfileReport(df_train)

# In[ ]:


df_train.columns


# In[ ]:


df_train['Province/State'] = df_train.apply(
    lambda row: row['Country/Region'] if pd.isnull(row['Province/State']) else row['Province/State'],
    axis=1
)
df_test['Province/State'] = df_test.apply(
    lambda row: row['Country/Region'] if pd.isnull(row['Province/State']) else row['Province/State'],
    axis=1
)


# In[ ]:


df_train['Date'] = df_train.apply(
    lambda row: pd.Timestamp(row['Date']).value//10**9,
    axis=1
)
df_test['Date'] = df_test.apply(
    lambda row: pd.Timestamp(row['Date']).value//10**9,
    axis=1
)


# In[ ]:


import matplotlib.pyplot as plt
df = df_train.sort_values('Date')
plt.plot(df['Date'],np.log2(df['ConfirmedCases'])/np.log2(1.5))
plt.show()
plt.plot(df['Date'],np.log2(df['Fatalities'])/np.log2(1.5))
plt.show()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(df_train['Province/State'])
state_train = vectorizer.transform(df_train['Province/State'])
state_test = vectorizer.transform(df_test['Province/State'])

vectorizer = CountVectorizer(binary=True)
vectorizer.fit(df_train['Country/Region'])
country_train = vectorizer.transform(df_train['Country/Region'])
country_test = vectorizer.transform(df_test['Country/Region'])


# In[ ]:


normalizer = Normalizer()
normalizer.fit(df_train['Lat'].values.reshape(1, -1))
lat_train = normalizer.transform(df_train['Lat'].values.reshape(1, -1))
lat_test = normalizer.transform(df_test['Lat'].values.reshape(1, -1))

normalizer = Normalizer()
normalizer.fit(df_train['Long'].values.reshape(1, -1))
long_train = normalizer.transform(df_train['Long'].values.reshape(1, -1))
long_test = normalizer.transform(df_test['Long'].values.reshape(1, -1))

normalizer = Normalizer()
normalizer.fit(df_train['Date'].values.reshape(1, -1))
date_train = normalizer.transform(df_train['Date'].values.reshape(1, -1))
date_test = normalizer.transform(df_test['Date'].values.reshape(1, -1))


# In[ ]:


print(state_train.shape)
print(country_train.shape)
print(lat_train.reshape(-1, 1).shape)
print(long_train.shape)
print(date_train.shape)


# In[ ]:


from scipy.sparse import hstack
data_train = hstack([state_train, country_train, lat_train.reshape(-1, 1), long_train.reshape(-1, 1), date_train.reshape(-1, 1)])
data_test = hstack([state_test, country_test, lat_test.reshape(-1, 1), long_test.reshape(-1, 1), date_test.reshape(-1, 1)])
#data_train=data_train.todense()
#data_test=data_test.todense()


# In[ ]:


#print(data_test[0])


# from sklearn import linear_model
# data_country={}
# label_country={}
# for index, row in df_train.iterrows():
#     if row['Country/Region'] not in data_country:
#         data_country[row['Country/Region']]=[]
#         label_country[row['Country/Region']]=[]
#     
#     data_country[row['Country/Region']].append(data_train[index])
#     label_country[row['Country/Region']].append(row['ConfirmedCases'])
# model_country={}
# for country in data_country.keys():
#     model_country[country]=linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
#     #print(np.array(data_country[country]).reshape(len(label_country[country]),-1).shape)
#     #print(len(label_country[country]))
#     model_country[country].fit(np.array(data_country[country]).reshape(len(label_country[country]),-1),label_country[country])

# conf_cased_pred=[]
# for index, row in df_test.iterrows():
#     conf_cased_pred.append(model_country[row['Country/Region']].predict(data_test[index]))
# normalizer = Normalizer()
# normalizer.fit(df_train['ConfirmedCases'].values.reshape(1, -1))
# conf_cased_pred_train = normalizer.transform(df_train['ConfirmedCases'].values.reshape(1, -1))
# conf_cased_pred_test = normalizer.transform(conf_cased_pred)
# data_train_with_conf = hstack([state_train, country_train, lat_train.reshape(-1, 1), long_train.reshape(-1, 1), date_train.reshape(-1, 1), conf_cased_pred_train.reshape(-1,1)])
# data_test_with_conf = hstack([state_test, country_test, lat_test.reshape(-1, 1), long_test.reshape(-1, 1), date_test.reshape(-1, 1),conf_cased_pred_test.reshape(-1, 1)])
# data_train_with_conf = data_train_with_conf.todense()
# data_test_with_conf = data_test_with_conf.todense()

# data_country={}
# label_country={}
# for index, row in df_train.iterrows():
#     if row['Country/Region'] not in data_country:
#         data_country[row['Country/Region']]=[]
#         label_country[row['Country/Region']]=[]
#     
#     data_country[row['Country/Region']].append(data_train_with_conf[index])
#     label_country[row['Country/Region']].append(row['Fatalities'])
# model_country={}
# for country in data_country.keys():
#     model_country[country]=linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
#     #print(np.array(data_country[country]).reshape(len(label_country[country]),-1).shape)
#     #print(len(label_country[country]))
#     model_country[country].fit(np.array(data_country[country]).reshape(len(label_country[country]),-1),label_country[country])

# fatalities_pred=[]
# for index, row in df_test.iterrows():
#     fatalities_pred.append(model_country[row['Country/Region']].predict(data_test_with_conf[index]))

# from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# param_grid = dict(n_estimators=np.array([50,100,200,300,400]), subsample=np.array([0.5,0.6,0.7,0.8,0.9,1]), max_depth=np.array([3,5,7,9]))
# model = GradientBoostingRegressor(random_state=21)
# kfold = KFold(n_splits=10, random_state=21)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
# grid_result = grid.fit(data_train.todense(), df_train['ConfirmedCases'])
# print(grid_result.best_estimator_)

# In[ ]:


from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
clf = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints=None,
             n_estimators=1000, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
clf.fit(data_train.todense(), df_train['ConfirmedCases'])


# In[ ]:


#conf_cased_pred_tr = clf.predict(data_test.todense())
conf_cased_pred = clf.predict(data_test.todense())
normalizer = Normalizer()
normalizer.fit(df_train['ConfirmedCases'].values.reshape(1, -1))
conf_cased_pred_train = normalizer.transform(df_train['ConfirmedCases'].values.reshape(1, -1))
conf_cased_pred_test = normalizer.transform(conf_cased_pred.reshape(1, -1))
data_train_with_conf = hstack([state_train, country_train, lat_train.reshape(-1, 1), long_train.reshape(-1, 1), date_train.reshape(-1, 1), conf_cased_pred_train.reshape(-1,1)])
data_test_with_conf = hstack([state_test, country_test, lat_test.reshape(-1, 1), long_test.reshape(-1, 1), date_test.reshape(-1, 1),conf_cased_pred_test.reshape(-1, 1)])


# In[ ]:


print(np.mean(conf_cased_pred))


# param_grid = dict(n_estimators=np.array([50,100,200,300,400]))
# model = GradientBoostingRegressor(random_state=21)
# kfold = KFold(n_splits=10, random_state=21)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
# grid_result = grid.fit(data_train_with_conf.todense(), df_train['Fatalities'])
# print(grid_result.best_estimator_)

# In[ ]:


clf = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints=None,
             n_estimators=1000, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
clf.fit(data_train_with_conf.todense(), np.log(df_train['Fatalities']+0.000000001))
fatalities_pred = clf.predict(data_test_with_conf.todense())
#fatalities_pred = np.exp(fatalities_pred)


# In[ ]:


print(np.mean(fatalities_pred))


# conf_cased_pred=np.concatenate( conf_cased_pred, axis=0 )
# fatalities_pred= np.concatenate( fatalities_pred, axis=0 )
# print(conf_cased_pred[:5])
# print(fatalities_pred[:5])

# In[ ]:


def make_submission(conf, fat, sub_name):
    my_submission = pd.DataFrame({'ForecastId':pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv').ForecastId,'ConfirmedCases':conf, 'Fatalities':fat})
    my_submission.to_csv('{}.csv'.format(sub_name),index=False)
    print('A submission file has been made')
make_submission(conf_cased_pred,fatalities_pred,'submission')


# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# NN_model = Sequential()
# 
# # The Input Layer :
# NN_model.add(Dense(256, kernel_initializer='normal',input_dim = 494, activation='relu'))
# 
# # The Hidden Layers :
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
# #NN_model.add(Flatten())
# NN_model.add(Dropout(0.4))
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# 
# # The Output Layer :
# NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
# 
# # Compile the network :
# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# NN_model.summary()
# NN_model.fit(data_train.todense(), df_train['ConfirmedCases'], epochs=100,batch_size=64, validation_split = 0.1)

# In[ ]:





# conf_cased_pred = NN_model.predict(data_test.todense())
# normalizer = Normalizer()
# normalizer.fit(df_train['ConfirmedCases'].values.reshape(1, -1))
# conf_cased_pred_train = normalizer.transform(df_train['ConfirmedCases'].values.reshape(1, -1))
# conf_cased_pred_test = normalizer.transform(conf_cased_pred.reshape(1, -1))
# data_train_with_conf = hstack([state_train, country_train, lat_train.reshape(-1, 1), long_train.reshape(-1, 1), date_train.reshape(-1, 1), conf_cased_pred_train.reshape(-1,1)])
# data_test_with_conf = hstack([state_test, country_test, lat_test.reshape(-1, 1), long_test.reshape(-1, 1), date_test.reshape(-1, 1),conf_cased_pred_test.reshape(-1, 1)])

# NN_model = Sequential()
# # The Input Layer :
# NN_model.add(Dense(256, kernel_initializer='normal',input_dim = 495, activation='relu'))
# 
# # The Hidden Layers :
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
# #NN_model.add(Flatten())
# NN_model.add(Dropout(0.4))
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# 
# # The Output Layer :
# NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
# 
# # Compile the network :
# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# NN_model.summary()
# NN_model.fit(data_train_with_conf.todense(), df_train['Fatalities'], epochs=100,batch_size=64, validation_split = 0.1)

# fatalities_pred = NN_model.predict(data_test_with_conf.todense())

# print(len(np.ndarray.flatten(conf_cased_pred)))
# print(fatalities_pred.shape)

# def make_submission(conf, fat, sub_name):
#     my_submission = pd.DataFrame({'ForecastId':pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv').ForecastId,'ConfirmedCases':conf, 'Fatalities':fat})
#     my_submission.to_csv('{}.csv'.format(sub_name),index=False)
#     print('A submission file has been made')
# make_submission(np.ndarray.flatten(conf_cased_pred),np.ndarray.flatten(fatalities_pred),'submission')
