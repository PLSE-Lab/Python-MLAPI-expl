#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


PATH_WEEK1='/kaggle/input/covid19-global-forecasting-week-1'


# In[ ]:


df_train = pd.read_csv(f'{PATH_WEEK1}/train.csv')


# In[ ]:


df_test = pd.read_csv(f'{PATH_WEEK1}/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.tail()


# In[ ]:


df_train.rename(columns={'Country/Region':'Country'}, inplace=True)
df_test.rename(columns={'Country/Region':'Country'}, inplace=True)

df_train.rename(columns={'Province/State':'State'}, inplace=True)
df_test.rename(columns={'Province/State':'State'}, inplace=True)


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


import plotly.express as px
pxdf = px.data.gapminder()

country_isoAlpha = pxdf[['country', 'iso_alpha']].drop_duplicates()
country_isoAlpha.rename(columns = {'country':'Country'}, inplace=True)
country_isoAlpha.set_index('Country', inplace=True)
country_map = country_isoAlpha.to_dict('index')


# In[ ]:


def getCountryIsoAlpha(country):
    try:
        return country_map[country]['iso_alpha']
    except:
        return country


# In[ ]:


df_train['iso_alpha'] = df_train['Country'].apply(getCountryIsoAlpha)
df_train.info()


# In[ ]:


df_train.isna().sum()


# In[ ]:


df_train.Country.unique()


# In[ ]:


df_plot = df_train.loc[:,['Date', 'Country', 'ConfirmedCases']]
df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")
df_plot.loc[:, 'Size'] = np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*100)
fig = px.scatter_geo(df_plot.groupby(['Date', 'Country']).max().reset_index(),
                     locations="Country",
                     locationmode = "country names",
                     hover_name="Country",
                     color="ConfirmedCases",
                     animation_frame="Date", 
                     size='Size',
                     #projection="natural earth",
                     title="Rise of Coronavirus Confirmed Cases")
fig.show()


# In[ ]:


df_train.drop(columns='iso_alpha', inplace=True)
df_train


# In[ ]:


X_Train = df_train.copy()

X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%Y%m%d")
X_Train["Date"]  = X_Train["Date"].astype(int)

X_Train.drop(columns=['ConfirmedCases', 'Fatalities'], inplace=True)
X_Train.drop(columns=['Id', 'State', 'Country', 'Lat'], inplace=True)
X_Train.head()


# X_Train['Year'] = X_Train.Date.dt.year
# X_Train['Month'] = X_Train.Date.dt.month
# X_Train['Day'] = X_Train.Date.dt.day
# X_Train.drop(columns = 'Date', inplace=True)

# In[ ]:


X_Train.head()


# EMPTY_VAL = "EMPTY_VAL"
# X_Train['State'].fillna(EMPTY_VAL, inplace=True)

# def fillState(state, country):
#     if state == EMPTY_VAL: return country
#     return state

# X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

# In[ ]:


X_Train.head()


# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# X_Train.Country = le.fit_transform(X_Train.Country)
# X_Train['State'] = le.fit_transform(X_Train['State'])
# X_Train.tail()

# columns = ['Lat', 'Long', 'Date']
# X_Train = df_train[columns]
# X_Train['Date'] = X_Train.Date.dt.strftime('%m%d')
# X_Train.Date.astype('int')
# X_Train.head()

# In[ ]:


y1_Train = df_train.ConfirmedCases


# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')


# from sklearn.model_selection import train_test_spdlit
# X_train, X_test, y_train, y_test = train_test_split(X_Train, y1_Train, test_size=0.25,random_state=1)

# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# 
# lin_reg.fit(X_train, y_train)
# y_pred = lin_reg.predict(X_test)
# 
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# print(mean_absolute_error(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))
# print(np.sqrt(mean_squared_error(y_test, y_pred)))

# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# 
# scores1 = cross_val_score(lin_reg, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
# scores2 = cross_val_score(lin_reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
# scores3 = cross_val_score(lin_reg, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')
# 
# print(scores1)
# print(scores2)
# print(scores3)

# parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LinearRegression
# 
# lin_reg = LinearRegression()
# grid = GridSearchCV(lin_reg, parameters, cv=10)
# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)
# 
# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(y_test, y_pred))
# 
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_pred))
# print(np.sqrt(mean_squared_error(y_test, y_pred)))

# from sklearn.ensemble import RandomForestRegressor
# 
# dec_tree_reg = RandomForestRegressor()
# #scores1 = cross_val_score(dec_tree_reg, X_Train, y1_Train, cv=10, scoring='neg_mean_absolute_error')
# #scores2 = cross_val_score(dec_tree_reg, X_Train, y1_Train, cv=10, scoring='neg_mean_squared_error')
# scores3 = cross_val_score(dec_tree_reg, X_Train, y1_Train, cv=10, scoring='neg_root_mean_squared_error')
# 
# #print(scores1)
# #print(scores2)
# print(scores3)

# In[ ]:


from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor 

MODELS = {"Linear_Reg": LinearRegression(), "KNN_Reg": KNeighborsRegressor(), "LinearSVR_Reg": LinearSVR(), "DecisionTree_Reg": DecisionTreeRegressor(), "DecisionTree_Class": DecisionTreeClassifier(), "ExtraTree_Reg": ExtraTreeRegressor(), "RandomForest_Reg": RandomForestRegressor(), "GB_Reg": GradientBoostingRegressor(), "XGB_Reg": XGBRegressor() }
# "SDG_Reg": SGDRegressor(), "RN_Reg": RadiusNeighborsRegressor(), "NuSVR_Reg": NuSVR(), "SVR_Reg": SVR(),


# import time
# from sklearn.model_selection import cross_val_score
# 
# for MODEL_NAME, MODEL in MODELS.items():
#     start = time.time()
#     scores = cross_val_score(MODEL, X_Train, y1_Train, cv=10, scoring='neg_root_mean_squared_error')
#     print (f'{MODEL_NAME} took a Time: {time.time() - start}')
#     print ("R2 SCore\n", scores)
#     print (f'(Min, Max, Mean) Values are : ({scores.min(), scores.max(), scores.mean()})')
#     print ()

# From the above *RMSE*, looks like **Decision Tree Regressor** predicts better

# from sklearn.model_selection import TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=5)
# i = 1
# scores = []
# for tr_index, val_index in tscv.split(X_Train):
#     print(X[tr_index])

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# 
# df = pd.DataFrame({'num_legs': [2, 4, 8, 0], 'num_wings': [2, 0, 0, 0], 'num_specimen_seen': [10, 2, 1, 8]}, index=[0,1,2,3])
# 
# kf = KFold(n_splits=2, shuffle=True)
# for train_index, val_index in kf.split(df):
#     df_train, df_val = df[train_index], df[val_index]

# from sklearn.model_selection import TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=5)
# i = 1
# scores = []
# for tr_index, val_index in tscv.split(X_Train):
#     X_tr, X_val = X[tr_index], X[val_index]
#     y_tr, y_val = y1[tr_index], y1[val_index]
#     for mf in np.linspace(100, 150, 6):
#         for ne in np.linspace(50, 100, 6):
#             for md in np.linspace(20, 40, 5):
#                 for msl in np.linspace(30, 100, 8):
#                     rfr = RandomForestRegressor(
#                         max_features=int(mf),
#                         n_estimators=int(ne),
#                         max_depth=int(md),
#                         min_samples_leaf=int(msl))
#                     rfr.fit(X_tr, y_tr)
#                     scores.append([i,
#                                   mf, 
#                                   ne,
#                                   md, 
#                                   msl, 
#                                   rfr.score(X_val, y_val)])
#     i += 1
# print(scores)

# model = XGBRegressor()
# param_search = {'max_depth' : [3, 5]}
# 
# tscv = TimeSeriesSplit(n_splits=2)
# gsearch = GridSearchCV(estimator=model, cv=tscv,
#                         param_grid=param_search)
# gsearch.fit(X_Train, y1_Train)
# y_pred = gsearch.predict(X_Test)
# y_pred

# In[ ]:


from sklearn.model_selection import GridSearchCV
import time

model = XGBRegressor()


# In[ ]:


#param_grid = {"criterion": ["mse", "mae"], "min_samples_split": [10, 20], "max_depth": [2, 6], "min_samples_leaf": [20, 40], "max_leaf_nodes": [5, 20]} #DTR
#param_grid = {"criterion": ["mae"], 'max_depth': [2], 'max_leaf_nodes': [5], 'min_samples_leaf': [20], 'min_samples_split': [10]} #DTR best
#param_grid = {'criterion':['gini'], 'max_depth': np.arange(2,5), 'min_samples_leaf': range(1,5)} #DTC
#param_grid = {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1} #DTC best
#param_grid = {'n_neighbors':range(1,2,4), 'leaf_size':[4,5,6], 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree','kd_tree','brute']} #KNR
#param_grid = {'algorithm': ['auto'], 'leaf_size': [4], 'n_neighbors': [1], 'weights': ['distance']} #KNR best
#param_grid = {'nthread':[4], 'objective':['reg:linear'], 'learning_rate': [.03, 0.05], 'max_depth': [5, 6], 'min_child_weight': [4], 'silent': [1], 'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [500, 1000]} #XGBR
param_grid = {'n_estimators': [1250]}


# In[ ]:


start = time.time()
grid_cv = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_cv.fit(X_Train, y1_Train)
print (f'{type(model).__name__} Hyper Paramter Tuning took a Time: {time.time() - start}')


# In[ ]:


print("Mean Squared Error: {}".format(grid_cv.best_score_))
print("Best Hyperparameters:\n{}".format(grid_cv.best_params_))

#df_dtr = pd.DataFrame(data=grid_cv.cv_results_)
#print(df_dtr.head())


# DecisionTreeClassifier
# {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1}

# Decision Tree Regressor Hyper Paramter Tuning took a Time: 823.2946598529816
# R-Squared::-0.040891715137972316
# Best Hyperparameters::
# {'criterion': 'mae', 'max_depth': 2, 'max_leaf_nodes': 5, 'min_samples_leaf': 20, 'min_samples_split': 10}

# from sklearn.model_selection import GridSearchCV
# rand_for_reg = RandomForestRegressor()
# 
# param_grid = {"criterion": ["mse", "mae"],
#               "min_samples_split": [10, 20],
#               "max_depth": [2, 6],
#               "min_samples_leaf": [20, 40],
#               "max_leaf_nodes": [5, 20],
#               }
# 
# grid_cv = GridSearchCV(rand_for_reg, param_grid, cv=5)
# grid_cv.fit(X_Train, y1_Train)
# 
# print("R-Squared::{}".format(grid_cv.best_score_))
# print("Best Hyperparameters::\n{}".format(grid_cv.best_params_))
# 
# df_rfr = pd.DataFrame(data=grid_cv.cv_results_)
# print(df_rfr.head())

# from sklearn.model_selection import cross_val_score
# # Checking the training model scores
# r2_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y1_Train, cv=10)
# mse_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y1_Train, cv=10, scoring='neg_mean_squared_error')
# rmse_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y1_Train, cv=10, scoring='neg_root_mean_squared_error')
# 
# print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
# print("MSE::{:.3f}".format(np.mean(mse_scores)))
# print("RMSE::{:.3f}".format(np.mean(rmse_scores)))

# # Checking the training model scores
# r2_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y1_Train, cv=10)
# mse_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y1_Train, cv=10, scoring='neg_mean_squared_error')
# rmse_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y1_Train, cv=10, scoring='neg_root_mean_squared_error')
# 
# print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
# print("MSE::{:.3f}".format(np.mean(mse_scores)))
# print("RMSE::{:.3f}".format(np.mean(rmse_scores)))

# X_Test = df_test.copy()
# X_Test['Year'] = X_Test.Date.dt.year
# X_Test['Month'] = X_Test.Date.dt.month
# X_Test['Day'] = X_Test.Date.dt.day
# X_Test.drop(columns = 'Date', inplace=True)
# X_Test['State'].fillna(EMPTY_VAL, inplace=True)
# #X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
# X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : x['Country'] if x['State'] == EMPTY_VAL else x['State'], axis=1)
# 
# X_Test.Country = le.fit_transform(X_Test.Country)
# X_Test['State'] = le.fit_transform(X_Test['State'])
# X_Test.head()

# In[ ]:


X_Test = df_test.copy()

X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%Y%m%d")
X_Test["Date"]  = X_Test["Date"].astype(int)

#X_Test.drop(columns=['ConfirmedCases', 'Fatalities'], inplace=True)
X_Test.drop(columns=['ForecastId', 'State', 'Country', 'Lat'], inplace=True)
X_Test.head()


# columns = ['Lat', 'Long', 'Date']
# X_Test = df_test[columns]
# X_Test['Date'] = X_Test.Date.dt.strftime('%m%d')
# X_Test.Date.astype('int')
# X_Test.head()

# In[ ]:


y1_best_dtr_model = grid_cv.best_estimator_
y1_pred = y1_best_dtr_model.predict(X_Test)


# In[ ]:


y1_pred = y1_pred.round(0)


# In[ ]:


y1_pred


# In[ ]:


from sklearn.metrics import r2_score
print(r2_score(y1_Train, y1_best_dtr_model.predict(X_Train))) 


# In[ ]:


y2_Train = df_train.Fatalities


# In[ ]:


start = time.time()
grid_cv = GridSearchCV(model, param_grid, cv=10)
grid_cv.fit(X_Train, y2_Train)
print (f'Decision Tree Regressor Hyper Paramter Tuning took a Time: {time.time() - start}')


# In[ ]:


print("R-Squared::{}".format(grid_cv.best_score_))
print("Best Hyperparameters::\n{}".format(grid_cv.best_params_))

#df_dtr = pd.DataFrame(data=grid_cv.cv_results_)
#print(df_dtr.head())


# Decision Tree Regressor Hyper Paramter Tuning took a Time: 609.3986613750458
# R-Squared::-0.03599184944916962
# Best Hyperparameters::
# {'criterion': 'mae', 'max_depth': 2, 'max_leaf_nodes': 5, 'min_samples_leaf': 20, 'min_samples_split': 10}

# from sklearn.model_selection import cross_val_score
# # Checking the training model scores
# #r2_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y2_Train, cv=10)
# #mse_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y2_Train, cv=10, scoring='neg_mean_squared_error')
# #rmse_scores = cross_val_score(grid_cv.best_estimator_, X_Train, y2_Train, cv=10, scoring='neg_root_mean_squared_error')
# 
# #print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
# #print("MSE::{:.3f}".format(np.mean(mse_scores)))
# #print("RMSE::{:.3f}".format(np.mean(rmse_scores)))

# In[ ]:


y2_best_dtr_model = grid_cv.best_estimator_
y2_pred = y2_best_dtr_model.predict(X_Test)


# In[ ]:


y2_pred = y2_pred.round(0)


# In[ ]:


y2_pred


# In[ ]:


df_sub = pd.read_csv(f'{PATH_WEEK1}/submission.csv')


# In[ ]:


df_sub.head()


# In[ ]:


df_sub.info()


# X_Train = df_train[['Long','Date']]
# X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%Y%m%d")
# X_Train["Date"]  = X_Train["Date"].astype(int)
# y1 = df_train[['ConfirmedCases']]
# y2 = df_train[['Fatalities']]
# X_Test = df_test[['Long','Date']]
# X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%Y%m%d")
# X_Test["Date"]  = X_Test["Date"].astype(int)
# 
# tree_reg = XGBRegressor(n_estimators=1000)
# tree_reg.fit(X_Train, y1)
# y1_pred = tree_reg.predict(X_Test)
# 
# y1_pred.round(2)
# 
# tree_reg.fit(X_Train, y2)
# y2_pred = tree_reg.predict(X_Test)
# 
# y2_pred.round(2)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X_Train)
# y1_scaled = scaler.transform(y1_Train)
# y2_scaled = scaler.transform(y2_Train)

# from sklearn.tree import DecisionTreeClassifier
# Tree_model = DecisionTreeClassifier(criterion='entropy')
# 
# Tree_model.fit(X_Train,y1_Train)
# pred1 = Tree_model.predict(X_test)
# pred1 = pd.DataFrame(pred1)
# 
# Tree_model.fit(X_Train,y2_Train)
# pred2 = Tree_model.predict(X_test)
# pred2 = pd.DataFrame(pred2)

# In[ ]:


df = pd.DataFrame({"ForecastId": df_test.ForecastId, "ConfirmedCases": y1_pred, "Fatalities": y2_pred})
df.head()


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:




