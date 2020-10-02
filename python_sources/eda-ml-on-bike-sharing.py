#!/usr/bin/env python
# coding: utf-8

# # Data Fields
# 
# datetime:  
# hourly date + timestamp  
# 
# season:  
# 1 = spring, 2 = summer, 3 = fall, 4 = winter  
# 
# holiday:  
# whether the day is considered a holiday  
# 
# workingday:  
# whether the day is neither a weekend nor holiday  
# 
# weather:  
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy  
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist  
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds  
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog  
# 
# temp:  
# temperature in Celsius  
# 
# atemp:  
# "feels like" temperature in Celsius  
# 
# humidity:  
# relative humidity  
# 
# windspeed:  
# wind speed  
# 
# casual:  
# number of non-registered user rentals initiated  
# 
# registered:  
# number of registered user rentals initiated  
# 
# count:  
# number of total rentals

# # Importation datas

# In[ ]:


import calendar
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:


df.head()


# # EDA

# ## Delete 'casual' & 'registered' columns

# casual:
# number of non-registered user rentals initiated 
# 
# registered:
# number of registered user rentals initiated 

# In[ ]:


drop_lst = ['casual', 'registered']
df = df.drop(drop_lst, axis=1)
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Univariate Analysis

# ## TARGET: Count

# count:
# number of total rentals

# In[ ]:


df['count'].head()


# In[ ]:


df['count'].describe()


# In[ ]:


plt.hist(df['count']);


# In[ ]:


count_log = np.log(df['count'])
plt.hist(count_log);


# **Right skew**

# In[ ]:


count_boxcox, _ = stats.boxcox(df['count'])
count_boxcox


# In[ ]:


plt.hist(count_boxcox);


# **Normal distribution with boxcox**

# In[ ]:


df['count_log'] = count_log
df['count_boxcox'] = count_boxcox


# In[ ]:


df.head()


# **Add count_log & count_boxcox features in the dataframe.**

# ## Datetime

# datetime:
# hourly date + timestamp 

# In[ ]:


df['datetime'] = pd.to_datetime(df['datetime'])


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['dow'] = df['datetime'].dt.dayofweek
df.head()


# In[ ]:


df['month'] = df['datetime'].dt.month
df.head()


# In[ ]:


df['week'] = df['datetime'].dt.week
df.head()


# In[ ]:


df['hour'] = df['datetime'].dt.hour
df.head()


# In[ ]:


df['year'] = df['datetime'].dt.year
df.head()


# In[ ]:


df['day'] = df['datetime'].dt.day
df.head()


# In[ ]:


df = df.set_index(df['datetime'])
df.head()


# In[ ]:


df = df.drop(labels='datetime', axis=1)
df.head()


# ## seasons

# season:
# 1 = spring, 2 = summer, 3 = fall, 4 = winter 

# In[ ]:


df['season'].describe()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names = ['1', '2', '3', '4']

values = df['season'][df['year'] == 2011].value_counts()
ax[0].bar(names, values)

values = df['season'][df['year'] == 2012].value_counts()
ax[1].bar(names, values)

fig.suptitle('Seasons in 2011 & 2012');


# In[ ]:


spring_2011 = int(df['season'][df['season'] == 1][df['year'] == 2011].value_counts())
summer_2011 = int(df['season'][df['season'] == 2][df['year'] == 2011].value_counts())
fall_2011 = int(df['season'][df['season'] == 3][df['year'] == 2011].value_counts())
winter_2011 = int(df['season'][df['season'] == 4][df['year'] == 2011].value_counts())

spring_2012 = int(df['season'][df['season'] == 1][df['year'] == 2012].value_counts())
summer_2012 = int(df['season'][df['season'] == 2][df['year'] == 2012].value_counts())
fall_2012 = int(df['season'][df['season'] == 3][df['year'] == 2012].value_counts())
winter_2012 =int(df['season'][df['season'] == 4][df['year'] == 2012].value_counts())

print("Spring 2011: {}".format(spring_2011))
print("Summer 2011: {}".format(summer_2011))
print("Fall 2011: {}".format(fall_2011))
print("Winter 2011: {}".format(winter_2011))
print("-----------------------------------------")
print("Spring 2012: {}".format(spring_2012))
print("Summer 2012: {}".format(summer_2012))
print("Fall 2012: {}".format(fall_2012))
print("Winter 2012: {}".format(winter_2012))


# **No difference between 2011 & 2012.**

# ## Holiday

# holiday:
# whether the day is considered a holiday 

# In[ ]:


df['holiday'].describe()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names = ['0', '1']

values = df['holiday'][df['year'] == 2011].value_counts()
ax[0].bar(names, values)

values = df['holiday'][df['year'] == 2012].value_counts()
ax[1].bar(names, values)

fig.suptitle('Holidays in 2011 & 2012');


# In[ ]:


no_holiday_2011 = int(df['holiday'][df['holiday'] == 0][df['year'] == 2011].value_counts())
holiday_2011 = int(df['holiday'][df['holiday'] == 1][df['year'] == 2011].value_counts())
no_holiday_2012 = int(df['holiday'][df['holiday'] == 0][df['year'] == 2012].value_counts())
holiday_2012 = int(df['holiday'][df['holiday'] == 1][df['year'] == 2012].value_counts())

print("No Holidays 2011: {}".format(no_holiday_2011))
print("No Holidays 2012: {}".format(no_holiday_2012))
print("Holidays 2011: {}".format(holiday_2011))
print("Holidays 2012: {}".format(holiday_2012))
print('----------------')
total_2011 = no_holiday_2011 + holiday_2011
total_2012 = no_holiday_2012 + holiday_2012
print('No Holidays 2011: {:.0f}%'.format(no_holiday_2011 / total_2011 * 100))
print('No Holidays 2012: {:.0f}%'.format(no_holiday_2012 / total_2012 * 100))


# **Difference between Holidays and no holidays.**

# ## Working day

# workingday:
# whether the day is neither a weekend nor holiday 

# In[ ]:


df['workingday'].describe()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names = ['0', '1']

values = df['workingday'][df['year'] == 2011].value_counts()
ax[0].bar(names, values)

values = df['workingday'][df['year'] == 2012].value_counts()
ax[1].bar(names, values)

fig.suptitle('Working day in 2011 & 2012');


# In[ ]:


no_workingday_2011 = int(df['workingday'][df['workingday'] == 0][df['year'] == 2011].value_counts())
workingday_2011 = int(df['workingday'][df['workingday'] == 1][df['year'] == 2011].value_counts())
no_workingday_2012 = int(df['workingday'][df['workingday'] == 0][df['year'] == 2012].value_counts())
workingday_2012 = int(df['workingday'][df['workingday'] == 1][df['year'] == 2012].value_counts())

print("No working day 2011: {}".format(no_workingday_2011))
print("working day 2011: {}".format(workingday_2011))
print("No working day 2012: {}".format(no_workingday_2012))
print("working day 2012: {}".format(workingday_2012))
print('----------------')
total_2011 = no_workingday_2011 + workingday_2011
total_2012 = no_workingday_2012 + workingday_2012
print('No working day 2011: {:.0f}%'.format(no_workingday_2011 / total_2011 * 100))
print('No working day 2012: {:.0f}%'.format(no_workingday_2012 / total_2012 * 100))


# **Difference between workingday & no workingday.**

# ## weather

# weather:  
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy  
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist  
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds  
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog   

# In[ ]:


df['weather'].describe()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names_2011 = ['1', '2', '3']
names_2012 = ['1', '2', '3', '4']

values = df['weather'][df['year'] == 2011].value_counts()
ax[0].bar(names_2011, values)

values = df['weather'][df['year'] == 2012].value_counts()
ax[1].bar(names_2012, values)

fig.suptitle('Weather in 2011 & 2012');


# In[ ]:


weather_2011_1 = df['weather'][df['weather'] == 1][df['year'] == 2011].value_counts()
weather_2011_2 = df['weather'][df['weather'] == 2][df['year'] == 2011].value_counts()
weather_2011_3 = df['weather'][df['weather'] == 3][df['year'] == 2011].value_counts()

weather_2012_1 = df['weather'][df['weather'] == 1][df['year'] == 2012].value_counts()
weather_2012_2 = df['weather'][df['weather'] == 2][df['year'] == 2012].value_counts()
weather_2012_3 = df['weather'][df['weather'] == 3][df['year'] == 2012].value_counts()
weather_2012_4 = df['weather'][df['weather'] == 4][df['year'] == 2012].value_counts()

print('weather_1 in 2011: {}'.format(int(weather_2011_1)))
print('weather_2 in 2011: {}'.format(int(weather_2011_2)))
print('weather_3 in 2011: {}'.format(int(weather_2011_3)))
print('--------------')
print('weather_1 in 2012: {}'.format(int(weather_2012_1)))
print('weather_2 in 2012: {}'.format(int(weather_2012_2)))
print('weather_3 in 2012: {}'.format(int(weather_2012_3)))
print('weather_4 in 2012: {}'.format(int(weather_2012_4)))
print('---------------')
total_2011 = int(weather_2011_1) + int(weather_2011_2) + int(weather_2011_3)
total_2012 = int(weather_2012_1) + int(weather_2012_2) + int(weather_2012_3) + int(weather_2012_4)
print('weather_1 in 2011: {:.0f}%'.format(int(weather_2011_1) / int(total_2011) * 100))
print('weather_2 in 2011: {:.0f}%'.format(int(weather_2011_2) / int(total_2011) * 100))
print('weather_3 in 2011: {:.0f}%'.format(int(weather_2011_3) / int(total_2011) * 100))
print('--------------')
print('weather_1 in 2012: {:.0f}%'.format(int(weather_2012_1) / int(total_2012) * 100))
print('weather_2 in 2012: {:.0f}%'.format(int(weather_2012_2) / int(total_2012) * 100))
print('weather_3 in 2012: {:.0f}%'.format(int(weather_2012_3) / int(total_2012) * 100))
print('weather_4 in 2012: {:.0f}%'.format(int(weather_2012_4) / int(total_2012) * 100))


# **Differences between weather_1 to weather_4 for 2011& 2012.**

# ## Temp

# temp:
# temperature in Celsius

# In[ ]:


df['temp'].describe()


# **No outliers.**

# In[ ]:


plt.hist(df['temp'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['temp'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');


# **Normal distribution for temp in 2011 & 2012.**

# ## Atemp

# atemp:
# "feels like" temperature in Celsius 

# In[ ]:


df['atemp'].describe()


# **No outliers.**

# In[ ]:


plt.hist(df['atemp'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['atemp'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');


# **Normal distribution for atemp in 2011 & 2012.**

# ## Humidity

# humidity:
# relative humidity 

# In[ ]:


df['humidity'].describe()


# **No outliers.**

# In[ ]:


plt.hist(df['humidity'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['humidity'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');


# **Right skew for 2011 & 2012.**

# ## windspeed

# windspeed:
# wind speed

# In[ ]:


df['windspeed'].describe()


# **No outliers.**

# In[ ]:


plt.hist(df['windspeed'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['windspeed'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');


# **Left skew for 2011 & 2012.**

# ## DayOfWeek

# Monday = 0  
# Sunday = 6

# In[ ]:


df['dow'].describe()


# In[ ]:


plt.hist(df['dow'][df['year'] == 2011], alpha=0.5, label='2011', bins=7)
plt.hist(df['dow'][df['year'] == 2012], alpha=0.5, label='2012', bins=7)

plt.legend(loc='upper right');


# **Equal distribution for 2011 & 2012**

# ## Month

# January = 0  
# December = 12

# In[ ]:


df['month'].describe()


# In[ ]:


plt.hist(df['month'][df['year'] == 2011], alpha=0.5, label='2011', bins=12)
plt.hist(df['month'][df['year'] == 2012], alpha=0.5, label='2012', bins=12)

plt.legend(loc='upper right');


# **Equal distribution for 2011 & 2012.**

# ## Week

# 1 to 52

# In[ ]:


df['week'].describe()


# In[ ]:


plt.hist(df['week'][df['year'] == 2011], alpha=0.5, label='2011', bins=52)
plt.hist(df['week'][df['year'] == 2012], alpha=0.5, label='2012', bins=52)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.legend(loc='upper right');


# **We have the 20th first days of each month for 2011 & 2012.**

# ## Hour

# 0 to 23

# In[ ]:


df['hour'].describe()


# In[ ]:


plt.hist(df['hour'][df['year'] == 2011], alpha=0.5, label='2011', bins=24)
plt.hist(df['hour'][df['year'] == 2012], alpha=0.5, label='2012', bins=24)
plt.legend(loc='upper right');


# **Equal distribution for 2011 & 2012.**

# ## Day

# 1 to 31

# In[ ]:


df['day'].describe()


# In[ ]:


plt.hist(df['day'][df['year'] == 2011], alpha=0.5, label='2011', bins=31)
plt.hist(df['day'][df['year'] == 2012], alpha=0.5, label='2012', bins=31)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.legend(loc='upper right');


# **We have the 20th first days for each month in 2011 & 2012.**

# ## Year

# 2011-2012

# In[ ]:


df['year'].describe()


# In[ ]:


names = ['2011', '2012']
values = df['year'].value_counts()
plt.bar(names, values);


# In[ ]:


count_2011 = df['year'][df['year'] == 2011].count()
count_2012 = df['year'][df['year'] == 2012].count()

print('2011: {}'.format(count_2011))
print('2012: {}'.format(count_2012))


# **Equal distribution for 2011 & 2012.**

# # Multivariate Analysis

# ## Heatmap

# In[ ]:


cor_mat = df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig = plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True);


# ## Count/temp

# In[ ]:


sns.pointplot(x=df['temp'], y=df['count'])
fig = plt.gcf()
fig.set_size_inches(30,12);


# In[ ]:


from scipy import stats
_, _, r_value, _, _ = stats.linregress(df['count'], df['temp'])
r_square = r_value ** 2
r_square.round(2)


# ## Count/Atemp

# In[ ]:


sns.pointplot(x=df['atemp'], y=df['count'])
fig = plt.gcf()
fig.set_size_inches(30,12);


# In[ ]:


_, _, r_value, _, _ = stats.linregress(df['count'], df['atemp'])
r_square = r_value ** 2
r_square.round(2)


# ## Count/Hour

# In[ ]:


sns.pointplot(x=df['hour'], y=df['count'])
fig = plt.gcf()
fig.set_size_inches(30,12);


# **2 High values: 8h et 17h**

# ## Temp/Atemp

# In[ ]:


sns.pointplot(x=df['temp'], y=df['atemp'])
fig = plt.gcf()
fig.set_size_inches(30,12);


# In[ ]:


_, _, r_value, _, _ = stats.linregress(df['temp'], df['atemp'])
r_square = r_value ** 2
r_square.round(2)


# **High correlation, we can drop atemp column.**

# ## Delete 'Atemp' column

# In[ ]:


df = df.drop(labels='atemp', axis=1)


# In[ ]:


df.head()


# ## Delet count_log & count_boxcox columns

# In[ ]:


df = df.drop(labels='count_log', axis=1)


# In[ ]:


df = df.drop(labels='count_boxcox', axis=1)


# In[ ]:


df.head()


# ## Sparse weather column

# In[ ]:


df = pd.get_dummies(df, columns=['weather'])
df.head()


# In[ ]:


df = df.drop(labels='weather_4', axis=1)
df.head()


# ## New features: Temp by weather

# In[ ]:


df['temp_weath_1'] = df['temp'] * df['weather_1']
df['temp_weath_2'] = df['temp'] * df['weather_2']
df['temp_weath_3'] = df['temp'] * df['weather_3']


# In[ ]:


df['temp_weath_1'] = df['temp_weath_1'].astype(int)
df['temp_weath_2'] = df['temp_weath_2'].astype(int)
df['temp_weath_3'] = df['temp_weath_3'].astype(int)


# In[ ]:


df.head()


# In[ ]:


X = df.loc[:, df.columns != 'count']
y = np.log(df['count'])


# In[ ]:


X.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, QuantileTransformer, RobustScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

from xgboost import XGBRegressor


# In[ ]:


pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('LASSO', Lasso(random_state=42))])))
pipelines.append(('ScaledRID', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('RID', Ridge(random_state=42))])))
pipelines.append(('ScaledKNN', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor(n_neighbors=2))])))
pipelines.append(('ScaledCART', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor(random_state=42))])))
pipelines.append(('ScaledGBM', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor(random_state=42))])))
pipelines.append(('ScaledRFR', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('RFR', RandomForestRegressor(random_state=42))])))
pipelines.append(('ScaledSVR', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('SVR', SVR(kernel='linear'))])))
pipelines.append(('ScaledXGBR', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('XGBR', XGBRegressor(random_state=42))])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(random_state=42)
    cv_results = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error')
    results.append(np.sqrt(cv_results))
    names.append(name)
    msg = "{}: {} ({})".format(name, cv_results.mean(), cv_results.std())
    print(msg)


# ## Best score:  
# ScaledXGBR: 0.014747506232883416 (0.0007606201073795001)

# # TEST SET

# In[ ]:


df_test = pd.read_csv("../input/test.csv")


# In[ ]:


df_test['datetime'] = pd.to_datetime(df_test['datetime'])


# In[ ]:


df_test['dow'] = df_test['datetime'].dt.dayofweek


# In[ ]:


df_test['month'] = df_test['datetime'].dt.month


# In[ ]:


df_test['week'] = df_test['datetime'].dt.week


# In[ ]:


df_test['hour'] = df_test['datetime'].dt.hour


# In[ ]:


df_test['year'] = df_test['datetime'].dt.year


# In[ ]:


df_test['day'] = df_test['datetime'].dt.day


# In[ ]:


df_test = df_test.set_index(df_test['datetime'])


# In[ ]:


df_test = df_test.drop(labels='datetime', axis=1)


# In[ ]:


df_test = df_test.drop(labels='atemp', axis=1)


# In[ ]:


df_test = pd.get_dummies(df_test, columns=['weather'])


# In[ ]:


df_test = df_test.drop(labels='weather_4', axis=1)


# In[ ]:


df_test['temp_weath_1'] = df_test['temp'] * df_test['weather_1']
df_test['temp_weath_2'] = df_test['temp'] * df_test['weather_2']
df_test['temp_weath_3'] = df_test['temp'] * df_test['weather_3']


# In[ ]:


df_test['temp_weath_1'] = df_test['temp_weath_1'].astype(int)
df_test['temp_weath_2'] = df_test['temp_weath_2'].astype(int)
df_test['temp_weath_3'] = df_test['temp_weath_3'].astype(int)


# In[ ]:


standardscaler = StandardScaler()
model = XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=7, min_child_weight=4, subsample=0.7, random_state=42)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.predict(df_test)


# In[ ]:


pipe = Pipeline([('poly', PolynomialFeatures()), ('StandardScaler', standardscaler), ('XGBR', model)])
pipe.fit(X_train, y_train)
y_pred = np.exp(pipe.predict(df_test))
y_pred


# In[ ]:


df_test['count'] = y_pred


# In[ ]:


df_test.head()


# In[ ]:


df_test[['count']].to_csv('submission.csv', index=True)


# In[ ]:


df_test[['count']].head()


# In[ ]:




