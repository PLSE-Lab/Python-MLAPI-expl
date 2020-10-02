# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:19:12 2018

@author: Haripriya
"""

#importing data and basic data exploration

#loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from scipy.stats import binned_statistic
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from numpy import inf


#reading in train and test data
train = pd.read_csv("train.csv")
train.shape
test = pd.read_csv("test.csv")
test.shape

#combining both train and test data for further understanding
test['registered'] = 0
test['casual'] = 0
test['count'] = 0
combined_data = pd.concat([train, test], sort = False)
combined_data.info()

#checking for missing values
combined_data.isnull().values.ravel().sum()

#understanding the data distribution of numerical variables
num_var = combined_data.select_dtypes(include = [np.number]).columns.tolist()
num_var

plt.figure(1)
plt.subplot(421)
plt.hist(combined_data['season'])
plt.title('season')

plt.subplot(422)
plt.hist(combined_data['holiday'])
plt.title('holiday')

plt.subplot(423)
plt.hist(combined_data['workingday'])
plt.title('workingday')

plt.subplot(424)
plt.hist(combined_data['weather'])
plt.title('weather')

plt.subplot(425)
plt.hist(combined_data['temp'])
plt.title('temp')

plt.subplot(426)
plt.hist(combined_data['atemp'])
plt.title('atemp')

plt.subplot(427)
plt.hist(combined_data['humidity'])
plt.title('humidity')

plt.subplot(428)
plt.hist(combined_data['windspeed'])
plt.title('windspeed')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1,
                    wspace=0.35)
plt.show()

#converting discrete variables into factors
combined_data['season'] = combined_data['season'].astype(object)
combined_data['weather'] = combined_data['weather'].astype(object)
combined_data['holiday'] = combined_data['holiday'].astype(object)
combined_data['workingday'] = combined_data['workingday'].astype(object)

#multivariate analysis
type(train.datetime)
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour.astype(int)
test['datetime'] = pd.to_datetime(test['datetime'])

combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
combined_data['hour'] = combined_data['datetime'].dt.hour

combined_data['hour'] = combined_data['hour'].astype(object)

plt.boxplot(train['hour'], train['count'])


#feature engineering
#hour bins
#generating dt
combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
combined_data['hour'] = combined_data['datetime'].dt.hour
combined_data['hour'] = combined_data['hour'].astype(object)
combined_data['hour']
combined_data['datetime']

y = train['registered']
x1 = train[['hour']]
reg = tree.DecisionTreeRegressor(max_depth = 3)
reg = reg.fit(x1, y)

dotfile = open("G://New Projects//Kaggle//Bike Sharing Demand//reg_hr.dot", 'w')
tree.export_graphviz(reg, out_file = dotfile)
dotfile.close()

#binning hour for registered users
combined_data['dp_reg']=0
combined_data.dp_reg[combined_data['hour']<=7]=1
combined_data.dp_reg[combined_data['hour']<=6]=2
combined_data.dp_reg[combined_data['hour']<=21]=3
combined_data.dp_reg[combined_data['hour']<=2]=4
combined_data.dp_reg[combined_data['hour']<=17]=5
combined_data.dp_reg[combined_data['hour']<=23]=6

#building dt for casual vs hour
y1 = train['casual']
x1 = train[['hour']]
reg_cas = tree.DecisionTreeRegressor(max_depth = 3)
reg_cas = reg_cas.fit(x1, y1)

dotfile = open("G://New Projects//Kaggle//Bike Sharing Demand//reg_ca.dot", 'w')
tree.export_graphviz(reg_cas, out_file = dotfile)
dotfile.close()

#binning hour for registered users
combined_data['dp_cas']=0
combined_data.dp_cas[combined_data['hour']<=10]=1
combined_data.dp_cas[combined_data['hour']<=8]=2
combined_data.dp_cas[combined_data['hour']<=20]=3
combined_data.dp_cas[combined_data['hour']<=7]=4
combined_data.dp_cas[combined_data['hour']<=9]=5
combined_data.dp_cas[combined_data['hour']<=11]=6
combined_data.dp_cas[combined_data['hour']<=22]=7

#temp bins
#generating dt
combined_data['temp'] = combined_data['temp'].astype(object)

y2 = train['registered']
x2 = train[['temp']]
reg_temp1 = tree.DecisionTreeRegressor(max_depth = 3)
reg_temp1 = reg.fit(x2, y2)

dotfile = open("G://New Projects//Kaggle//Bike Sharing Demand//temp_reg.dot", 'w')
tree.export_graphviz(reg_temp1, out_file = dotfile)
dotfile.close()

#binning hour for registered users
combined_data['temp_reg']=0
combined_data.temp_reg[combined_data['temp']<=23]=1
combined_data.temp_reg[combined_data['temp']<=13]=2
combined_data.temp_reg[combined_data['temp']<=30]=3
combined_data.temp_reg[combined_data['temp']<=11]=4
combined_data.temp_reg[combined_data['temp']<=19]=5
combined_data.temp_reg[combined_data['temp']<=28]=6
combined_data.temp_reg[combined_data['temp']<=31]=7

#binning for casual users
y3 = train['casual']
x2 = train[['temp']]
reg_temp2 = tree.DecisionTreeRegressor(max_depth = 3)
reg_temp2 = reg.fit(x2, y3)

dotfile = open("G://New Projects//Kaggle//Bike Sharing Demand//temp_cas.dot", 'w')
tree.export_graphviz(reg_temp2, out_file = dotfile)
dotfile.close()

combined_data['temp_cas']=0
combined_data.temp_cas[combined_data['temp']<=23]=1
combined_data.temp_cas[combined_data['temp']<=13]=2
combined_data.temp_cas[combined_data['temp']<=30]=3
combined_data.temp_cas[combined_data['temp']<=11]=4
combined_data.temp_cas[combined_data['temp']<=19]=5
combined_data.temp_cas[combined_data['temp']<=28]=6
combined_data.temp_cas[combined_data['temp']<=31]=7

#creating year bins quarterly for two years
combined_data['year'] = combined_data['datetime'].dt.year
combined_data['year'] = combined_data['year'].astype(object)

combined_data['month'] = combined_data['datetime'].dt.month
combined_data['month'] = combined_data['month'].astype(object)

combined_data['year_part'] = 0
combined_data.year_part[combined_data['year']=='2011']=1
combined_data.year_part[(combined_data['year'] =='2011') & (combined_data['month']>3)]=2
combined_data.year_part[(combined_data['year'] =='2011') & (combined_data['month']>6)]=3
combined_data.year_part[(combined_data['year']=='2011') & (combined_data['month']>9)]=4
combined_data.year_part[combined_data['year']=='2012']=5
combined_data.year_part[(combined_data['year']=='2012') & (combined_data['month']>3)]=6
combined_data.year_part[(combined_data['year']=='2012') & (combined_data['month']>6)]=7
combined_data.year_part[(combined_data['year']=='2012') & (combined_data['month']>9)]=8

#creating daytype with categories weekend, weekday, holiday
combined_data['day_type']=""
combined_data.day_type[(combined_data['holiday']==0) & (combined_data['workingday']==0)]="weekend"
combined_data.day_type[combined_data['holiday']==1]="holiday"
combined_data.day_type[(combined_data['holiday']==0) & (combined_data['workingday']==1)]="working day"

#separate variable for weekend
combined_data['day'] = combined_data['datetime'].dt.day
combined_data['day'] = combined_data['day'].astype(object)

combined_data['weekend']=0
combined_data.weekend[(combined_data['day'] == "Sunday") | (combined_data['day']=="Saturday") ]=1

#model building
combined_data.info()

#splitting train and test sets 
train_len = len(train)
train_len

train = combined_data[:train_len]
train.shape

test = combined_data[train_len:]
test.shape
test.info()

#dropping target variables from test data
test = test.drop('casual', axis = 1)
test = test.drop('registered', axis = 1)
test = test.drop('count', axis = 1)
test.info()

le = preprocessing.LabelEncoder()
le_train = train.apply(le.fit_transform)
le_test = test.apply(le.fit_transform)

#predicting the log of registered users.
y1 = np.log(le_train.registered)+1

y1[np.isneginf(y1)] = 0
y1[np.isposinf(y1)] = y1.max()
y1[np.isnan(y1)] = 0

x1 = le_train.loc[:,('hour','workingday','day','holiday','day_type','temp_reg',
           'humidity','atemp','windspeed','season','weather','dp_reg',
           'weekend','year','year_part')]

test_x1 = le_test.loc[:,('hour','workingday','day','holiday','day_type','temp_reg',
           'humidity','atemp','windspeed','season','weather','dp_reg',
           'weekend','year','year_part')]


rf1 = RandomForestRegressor(n_estimators = 500)
fit1 = rf1.fit(x1, y1)
pred1 = rf1.predict(test_x1)
test_x1['logreg'] = pred1
test_x1.info()

#predicting the log of casual users.
y2 = np.log(le_train.casual)+1

y2[np.isneginf(y2)] = 0
y2[np.isposinf(y2)] = y1.max()
y2[np.isnan(y2)] = 0

x2 = le_train.loc[:,('hour','day_type','day','humidity','atemp','temp_cas','windspeed',
                  'season','weather','holiday','workingday','dp_cas',
                  'weekend','year','year_part')]

test_x2 = le_test.loc[:,('hour','day_type','day','humidity','atemp','temp_cas','windspeed',
                  'season','weather','holiday','workingday','dp_cas',
                  'weekend','year','year_part')]


rf2 = RandomForestRegressor(n_estimators = 500)
fit2 = rf2.fit(x2, y2)
pred2 = rf2.predict(test_x2)
test_x2['logcas'] = pred2
test_x2.info()

#Re-transforming the predicted variables and then writing the output of count 
#to the file submit.csv

test['registered'] = np.exp(test_x1['logreg']) - 1
test['casual'] = np.exp(test_x2['logcas']) - 1
test['count'] = test['casual'] + test['registered']

train.datetime
submission = pd.DataFrame(test, columns = ['datetime','count'])
submission.to_csv("submit.csv", header = True)
