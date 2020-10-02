# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:07:07 2019

@author: gauravmalik
"""
#Loading all the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#Importing the data 
df = pd.read_csv("../input/bike_sharing_daily.csv")

#Taking a quick look at the data 
df.head()
#Checking the dimension of the data
df.shape
#Looking at all the unique values in the column
df.nunique()

df.describe()
#Checking in there are any null values in the columns and their data types
df.info()

#Converting the dteday column which is of type object to datetime64
df['dteday']=df['dteday'].astype('datetime64')

#To plot a timeseries plot using date to check the dependent variables
#Start of plotting timeseries plot
#New dataframe to include only date and dependent variables
timeSeriesDf = df[['dteday', 'casual', 'registered', 'cnt']]
#Setting date as index
timeSeriesDf.set_index('dteday', inplace = True)
#Timeseries plot
timeSeriesDf.asfreq('W').plot(figsize = (20,10), linewidth = 2, fontsize = 20)
plt.xlabel('731 days', fontsize = 20)
plt.title('Timeseries: 2011-2012')
plt.show()
#End of plotting timeseries plot

#As the instant columns in just the index removing them won't effect
df.drop(columns=['instant'], inplace = True)
#Setting date as index in original dataframe for a new dataframe which might be useful in future
dfDate = df.set_index('dteday')

#Taking the df.columns result and storing them in a list
columns = df.columns.tolist()
#Specifying all the categorical columns
categorical_columns = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                       'weathersit']
#removing all the categorical columns from columns list leaving all the numerical variables
numerical_columns = list(set(columns) - set(categorical_columns))
numerical_columns.remove('dteday')
numerical_columns.remove('cnt')
numerical_columns.remove('registered')
numerical_columns.remove('casual')

#Plotting all the numerical columns data to check the distribution
for i in range(len(numerical_columns)):
    column = numerical_columns[i]
    if(column != 'dteday'):
        print (column)
        sns.distplot(df[column])
        plt.title(column)
        plt.show()
        
       
#As the numerical columns was scaled using min max scaler we will remove all the outliers
#beacuse min max is sensitive to the outliers
#Boxplot before removing the ouliers
for i in range(len(numerical_columns)):
    column = numerical_columns[i]
    if(column != 'dteday'):
        print (column)
        sns.boxplot(df[column])
        plt.title(column)
        plt.show()
 #declaring y just to know how many outliers were updated
y=0
#checking for the outliers and replacing it with median
for i in range(len(numerical_columns)):
    column = numerical_columns[i]
    print(column)
    q1=df[column].quantile(0.25)
    q3=df[column].quantile(0.75)
    lowerFence = q1 - (1.5*(q3-q1))
    upperFence = q3 + (1.5*(q3-q1))
    df_temp = df[column]
    for i in range(len(df[column])):
        if(lowerFence> df_temp[i] or df_temp[i] > upperFence):
            df_temp[i] = df_temp.median()
            y+=1
    print(y)
#Bolplot after removing the outliers
for i in range(len(numerical_columns)):
    column = numerical_columns[i]
    if(column != 'dteday'):
        print (column)
        sns.boxplot(df[column])
        plt.title(column)
        plt.show()
        
##They have used min max scaler
 #Checking the correlation of the features
#All Numerical variables       
corr = df[numerical_columns].corr()
sns.heatmap(corr, annot = True)
plt.title('Correlation of Numerical Features')
plt.show()
df.drop(columns = ['atemp'], inplace = True)
numerical_columns.remove('atemp')

#All categorical variables
corr = df[categorical_columns].corr()
sns.heatmap(corr, annot = True)
plt.title('Correlation of Categorical Features')
plt.show()
df.drop(columns = ['mnth'], inplace = True)
categorical_columns.remove('mnth')
#All the variables 
listll = numerical_columns + categorical_columns
corr = df[listll].corr()
sns.heatmap(corr, annot = True)
plt.title('Correlation of all variables')
plt.show()
df.drop(columns = ['hum'], inplace = True)
df.drop(columns = ['dteday'], inplace = True)

#
X=df.iloc[:, :-3].values
y=df.iloc[:, -3:].values
#We will predict casual and registered seperately
#Now all we need to do is seperate the data for casual and registered
#use one hot encoder
onehotencoder = OneHotEncoder(categorical_features= [0, 3, 5])
X = onehotencoder.fit_transform(X).toarray()
#To avoid dummy variable trap
X = np.delete(X, [3, 10, 13], axis = 1)

#Splitting the training and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Linear Regression

lineReg = LinearRegression()
lineReg.fit(X_train, y_train[:,0])

y_casual_pred = lineReg.predict(X_test)

lineR2 = r2_score(y_test[:, 0], y_casual_pred)
linearMSE = mean_squared_error(y_test[:, 0], y_casual_pred)


plt.plot(y_test[:, 0], y_casual_pred, "*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.title('Plot for Linear Regressor(Casual)')
plt.show()

#Fitting SVM

svReg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svReg.fit(X_train, y_train[:,0])

y_casual_svr = svReg.predict(X_test)
svrR2 = r2_score(y_test[:, 0], y_casual_svr)
svrMSE = mean_squared_error(y_test[:, 0], y_casual_svr)

plt.plot(y_test[:, 0], y_casual_svr, "*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.title('Plot for Support Vector Machine(Casual)')
plt.show()

#Fitting Random Forest Regressor

rFReg = RandomForestRegressor(n_estimators= 150, max_depth = 200, random_state= 0)
rFReg.fit(X_train, y_train[:, 0])

y_casual_rf = rFReg.predict(X_test)
rfR2 = r2_score(y_test[:, 0], y_casual_rf)
rfMSE = mean_squared_error(y_test[:, 0], y_casual_rf)

plt.plot(y_test[:, 0], y_casual_rf, "*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.title('Plot for Random Forest(Casual)')
plt.show()

#As Random Forest seems to be the best fit, will fit for regular too
rFReg.fit(X_train, y_train[:, 1])

y_regular_rf = rFReg.predict(X_test)
rfRegularR2 = r2_score(y_test[:, 1], y_regular_rf)
rfRegularMSE = mean_squared_error(y_test[:, 1], y_casual_rf)
rfRegularMEA = mean_absolute_error(y_test[:, 1], y_casual_rf)
rfRegularRMSE = np.sqrt(rfRegularMSE)

plt.plot(y_test[:,1], y_regular_rf,'.')
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.title('Plot for Random Forest(Registered)')
plt.show()

#Now adding both regular and casual and looking at the plot of cnt
y_cnt_rf = y_regular_rf + y_casual_rf

plt.plot(y_test[:, 2], y_cnt_rf, '*')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Plot of "cnt"')
plt.show
#fitting xgboost
#Tried to fit xgboost but seems unstable as it clears the memory whenever run
#Might work on lower python version(current python version 3.7)
#import xgboost as xgb
#dtrain=xgb.DMatrix(X_train, y_train[:, 0])
#our_params = {'eta':0.1, 'seed':0, 'subsample':0.8, 'colsample_bytree':0.8}
#final_gb = xgb.train(our_params, dtrain)
#dtest  = xgb.DMatrix(X_test)
#y_casual_xgb=final_gb.predict(dtest)
#rfXgb = r2_score(y_test[:, 0], y_casual_xgb)