# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
#selecting data only for United States
data=data[data['Country']=='United States']

#Calculating the missing values
null=data.isnull().sum()
print("Null values in the data ")
print(null)
#Dropping the na ormissing values as they only account for 4% of all values
data=data.dropna()

#Re-verifying the missing values after dropping them
null_After=data.isnull().sum()
print("Null values in the data ")
print(null_After)
#data types of the dataframe
data_type=data.dtypes
print("Data type of the dataframe columns are ")
print(data_type)

data[['dt']]=pd.to_datetime(data['dt'])
# data before 1870 seems to be noisy hence selecting from 1870
data_new=data[data['dt']>'1870']

#grouping the data in terms of year and finding the mean of the average temp
mean_temp_inc_over_year=data_new.groupby(data_new.dt.dt.year).mean()
print("Average temperature per year in United States")
print(mean_temp_inc_over_year)


mean_temp_inc_over_year.reset_index(level=0, inplace=True)

x=mean_temp_inc_over_year['dt']
y=mean_temp_inc_over_year['AverageTemperature']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

#Training the data and predicting the value for test data
lm=smf.OLS(endog=y_train,exog=X_train).fit()
predict=lm.predict(X_test)
print("Predicted values ")
print(predict)
print("Actual values")
print(y_test)
#RMSE value
rms = sqrt(mean_squared_error(y_test, predict))
print(rms)

#Summary of regression
print("Summary of the regrssion model")
print(lm.summary())

