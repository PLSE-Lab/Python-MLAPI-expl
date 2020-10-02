# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/kc_house_data.csv")
df
# Any results you write to the current directory are saved as output.
df.info()
X1 = df.drop('price',axis=1)
y = df.iloc[:,2]
X1.info()
print(X1['grade'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1,y,test_size=0.2,random_state=0)

type(df['date'])
X_train = X_train.drop('date',axis=1)
X_test = X_test.drop('date',axis=1)
X_train.info()
X_test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_test
y_pred

import statsmodels.formula.api as smf
from sklearn import metrics
import seaborn as sns
X_train.shape
X_test.shape

#%matplotlib inline
df = df.drop('date',axis=1)
sns.pairplot(df, x_vars=['id','bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], y_vars='price', size=7, aspect=0.7)
sns.pairplot(df, x_vars=['id','bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], y_vars='price', size=7, aspect=0.7, kind='reg')

regressor.score(X_train,y_train)    #0.700603884821694
regressor.score(X_test,y_test)      #0.6951598737989825

feature_cols=['id','bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
list(zip(feature_cols, regressor.coef_))

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

lm1 = smf.ols(formula='price ~ id+bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+zipcode+lat+long+sqft_living15+sqft_lot15', data=df).fit()
lm1.params
lm1.rsquared        #0.6998463505666735
lm1.summary()

lm1 = smf.ols(formula='price ~ id+bedrooms+bathrooms+sqft_living+sqft_lot+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+zipcode+lat+long+sqft_living15+sqft_lot15', data=df).fit()
lm1.params
lm1.rsquared       #0.6997971459089627 
lm1.summary()

X_train1 = X_train.drop('floors',axis=1)
X_test1 = X_test.drop('floors',axis=1)
X_train1.info()
X_test1
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train1,y_train)

y_pred1 = regressor1.predict(X_test1)

regressor1.score(X_train1,y_train)    #0.7005445716199037
regressor1.score(X_test1,y_test)        #0.6951608024139833
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print(metrics.mean_absolute_error(y_test, y_pred1))