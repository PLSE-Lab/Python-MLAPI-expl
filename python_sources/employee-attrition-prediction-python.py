# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 01:18:39 2017

@author: Eeshan
"""

dataset = pd.read_csv('../input/HR_comma_sep.csv')
dataset = dataset[['sales','salary','satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','left']]

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,9].values

#Encoding categorical values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencod = LabelEncoder()
x[:,0] = labelencod.fit_transform(x[:,0])
x[:,1] = labelencod.fit_transform(x[:,1])

onehotty1 = OneHotEncoder(categorical_features = [0])
onehotty2 = OneHotEncoder(categorical_features = [10])
x = onehotty1.fit_transform(x).toarray()
x = onehotty2.fit_transform(x).toarray()

x = x[:,[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]]

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((14999,1)).astype(int), values = x, axis = 1)

#step 1

x_opt = x[:, :]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,1,2,3,4,6,7,9,10,11,12,13,14,15,16,17,18]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,1,2,3,4,6,7,9,11,12,13,14,15,16,17,18]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,1,2,3,4,6,7,9,12,13,14,15,16,17,18]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

z = y_pred
for i in range(len(z)):
    if z[i] <0:
        z[i] = -z[i]

z = np.around(z, decimals=0)
z = z.astype(int)
count = 0
for a in range(len(z)):
    if z[a] == y_test[a]:
        count = count + 1

fig = plt.figure()
x_data = np.arange(0,3000,1)
y_trial = y_test.astype(int)
y_predicted = z
ax = fig.add_subplot(111)
ax.scatter(x_data,y_trial, c ='b', label = 'first')
ax.scatter(x_data,y_predicted, c ='r', label = 'second')
plt.legend(loc='upper left');
plt.show()

plt.hist(y_predicted)
plt.show()

plt.hist(y_trial)
plt.show()