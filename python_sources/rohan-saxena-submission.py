# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

test_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
unfiltered_data = pd.read_csv(test_path)
unfiltered_data.head()

#Checking for any Correlations between SalePrice and Inputs I think will work
import seaborn as sb

first_feature_selection = unfiltered_data[['LotFrontage','LotArea','MSSubClass','YearBuilt','OverallQual','OverallCond','MasVnrArea','1stFlrSF','2ndFlrSF','GarageArea','SalePrice']]
sb.pairplot(first_feature_selection)

#Looking at the pairplot, I have decided to remove MSSubClass, OverallCond, MasVnrArea and 2ndFlrSF

feature_names = ['LotFrontage','LotArea','YearBuilt','OverallQual','1stFlrSF','GarageArea']
train_x = unfiltered_data[feature_names]
train_y = unfiltered_data['SalePrice']

train_x.isnull().sum() #LotFrontage appears to have a lot of misssing values
print(train_x['LotFrontage'].mean()) #I will replace missing values with the mean of LotFrontage

train_x.fillna(value= 70.049958, inplace=True)
train_x.isnull().sum() #No more missing values, it worked :)


#Attempting to use a linear regressor
from sklearn.linear_model import LinearRegression
linreg = LinearRegression(fit_intercept=True)

first_model = linreg.fit(train_x, train_y)

#Loading in test_data
test_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
test_data = pd.read_csv(test_path)

#Setting up the x test data
x_test = test_data[feature_names]
print(x_test.isnull().sum()) #Yet again Lot Frontage has missing data, and garage Area has one missing data

#finding out the mean of each
print(x_test['LotFrontage'].describe()) #the mean is 68.580357
print(x_test['GarageArea'].describe()) #472.768861

#Filling out the missing data
x_test['LotFrontage'].fillna(value=68.580357, inplace=True)
x_test['GarageArea'].fillna(value=472.768861, inplace=True)

#Checking to see missing values are gon
x_test.isnull().sum() #Perfect!

#Making predictions
predicted_prices = first_model.predict(x_test)
print(predicted_prices)

#Preparing for Submission
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission_csv', index=False)


