# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#For this analysis, I gathered 22 characteristics of 170 countries.
#I've applied ordinary least squares regression to determine the most effective variables on prediction of the number of confirmed cases.
#I've handled missing values imputing with their mean values.
# For standardization, I've used standard scaler.
#The variables having a significant impact on prediction of confirmed cases are population, hospital bed ratio, gdp, 
#the number of days btwn lockdown starts and the first case is seen, the number of days btwn school closure starts and the first case is seen,
#and the number of tourists.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import io

# example of imputing missing values using scikit-learn
from numpy import nan
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

from math import sqrt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets, linear_model # linear model includes {LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet}

import statsmodels.api as sm
from scipy import stats


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
covid = pd.read_csv("../input/non-pharmaceutical-data/COVID_WHOLE_DATA.csv") 

#drop unnecessary variables
covid = covid.drop('lockdown_date', axis=1)
covid = covid.drop('border_closure', axis=1)
covid = covid.drop('school closure', axis=1)
covid = covid.drop('first_case', axis=1)
covid = covid.drop('elected', axis=1)
covid = covid.drop('age', axis=1)
covid = covid.drop('male', axis=1)
covid = covid.drop('militarycareer', axis=1)
covid = covid.drop('tenure_months', axis=1)
covid = covid.drop('government', axis=1)

covid = covid.set_index("country")
covid.head()
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='mean')
# transform the dataset
covid[:] = imputer.fit_transform(covid)

#Standardization
scaler = StandardScaler()
scaler.fit(covid)
StandardScaler()
covid[:] = scaler.transform(covid)

# Labels are the values we want to predict
label_conf = np.array(covid['Confirmed'])
label_fat = np.array(covid['Fatalities'])
label_ratio = np.array(covid['Fatality_rate'])

#Remove labels from dataset
covid = covid.drop('Confirmed', axis=1) 
covid = covid.drop('Fatalities', axis=1) 
covid = covid.drop('Fatality_rate', axis=1)

X2 = sm.add_constant(covid)
est = sm.OLS(label_conf, X2)
est2 = est.fit()
print(est2.summary())

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session