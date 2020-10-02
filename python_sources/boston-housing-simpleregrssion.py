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
dataset = pd.read_csv('/kaggle/input/boston-housing-dataset/HousingData.csv', header=None, skiprows=1)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,13].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0,)
imputer = imputer.fit(X[:,0:13])
X = imputer.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

var = regressor.predict(X_test)

#With backward elimination

X_opt = X_train[:, [0,1,3,5,6,7,9,10,11,12]]

regressor_opt = LinearRegression()
regressor_opt.fit(X_opt,y_train)

var_opt = regressor_opt.predict(X_test[:, [0,1,3,5,6,7,9,10,11,12]])

