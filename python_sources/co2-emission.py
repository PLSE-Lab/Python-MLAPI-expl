import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/fuel-consumption/FuelConsumption.csv")

data = df[['ENGINESIZE', 'CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_CITY',
       'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
       'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

column_trans = make_column_transformer(
                (OneHotEncoder(), ["FUELTYPE"]), 
                remainder = "passthrough")

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, linreg)

X = data.drop("CO2EMISSIONS", axis = 1)

Y = data.CO2EMISSIONS

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse