# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

filename = "/kaggle/input/covid19-corona-virus-india-dataset/nation_level_daily.csv"
df = pd.read_csv(filename)
cases = df['totalconfirmed']

days_since_first_case = np.array([i for i in range(len(cases.index))]).reshape(-1, 1)
india_cases = np.array(cases).reshape(-1, 1)

#Preparing indexes to predict next 15 days
days_in_future = 3
future_forecast = np.array([i for i in range(len(cases.index) + days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-3]


#Splitting data into train and test to evaluate our model
X_train, X_test, y_train, y_test = train_test_split(days_since_first_case
                                                    , india_cases
                                                    , test_size= 10
                                                    , shuffle=False
                                                    , random_state = 42)


root_mean_squared_error = 10000
degree = 0
# for i in range(101):
#     # Transform our cases data for polynomial regression
#     poly = PolynomialFeatures(degree=i)
#     poly_X_train = poly.fit_transform(X_train)
#     poly_X_test = poly.fit_transform(X_test)
#     poly_future_forcast = poly.fit_transform(future_forcast)
#
#     # polynomial regression cases
#     linear_model = LinearRegression(normalize=True, fit_intercept=False)
#     linear_model.fit(poly_X_train, y_train)
#     test_linear_pred = linear_model.predict(poly_X_test)
#     linear_pred = linear_model.predict(poly_future_forcast)
#
#     # evaluating with RMSE
#     rm = np.sqrt(mean_squared_error(y_test, test_linear_pred))
#     if(rm<rmse):
#         rmse = rm
#         degree = i
#     if(i==100):
#         print('the best mae is:',round(rmse,2))
#         print('the best degree for cases is:',degree)



# Transform our cases data for polynomial regression
poly = PolynomialFeatures(degree=8)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)
poly_future_forecast = poly.fit_transform(future_forecast)

# polynomial regression cases
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train, y_train)
test_linear_pred = linear_model.predict(poly_X_test)
linear_pred = linear_model.predict(poly_future_forecast)

# evaluating with RMSE
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, test_linear_pred)))

# Figure 1
plt.figure(figsize=(12,7))

plt.plot(y_test, label = "Real cases")
plt.plot(test_linear_pred, label = "Predicted")
plt.title("Predicted vs Real cases", size = 20)
plt.xlabel('Days', size = 18)
plt.ylabel('Cases', size = 18)
plt.xticks(size=18)
plt.yticks(size=18)

# defining legend config
plt.legend(loc = "upper left"
           , frameon = True
           , ncol = 2
           , fancybox = True
           , framealpha = 0.95
           , shadow = True
           , borderpad = 1
           , prop={'size': 18})

plt.show()

# Figure 2
plt.figure(figsize=(12, 7))

plt.plot(adjusted_dates
         , india_cases
         , label = "Real cases")

plt.plot(future_forecast
         , linear_pred
         , label = "Polynomial Regression Predictions"
         , linestyle='dashed'
         , color='orange')

plt.title('Cases in India over the time: Predicting Next 3 days', size=18)
plt.xlabel('Days Since 30/01/20', size=18)
plt.ylabel('Cases', size=18)
plt.xticks(size=18)
plt.yticks(size=18)

plt.axvline(len(X_train), color='black'
            , linestyle="--"
            , linewidth=1)

plt.text(18, 5000
         , "model training"
         , size = 15
         , color = "black")

plt.text((len(X_train)+0.2), 15000
         , "prediction"
         , size = 15
         , color = "black")

# defining legend config
plt.legend(loc = "upper left"
           , frameon = True
           , ncol = 2
           , fancybox = True
           , framealpha = 0.95
           , shadow = True
           , borderpad = 1
           , prop={'size': 15})

plt.show()