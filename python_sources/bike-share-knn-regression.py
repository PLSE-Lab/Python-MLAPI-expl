#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# Total number of bikes (count) is the sum of number of registered and casual bikes. On any random day, the registered bikes used and casual bikes used would be independent of each other. Also, it can be intutively concluded that number of registered bikes and number of casual bikes used on a random day would depend differently on the features listed in the data-set. 
# 
# One such example,it maybe possible that on a working-day the number of casual bikes used would be less while number of registered bikes used would be greater. This example on an intutive level seems plausible, since people on working day are less likely to casually use bikes for some work. 
# 
# So we model this using k-nearest neigbour regressor. First on any random day, we calculate number of casual bikes used based on the features in the data-set. And then we calculate the number of registered bikes used. The summation of these two predict the values of total number of bikes used.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing


# In[ ]:


data_frame = pd.read_csv('../input/bike_share.csv')
data_frame.head()


# In the above data-set, we notice a column for week-end is missing. Apart from holidays and working-days, a third parameter for week-end is added to the data-frame. 

# In[ ]:


holidays_1 = data_frame['holiday'].values
working_days_1 = data_frame['workingday'].values

weekend_1 = []
for i in range(len(holidays_1)):
    if working_days_1[i] == 0 and holidays_1[i] == 0:
        weekend_1.append(1)
    else:
        weekend_1.append(0)
data_frame['weekend'] = weekend_1
data_frame.head()


# In[ ]:


x_casual_o = data_frame[['season', 'holiday', 'workingday', 'weather', 'atemp', 'humidity', 'windspeed', 'weekend']].values
y_casual = data_frame['casual'].values

x_registered_o = data_frame[['season', 'holiday', 'workingday', 'weather', 'atemp', 'humidity', 'windspeed', 'weekend']].values
y_registered = data_frame['registered'].values


# x_dataset is scaled, so as to avoid the domination of objective function by one of the features in the data-set having larger values.

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
x_casual_s = min_max_scaler.fit_transform(x_casual_o)
x_registered_s = min_max_scaler.fit_transform(x_registered_o)
train_x1, test_x1, train_y1, test_y1 = train_test_split(x_casual_s, y_casual, test_size=0.2, random_state=4)
train_x2, test_x2, train_y2, test_y2 = train_test_split(x_registered_s, y_registered, test_size=0.2, random_state=4)

y_true_1 = np.add(test_y1, test_y2)


# Find best k value for kNN regressor.

# In[ ]:


k_min_casual = 5
k_min_registered = 5
test_MAE_casual_array = []
test_MAE_registered_array = []
k_values = []
mae_casual = 10000
mae_registered = 10000
for k in range(5, 50):
    model_casual = KNeighborsRegressor(n_neighbors=k).fit(train_x1, train_y1)
    model_registered = KNeighborsRegressor(n_neighbors=k).fit(train_x2, train_y2)

    predict_casual = model_casual.predict(test_x1)
    predict_registered = model_registered.predict(test_x2)

    test_MAE_casual = mean_absolute_error(test_y1, predict_casual)
    test_MAE_registered = mean_absolute_error(test_y2, predict_registered)
    if test_MAE_casual < mae_casual:
        mae_casual = test_MAE_casual
        k_min_casual = k

    if test_MAE_registered < mae_registered:
        mae_registered = test_MAE_registered
        k_min_registered = k
    print("k_value",k , "MAE casual", int(test_MAE_casual),"    ", "MAE registered", int(test_MAE_registered))
    test_MAE_casual_array.append(test_MAE_casual)
    test_MAE_registered_array.append(test_MAE_registered)
    k_values.append(k)


# In[ ]:


plt.plot(k_values, test_MAE_casual_array)
plt.xlabel("k_values")
plt.ylabel("mean absolute error for number of casual bikes")
plt.show()


# In[ ]:


plt.plot(k_values, test_MAE_registered_array, 'r')
plt.xlabel("k_value")
plt.ylabel("mean absolute error for number of registered bikes")
plt.show()


# The features present in the data-set predicts the number of casual bikes very accurately, with MAE in the range of 15 to 20, but struggles to achieve same level of accuracy for the prediction of number of registered bikes, with MAE being in the range on 90 to 100.
# 
# This suggests that the features in the data-set are good to go for casual bikes number but for registered bikes we kind of need more features in the data-set.
# 
# Since we don't have any other features presently available, we move on with our process with the same data-set.

# In[ ]:


print("Best k value for the prediction of casual bikes number", k_min_casual)
print("Best k value for the prediction of registered bikes number", k_min_registered)


# In[ ]:


model_casual_final = KNeighborsRegressor(n_neighbors=k_min_casual).fit(train_x1, train_y1)
model_registered_final = KNeighborsRegressor(n_neighbors=k_min_registered).fit(train_x2, train_y2)

prediction_casual_f = model_casual_final.predict(test_x1)
prediction_registered_f = model_registered_final.predict(test_x2)

prediction_count_f = np.add(prediction_casual_f, prediction_registered_f)

print("Final Prediction")
print(prediction_count_f)


# In[ ]:


MAE = mean_absolute_error(y_true_1, prediction_count_f)
MSE = mean_squared_error(y_true_1, prediction_count_f)

print("Mean Absolute error", MAE)
print("Mean Squared Error", MSE)

