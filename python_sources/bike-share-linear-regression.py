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
# So we model this using linear regression. First on any random day, we calculate number of casual bikes used based on the features in the data-set. And then we calculate the number of registered bikes used. The summation of these two predict the values of total number of bikes used.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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


data_frame_1 = data_frame.drop(columns=["registered", "count"])
data_frame_2 = data_frame.drop(columns=["casual", "count"])


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
print("Co-relation coefficient for casual bikes")
print(data_frame_1.corr())


# In[ ]:


print("Co-relation coefficient for registered bikes")
print(data_frame_2.corr())


# Co-relation coefficient measures the strength of the relationship between two variables. For this analysis, features having co-relation coefficient less than 0.2 are dropped or not considered while modelling.(Since, their relationship strength is too weak).
# 
# Note: The number of casual bikes have four features (weekend, humidity, workingday, atemp) which have co-relational coefficient greater than 0.2, while number of registered bikes have only two feature with co-relational coefficient greater than 0.2 (atemp and humidty).

# Data_Visualisation for number of casual bikes.

# In[ ]:


plt.scatter(data_frame_1['weekend'], data_frame_1['casual'])
plt.xlabel('Weekend')
plt.ylabel('Number of casual bikes')
plt.show()


# In[ ]:


plt.scatter(data_frame_1['humidity'], data_frame_1['casual'], c='r')
plt.xlabel('Humidity')
plt.ylabel('Number of casual bikes')
plt.show()


# In[ ]:


plt.scatter(data_frame_1['atemp'], data_frame_1['casual'], c='r')
plt.xlabel('feel_temperature')
plt.ylabel('Number of casual bikes')
plt.show()


# In[ ]:


plt.scatter(data_frame_1['workingday'], data_frame_1['casual'])
plt.xlabel('working_day')
plt.ylabel('Number of casual bikes')
plt.show()


# Data_Visualisation for number of registered bikes.

# In[ ]:


plt.scatter(data_frame_2['humidity'], data_frame_2['registered'])
plt.xlabel('humidity')
plt.ylabel('Number of registered bikes')
plt.show()


# In[ ]:


plt.scatter(data_frame_2['atemp'], data_frame_2['registered'])
plt.xlabel('feel_temperature')
plt.ylabel('Number of registered bikes')
plt.show()


# In[ ]:


x1 = data_frame_1[['humidity', 'weekend', 'atemp', 'workingday']].values
y1 = data_frame_1['casual'].values

x2 = data_frame_1[['humidity', 'atemp']].values
y2 = data_frame_2['registered'].values

train_x1, test_x1, train_y1, test_y1 = train_test_split(x1, y1, test_size=0.2, random_state=4)
train_x2, test_x2, train_y2, test_y2 = train_test_split(x2, y2, test_size=0.2, random_state=4)


# In[ ]:


model_casual = linear_model.LinearRegression()
model_casual.fit(train_x1, train_y1)

model_registered = linear_model.LinearRegression()
model_registered.fit(train_x2, train_y2)


# In[ ]:


predict_casual = model_casual.predict(test_x1)
predict_registered = model_registered.predict(test_x2)


# In[ ]:


print("For number of casual bikes")
test_MSE_1 = mean_squared_error(test_y1, predict_casual)
test_MAE_1 = mean_absolute_error(test_y1, predict_casual)
r2_value_1 = r2_score(test_y1, predict_casual)
print('test_MSE', test_MSE_1, 'test_MAE', test_MAE_1, 'r2_score', r2_value_1)


# In[ ]:


plot_y_1 = np.vstack((test_y1, predict_casual))
plot_y_1 = plot_y_1.T
plot_y_1 = plot_y_1[np.argsort(plot_y_1[:, 1])]
plot_x_1 = [i for i in range(len(test_y1))]
plt.scatter(plot_x_1, plot_y_1[:, 0], c='r', label='true_value')
plt.plot(plot_x_1, plot_y_1[:, 1], c='b', label='predicted_value')
plt.show()


# Mean absolute error for the number of casual bikes is around 25, whereas r2 score is 0.44. The above plot shows the scatter plot of true values and fit line.

# In[ ]:


print("For number of registered bikes")
test_MSE_2 = mean_squared_error(test_y2, predict_registered)
test_MAE_2 = mean_absolute_error(test_y2, predict_registered)
r2_value_2 = r2_score(test_y2, predict_registered)
print('test_MSE', test_MSE_2, 'test_MAE', test_MAE_2, 'r2_score', r2_value_2)


# In[ ]:


plot_y_2 = np.vstack((test_y2, predict_registered))
plot_y_2 = plot_y_2.T
plot_y_2 = plot_y_2[np.argsort(plot_y_2[:, 1])]
plot_x_2 = [i for i in range(len(test_y2))]
plt.scatter(plot_x_2, plot_y_2[:, 0], c='r', label='true_value')
plt.plot(plot_x_2, plot_y_2[:, 1], c='b', label='predicted_value')
plt.show()


# The number of registered bikes as can be seen from the plot above isn't measured accurately using the features present in the data-set. Mean absolute error is around 99 and r2 score for the fit line is close to 0.17, which suggests a poorly fitted regresion line.
# 
# This suggests that to predict number of registered bikes, we certainly need more features and linear regression model isn't entirely collect for its prediction. 

# In[ ]:


y_true_f = np.add(test_y1, test_y2)
y_pred_f = np.add(predict_registered, predict_casual)

MAE = mean_absolute_error(y_true_f, y_pred_f)
MSE = mean_squared_error(y_true_f, y_pred_f)
print("Mean absolute error", MAE)
print("Mean Squared error", MSE)


# In[ ]:


plot_y = np.vstack((y_true_f, y_pred_f))
plot_y = plot_y.T
plot_y = plot_y[np.argsort(plot_y[:, 1])]
plot_x = [i for i in range(len(y_true_f))]
plt.scatter(plot_x, plot_y[:, 0], c='r', label='true_value')
plt.plot(plot_x, plot_y[:, 1], c='b', label='predicted_value')
plt.show()

