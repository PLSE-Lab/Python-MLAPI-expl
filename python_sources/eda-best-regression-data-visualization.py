#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:





# In[ ]:


import pandas as pd
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# Agent, Company, country  have some missing data, 

# In[ ]:


print("Nan in each columns" , data.isna().sum(), sep='\n')


# Country Has 488 missing Values, Agent has 16340 missing value & company has 112593 missing value

# In[ ]:


data = data.drop(['company'], axis = 1)
data = data.dropna(axis = 0)


# In[ ]:


data1 = data.copy()


# As company has maximum missing data lets drop that column
# Aslo drop all the rows that have NaN in them as per above code

# In[ ]:


data.info()


# Now we have 31 columns with equal data i.e. 102894

# Lets now check the unique values in each column

# In[ ]:


data['hotel'].unique()


# In[ ]:


data['hotel'] = data['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
data['hotel'].unique()


# with the above code line we have converted object values to integer values of 0 & 1
# With below codes we will convert all the object type data into integer values which machine can read

# In[ ]:


data['arrival_date_month'].unique()


# In[ ]:


data['arrival_date_month'] = data['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
data['arrival_date_month'].unique()


# In[ ]:


data['customer_type'].unique()


# In[ ]:


data['deposit_type'].unique()


# In[ ]:


data['reservation_status'].unique()


# In[ ]:


data['assigned_room_type'].unique()


# In[ ]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column. 
data['customer_type']= label_encoder.fit_transform(data['customer_type']) 
data['assigned_room_type'] = label_encoder.fit_transform(data['assigned_room_type'])
data['deposit_type'] = label_encoder.fit_transform(data['deposit_type'])
data['reservation_status'] = label_encoder.fit_transform(data['reservation_status'])
data['meal'] = label_encoder.fit_transform(data['meal'])
data['country'] = label_encoder.fit_transform(data['country'])
data['distribution_channel'] = label_encoder.fit_transform(data['distribution_channel'])
data['market_segment'] = label_encoder.fit_transform(data['market_segment'])
data['reserved_room_type'] = label_encoder.fit_transform(data['reserved_room_type'])
data['reservation_status_date'] = label_encoder.fit_transform(data['reservation_status_date'])
  
print('customer_type:', data['customer_type'].unique())
print('reservation_status', data['reservation_status'].unique())
print('deposit_type', data['deposit_type'].unique())
print('assigned_room_type', data['assigned_room_type'].unique())
print('meal', data['meal'].unique())
print('Country:',data['country'].unique())
print('Dist_Channel:',data['distribution_channel'].unique())
print('Market_seg:', data['market_segment'].unique())
print('reserved_room_type:', data['reserved_room_type'].unique())


# We have converted strings and object data into machine readable format

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor


# Let's now use Regression modles to check the best one.

# In[ ]:


X = data.drop(['previous_cancellations'], axis = 1)
y = data['previous_cancellations']


# Our Target is y with previous_cancellations, & X contains all the data except previous_cancellation
# with below codes we will train_test_split the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('Mean Absolute Error_lng:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_lng:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_lng:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_lng:', r2_score(y_test, y_pred).round(3))

## Linear Regression above##


# In[ ]:


ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train) #training the algorithm

y_pred = ridge.predict(X_test)

print('Mean Absolute Error_ridge:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_ridge:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_ridge:', r2_score(y_test, y_pred).round(3))

## Ridge Regression above##


# In[ ]:


clf = Lasso(alpha=0.1)

clf.fit(X_train, y_train) #training the algorithm

y_pred = clf.predict(X_test)

print('Mean Absolute Error_lasso:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_lasso:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_lasso:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_lasso:', r2_score(y_test, y_pred).round(3))

## Lasso Regression above##


# In[ ]:


logreg = LogisticRegression(solver = 'lbfgs')
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print('Mean Absolute Error_logreg:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_logreg:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_logreg:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_logreg:', r2_score(y_test, y_pred).round(3))

## Logistics Regression above ##


# In[ ]:


# Ridge Regression with Gridsearch ##
from sklearn.model_selection import GridSearchCV

parameters= {'alpha':[50,75,100,200, 230, 250], 'random_state':[5,10,20,50,], 'max_iter':[0.1,0.5,1,2,3,5]}

grid = GridSearchCV(ridge, parameters, cv=5)
grid.fit(X_train, y_train)
print ("Best_Score_Ridge : ", grid.best_score_)
print('best_para_Ridge:', grid.best_params_)


# In[ ]:


# Lasso Regression with Gridsearch ##
from sklearn.model_selection import GridSearchCV

parameters= {'alpha':[200, 230, 250,265, 270, 275, 290, 300], 'random_state':[2,5,10,20,50,], 'max_iter':[5,10,15,20,30,50,100]}

grid = GridSearchCV(clf, parameters, cv=5)
grid.fit(X_train, y_train)
print ("Best_Score_Lasso : ", grid.best_score_)
print('best_para_Lasso:', grid.best_params_)


# In[ ]:


# create regressor object 
rfe = RandomForestRegressor(n_estimators = 100, random_state = 42) 
 
# fit the regressor with x and y data 
rfe.fit(X, y)   
y_pred=rfe.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_score_RFE:', r2_score(y_test, y_pred).round(3))


# In[ ]:


ABR = AdaBoostRegressor(n_estimators = 100, random_state = 42) 
  
# fit the regressor with x and y data 
ABR.fit(X, y)   
y_pred=ABR.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_score_ABR:', r2_score(y_test, y_pred).round(3))


# Conclusion:
# Basd on the r2_square evalaution, Lasso Regression (r2_square= 0.039) is the best fit model 

# > ***Data Visualization***
# Lets do some data visualization

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ***Lead time for booking year on year***

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x ='arrival_date_year', y = 'lead_time', data = data1)


# In[ ]:


plt.figure(figsize = (12,5))
sns.barplot(x ='arrival_date_month', y = 'adults', data = data1)


# In[ ]:



data1.groupby(['arrival_date_year', 'arrival_date_month']).size().plot.bar(figsize=(15,5))


# Year on year arrival count is depecited in above graph

# In[ ]:


data1.groupby(['arrival_date_month'])['arrival_date_year'].size().plot.bar(figsize=(15,5))


# Average month count of arrival of all the three years is shown in above chart 

# In[ ]:


plt.figure(figsize = (30,10))

data1.groupby(['country']).size().sort_values(ascending= False).head(15).plot.bar()


# in the above chart you can see from which country maximum tourist are arriving.
# As shown below, numbers show that the top five countries are Poland, Britan, France, Spain & Germany

# In[ ]:


data1.groupby(['country']).size().sort_values(ascending=False)


# In[ ]:



data.groupby(['arrival_date_month','arrival_date_year'])['children', 'babies'].sum().plot.bar(figsize=(15,5))


# > ***From the above chart you can see how many childrens and babies arrived ***

# In[ ]:


plt.title('Cancellation')
plt.ylabel('Cancel_Sum')

data1.groupby(['hotel','arrival_date_year'])['is_canceled'].sum().plot.bar(figsize=(10,5))


# ***Above chart shows in the years which hotels were mostly cancelled***

# In[ ]:


data1.groupby(['hotel'])['booking_changes'].sum().plot.pie(radius = 2)
plt.show()


# ***Above Pie chart shows in which hotels booking changes were maximum.***

# In[ ]:


data1.groupby(['country'])['required_car_parking_spaces'].sum().sort_values(ascending=False)


# ***Above figures show that the tourists arriving from Poland, Spain,France, GB & Germany require car parking
# This is also shown in the chart below.***

# In[ ]:


data1.groupby(['country'])['required_car_parking_spaces'].sum().sort_values(ascending=False).head(15).plot.bar(figsize=(10,5))


# Showing only top 15 countries requiring parking space at hotel

# In[ ]:


data1.groupby(['deposit_type']).size().plot.bar()


# > ***What Kind of deposits are people paying is shown above***

# In[ ]:


data1.country.unique()


# In[ ]:


df_us = data1[data1.country == 'USA']
df_uk = data1[data1.country == 'GBR']
df_po = data1[data1.country == 'PRT']
df_ger = data1[data1.country == 'DEU']
df_sp = data1[data1.country == 'ESP']


# In[ ]:


df_us.head()


# ******

# In[ ]:


df_merged = pd.concat([df_us, df_uk, df_po, df_ger, df_sp])


# # Let's analyse data for US visitors

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x= 'arrival_date_month', y = 'booking_changes', data = df_us)


# ***Above chart shows in which month US visitors are arriving the most & also the booking change frequency***

# ***Stay in weekend nights or the five countries is decipted below***

# In[ ]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= 'stays_in_weekend_nights', y='stays_in_week_nights', hue = 'country', data = df_merged )


# In[ ]:




