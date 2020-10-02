#!/usr/bin/env python
# coding: utf-8

# Hello, this is the first time I write the explanation about what I did with the Indian Metro Data. So this data describes the traffic volume in India, along with the variables such as time, weather, wind direction, temperature, rain intensity, snow intensity, etc. So let's get started with importing the module.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# There are 2 datasets here: the train data and the test data.

# In[ ]:


train = pd.read_csv('/kaggle/input/Train.csv')
test = pd.read_csv('/kaggle/input/Test.csv')


# And here's our data.

# In[ ]:


train.head()


# As you can see, there are 13 variables to predict traffic volume in India. And they are:
# * date_time: The date and the time the data were collected
# * is_holiday: The categorical variable that describe whether the time the data were collected was in a holiday or not
# * air_pollution_index: Air Pollution Index on that day from 10 to 300
# * Humidity: The humidity measured in Celsius
# * wind_speed: The wind speed measured in miles per hour
# * visibility_in_miles: The visibility radius measured in miles
# * dew_point: The dew point measured in Celsius
# * temperature: The average temperature on that day measured in Kelvin
# * rain_p_h: The rain intensity measured in millimeters
# * snow_p_h: The snow intensity measured in millimeters
# * clouds_all: The percentage of cloud cover on that day
# * weather_type: The weather in brief description on that day
# * weather_description: The weather in full description on that day
# * traffic_volume: The traffic volume we want to predict

# Then, we check whether the data contains missing value as follows.

# In[ ]:


train.isnull().any()


# The data doesn't contain any missing value. So that's good news.

# Let's do some tweaks to extract the date_time variable. First by checking the variable type

# In[ ]:


train.dtypes


# The date_time variable is still an object variable. We need to change to datetime variable first in order to extract the variable.

# In[ ]:


train.date_time = pd.to_datetime(train.date_time)


# In[ ]:


train['year'] = train['date_time'].dt.year
train['month'] = train['date_time'].dt.month
train['day'] = train['date_time'].dt.day
train['dayofweek'] = train['date_time'].dt.dayofweek.replace([0,1,2,3,4,5,6],['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])
train['hour'] = train['date_time'].dt.hour
train.head()


# And here they are. The date_time variable has been extracted to 5 new variables. By this, we can see the traffic volume in a day hourly.

# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x=train['hour'],y=train['traffic_volume'])


# You can see from the graph above, the traffic volume is high during the day, especially in the rush hour (from 6 to 9 and from 16 to 18). Traffic volume is at the lowest point during the dusk, and then increasing when the day is stared. This is because in the day, Indian people are going to work, doing their job, and then going home.
# 
# We want to see the traffic volume difference by the day of week.

# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x=train['hour'],y=train['traffic_volume'],hue=train['dayofweek'])


# As you can see, during the weekend, people don't work so the traffic volume in the morning is not as high as during the day. You can also see that during the weekday, there are 2 peaks: in the morning rush hour and in the evening rush hour. The traffic volume doesn't show significant difference in the afternoon and after the evening rush hour.

# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x=train['hour'],y=train['traffic_volume'],hue=train.query("dayofweek in ['monday','sunday']")['dayofweek'])


# That is the clear difference of traffic volume on the weekday and on the weekend.

# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x=train['day'],y=train['traffic_volume'])


# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x=train['month'],y=train['traffic_volume'])


# In[ ]:


plt.figure(figsize=(10,7))
sns.lineplot(x=train['year'],y=train['traffic_volume'])


# Those 3 graphs I show you clearly shows that there are no significant increase or decrease if we see yearly, monthly, or daily.

# In[ ]:


train.groupby('is_holiday').agg(len)['date_time'].plot.bar()


# As you can see that there are very-very little data that has been collected on the holiday.

# In[ ]:


train.groupby('weather_type').agg(len)['date_time'].sort_values(ascending=False).plot.bar()


# Most of the time, the weather is cloudy or clear. There are very few thunderstorm, smoke, or squall there.

# Let's see the scatterplot of the numerical variable, to see if there are two variables that are highly correlated or if there are outliers.

# In[ ]:


sns.pairplot(train[['air_pollution_index','humidity','wind_speed','visibility_in_miles','dew_point','temperature','rain_p_h','snow_p_h','clouds_all','traffic_volume']])


# As you can see, the dew_point and the visibility in miles are highly correlated. So we will use one of them in the model. The graph also shows that temperature and rain_p_h have outliers. We will replace those data points with the median of each variables.

# In[ ]:


train['rain_p_h'] = train['rain_p_h'].replace(train['rain_p_h'].max(),train['rain_p_h'].median())
sns.distplot(a=train['rain_p_h'])


# In[ ]:


train['temperature'] = train['temperature'].replace(train['temperature'].min(),train['temperature'].median())
sns.distplot(a=train['temperature'])


# After some outlier-handling processes, we are ready to build the model. The predictor variable I'm going to choose is all of them except date_time, is_holiday, weather_description, year, month, day, dew_point. The I set the response variable to traffic_volume

# In[ ]:


X=train.drop(['date_time','is_holiday','weather_description','year','month','day','traffic_volume','dew_point'],axis=1)
y=train['traffic_volume']
X=pd.get_dummies(X)


# Then we split the data into training data and the testing data (on the train data!). We set 70% of the data are training data and the rest are testing data.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=1)


# We use 5 different model to see which model we choose as the final model. They are Linear Regression, Ridge Regression, Lasso Regression, XGBoost, and Random Forest. I know there are still a lot of model, but I'm gonna stick with these simple models.

# First, let's start by checking the cross validation score of Linear Regression. I check the cross validation score first in order to avoid overfitting.

# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error

linreg = LinearRegression()
linreg.fit(X_train,y_train)
print('Cross Validation Score: ',-1*cross_val_score(linreg,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())


# The cross validation score isn't too low to be the model. But if we want to see the variables that are able to predict the traffic volume, let's see that by the coefficient.

# In[ ]:


grafik_1 = pd.DataFrame({'Coef':linreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).head()
grafik_2 = pd.DataFrame({'Coef':linreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).tail()
grafik = pd.concat([grafik_1,grafik_2])
grafik.plot.bar()


# You can see that the variable that have significant amount to predict the traffic_volume is dayofweek, weather type, and snow_p_h. If the amount of snow is higher (or whether it is snowy on that day), then the traffic volume will decrease significantly. If it is friday, wednesday, or thursday (or probably weekday in general), then the traffic volume will increase. This will be very different if it is the weekend. If it is hazy or cloudy, the traffic volume will increase, while it will decrease if there are thunderstorm and squall.

# Let's see if we use Ridge Regression. Again, we will check the cross validation score.

# In[ ]:


ridreg = Ridge()
ridreg.fit(X_train,y_train)
print('Cross Validation Score: ',-1*cross_val_score(ridreg,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())


# It doesn't give much better result than Linear Regression though. Let's do the same with Ridge Regression: seeing the most significant variable to predict traffic volume.

# In[ ]:


grafik_1 = pd.DataFrame({'Coef':ridreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).head()
grafik_2 = pd.DataFrame({'Coef':ridreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).tail()
grafik = pd.concat([grafik_1,grafik_2])
grafik.plot.bar()


# The most significant ones are the same as the linear one though (probably as the consequence due to the same cross validation score). Let's see the Lasso Regression.

# In[ ]:


lasreg = Lasso()
lasreg.fit(X_train,y_train)
print('Cross Validation Score: ',-1*cross_val_score(lasreg,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())


# Lasso Regression also doesn't give better result. So I guess Regression isn't the best model to look for.

# In[ ]:


grafik_1 = pd.DataFrame({'Coef':lasreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).head()
grafik_2 = pd.DataFrame({'Coef':lasreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).tail()
grafik = pd.concat([grafik_1,grafik_2])
grafik.plot.bar()


# The most significant variables are still the same, except hour has significant role on predicting traffic volume though.

# Because Regression isn't the best model to look for, let's see if XGBoost can fix this.

# In[ ]:


from xgboost import XGBRegressor
warnings.simplefilter("ignore",UserWarning)
xgb = XGBRegressor(objective='reg:squarederror')
xgb.fit(X_train,y_train)
print('Cross Validation Score: ',-1*cross_val_score(xgb,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())


# It does gives us significant better result than regression! Great! But we have one final model to check. It is Random Forest. Let's see.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
print('Cross Validation Score: ',-1*cross_val_score(rf,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())


# It is better than XGBoost if we look at the cross validation score. So I guess I will use this model to predict traffic volume. But before we implement it, we should tune the parameters so that the mean absolute error is minimal.

# In[ ]:


hasil=[]
j=[]
for i in range(10,310,10):
    rf = RandomForestRegressor(n_estimators=i)
    rf.fit(X_train,y_train)
    pred = rf.predict(X_valid)
    hasil.append(mean_absolute_error(pred,y_valid))
    j.append(i)
score = pd.DataFrame({'Mean Absolute Error':hasil},index=j)
score.plot.line()


# As you can see on the graph, the mean absolute error isn't changing when the parameter is 150 or more. So we will tune the n_estimators to 150, and get the result

# In[ ]:


rf = RandomForestRegressor(n_estimators=150)
rf.fit(X_train,y_train)
pred = rf.predict(X_valid)
print('MAE: ',mean_absolute_error(pred,y_valid))


# Let's see which variables are the usefull to predict traffic volume.

# In[ ]:


tabel = pd.DataFrame({'Importance':np.round(rf.feature_importances_,decimals=3)},index=X_train.columns).sort_values(by='Importance',ascending=False).head(10)
tabel


# We see that hour, dayofweek, temperature, and wind_direction are 4 variables that are usefull to predict the traffic volume. We will then use this variable to our final model.

# In[ ]:


X=train[['hour','dayofweek','temperature','air_pollution_index','wind_direction']]
y=train['traffic_volume']
X=pd.get_dummies(X)
X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.3,random_state=1)
rf=RandomForestRegressor(n_estimators=150)
rf.fit(X_train,y_train)
pred = rf.predict(X_valid)
print(mean_absolute_error(pred,y_valid))


# Then, we predict the test data to get the final result. First by doing some edit to the data.

# In[ ]:


test_1 = test.copy()
test_1['date_time'] = pd.to_datetime(test_1['date_time'])
test_1['dayofweek'] = test_1['date_time'].dt.dayofweek.replace([0,1,2,3,4,5,6],['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])
test_1['hour'] = test_1['date_time'].dt.hour
test_1 = test_1[['hour','dayofweek','temperature','air_pollution_index','wind_direction']]
test_1 = pd.get_dummies(test_1)


# And this is the data to be used.

# In[ ]:


test_1.head()


# Finally, predict those data and save them in the kernel.

# In[ ]:


final_pred = rf.predict(test_1)
final_test = pd.read_csv('/kaggle/input/Test.csv')
final_pred = pd.DataFrame(np.round(final_pred,decimals=0),columns=['Predictions'])
result = pd.concat([final_test,final_pred],axis=1)
result.head()


# In[ ]:


result.to_csv('result.csv')


# And that's it! Thank you.
