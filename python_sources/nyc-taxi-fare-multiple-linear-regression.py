#!/usr/bin/env python
# coding: utf-8

# # NYC Taxi Fare Prediction: Multiple Linear Regression
# 
# The dataset is provided by Google. It is very large dataset having around 50M records. The decription of the features are avilable in Data section at Kaggle.
# 
# There are two main files. 
# 1. **train.csv **(We will use this to perform EDA, build and train our model.)
# 2. **test.csv** (We will use this file to validate our model by generating predictions.)
# 
# For an excellent EDA you can refer to <a href="https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration">NYC Taxi Fare: Data Exploration.</a> kernel on Kaggle.

# ## Multiple Linear Regression: 
# 
# We all knows that multiple linear regression is used on problems having more than 1 independent variables. Here,
# independent variables means features(columns of dataset), which are used to predict target variable (dependent
# variable).  Here, I am going to use MLR technique.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv', nrows= 10_000_000)


# Let's slicing off unecessary components of the datetime and specify the date format. This will results in a more efficiecnt conversion to a datetime object.

# In[ ]:


df_train.pickup_datetime = df_train.pickup_datetime.str.slice(0,16)


# In[ ]:


df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'], utc=True, format = '%Y-%m-%d %H:%M')


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# ## Data Cleaning: 

# There are few important observation from this description of dataset.
# 
# * Minimum fare_amount is negative, which is quite unrealistic in the scenario. I will drop those fields.  
# * Notice that some minimum and maximum longitude/latitude are off the boundary of New York city. I will also remove those. May be, I will define boundary for longitude/lattitude(lat: 40.7141667, long:  -74.0063889) to remove outliers.
# * Also, maximum passenger_count is 208, which is also an outlier. I will drop those records too.
# 

# In[ ]:


# Filter out negative fare amount and maximum passenger_count
print("Old Size Before Filter: %d" %(len(df_train)))
df_train = df_train[(df_train.fare_amount >=0) & (df_train.passenger_count <=10)]
print("New Size After Filter: %d" %(len(df_train)))


# I am refering <a href="https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration">this kernel </a> for EDA. It is really an inspiring one and have very deep analysis of data. There are many good explorations in that kernel that, I have used here.

# In[ ]:


# Let's plot histogram of fare_amount to see its distribution across data.

df_train[df_train.fare_amount < 100].fare_amount.hist(bins=100, figsize=(10,5))
plt.xlabel("Fare in USD $")
plt.title("Fare Amount Distribution")


# In the histogram of the fare_amount there are some small spikes between USD 40 and USD 60. This could indicate some fixed fare price (e.g. to/from airport). That is further explored in <a href="https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration">NYC Taxi Fare: Data Exploration.</a>

# Let's check for missing or null values present in the dataset. It would not affect model if we remove those null values as the dataset is large.

# In[ ]:


print(df_train.isnull().sum())


# In[ ]:


print("Old Size : %d" %(len(df_train)))
df_train=df_train.dropna(axis = 0)
print("New Size: %d" %(len(df_train)))


# As we know that, some data oints have minimum and maximum lattitude/longitude off the boundary of NYC. So, we have to filter those outliers too.

# In[ ]:


# Filtering out off boundary points. Boundary of New York City is (-75, -73, 40, 42)
def NYC(df):
    boundary_filter = (df.pickup_longitude >= -75) & (df.pickup_longitude <= -73) &                       (df.pickup_latitude >= 40) & (df.pickup_latitude <= 42) &                       (df.dropoff_longitude >= -75) & (df.dropoff_longitude <= -73) &                       (df.dropoff_latitude >= 40) & (df.dropoff_latitude <= 42)
    df = df[boundary_filter]
    return df


# In[ ]:


print('Old size: %d' % len(df_train))
df_train = NYC(df_train)
print('New size: %d' % len(df_train))


# ### Computing Distance :
# 
# We know that **"Manhattan Distance"** metric will give us better approximation of distance between two points in given plane.So, we will use manhattan distance formula to compute distance between pickup and dropoff points.

# In[ ]:


def distance_between_pickup_dropoff(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    d = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    return d


# ### Extracting New Features:
# 
# I have used taxi many times (e.g Uber) to travel from school to home and home to school. In my general observation I have found many interesting facts about fare prices. 
# 
# 
# 1. The fare amount was high in peak hours. (for ex: Morning and Night time)
# 2. The fare amount was high during bad weathers too. (snow or thunder storm)
# 3. The fare amount was high on holidays and sometimes in weekends.
# 4. The fare amount was low in afternoons and before noon.
# 
# From this observations, I can say that time is also one of the factors affecting fare_amount. So, we can use features like "hour" and "day". Also, we can consider "month" and "year"  too as features and see if those affects fare_amount.
# 

# In[ ]:


# Extracting Features 

df_train['hour'] = df_train.pickup_datetime.dt.hour
df_train['day'] = df_train.pickup_datetime.dt.day
df_train['month'] = df_train.pickup_datetime.dt.month
df_train['year'] = df_train.pickup_datetime.dt.year
df_train.drop('pickup_datetime', axis =1, inplace = True)

# Creating actual_distance column as measure of manhattan distance

df_train['actual_distance'] = distance_between_pickup_dropoff(df_train.pickup_latitude, df_train.pickup_longitude,
                                                             df_train.dropoff_latitude, df_train.dropoff_longitude)


# In[ ]:


# Let's check how our new data set looks like.

df_train.head()


# ## Visualization
# 
# We have latitude and longitude values. So, we can use those to plot the data and see what will come up.
# 
# Here, I am using pickup_latitude and pickup_longitude. We can generate smae plot for dropoff_latitude and dropoff_longitude too.

# In[ ]:


# Here, i am bounding the longitude and latitude values to get clear and zoomed plot.
df_plot = df_train[(df_train.pickup_longitude >= -74.1)&(df_train.pickup_longitude <= -73.8) & (df_train.pickup_latitude >=40.6)
                  & (df_train.pickup_latitude <=40.9)]


# In[ ]:


# In scatter plot arguments c = 'r' is "color red", s= 0.01 is "size of dots" and alpha = 0.5 is "opacity of dots"
fig, ax = plt.subplots(1, 1, figsize=[10,10])
ax.scatter(df_plot.pickup_longitude[:3_00_000], df_plot.pickup_latitude[:3_00_000],c = 'r', s= 0.01,alpha=0.5)


# In[ ]:


# Let's zoom little more

zoomed_data =  df_train[(df_train.pickup_longitude >= -74.02)&(df_train.pickup_longitude <= -73.95) & (df_train.pickup_latitude >=40.7)
                  & (df_train.pickup_latitude <=40.80)]


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=[10,10])
ax.scatter(zoomed_data.pickup_longitude[:3_00_000], zoomed_data.pickup_latitude[:3_00_000],c = 'b', s= 0.01,alpha=0.5)


# ## Building a Model:
# 
# It's time to build a model for training. I will use Linear Regression model of Scikit learn library. I will also measure RMSE(Root Mean Squared Error) to check accuracy of training set and test set.

# In[ ]:


# Let's create feature vector
# We do not want trip involving 0 passenger_count
filt = (df_train.passenger_count > 0) & (df_train.fare_amount < 250)
features = ['passenger_count','hour','year','day','month','actual_distance']


# In[ ]:


for f in features:
    related = df_train.fare_amount.corr(df_train[f])
    print("%s: %f" % (f,related))


# In[ ]:


final_features =['year','hour','actual_distance','passenger_count']


# In[ ]:


X = df_train[filt][final_features].values # Feature Vector
Y = df_train[filt]['fare_amount'].values # Target Variable


# In[ ]:


X.shape, Y.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# In[ ]:


# Splitting data set into train and test 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)


# In[ ]:


regressor = LinearRegression()


# In[ ]:


metric = 'neg_mean_squared_error'
scores = cross_val_score(regressor, X_test, y_test, cv = 10, scoring = metric)


# In[ ]:


scores


# In[ ]:


np.sqrt(np.abs(scores))


# In[ ]:


np.sqrt(np.abs(scores.mean()))


# In[ ]:


regressor.fit(X_train, y_train)


# In[ ]:


y_train_pred = regressor.predict(X_train)


# In[ ]:


def error(y, y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))


# In[ ]:


rmse = error(y_train, y_train_pred)


# In[ ]:


rmse


# In[ ]:


y_test_pred = regressor.predict(X_test)


# In[ ]:


rmse = error(y_test, y_test_pred)


# In[ ]:


rmse


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


alphas =[1e-5,1e-3, 1e-2, 0.02, 0.04,0.08,0.1]


# In[ ]:


for alpha in alphas:
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    rmse = error(y_train, y_train_pred)
    print("alpha : {%.5f} RMSE : {%.9f}" %(alpha,rmse))


# In[ ]:


lasso = Lasso(alpha = 0.01)


# In[ ]:


lasso.fit(X_train, y_train)


# In[ ]:


y_test_pred = lasso.predict(X_test)


# In[ ]:


rmse = error(y_test, y_test_pred)


# In[ ]:


rmse


# ## Decision Tree Regressor
# 
# The RMSE value using Lasso and Simple Linear Regression is almost same and little bigger. So, I search on the internet for, "How to reduce RMSE for regression model?" The use of cross validation was one of the idea. However, I also came to know that Decision Tree Regressor might come in handy to reduce RMSE or improve accuracy of regression problem. 
# 
# You can see how its easy to use Decision Tree Regressor on regression problem. One thing to notice that you have to provide value for **max_depth** argument. If you do not provide value then the algorithm will go in too much depth that you will get RMSE near to 0. It means that decision tree run into problem of **"Overfitting"**. No one wants that.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


reg = DecisionTreeRegressor(max_depth = 17)


# In[ ]:


reg.fit(X_train, y_train)


# In[ ]:


y_trn_pred = reg.predict(X_train)


# In[ ]:


rmse = error(y_train, y_trn_pred)


# In[ ]:


rmse


# In[ ]:


y_tst_pred = reg.predict(X_test)


# In[ ]:


rmse = error(y_test, y_tst_pred)


# In[ ]:


rmse


# You can see that RMSE value for training is low than RMSE value for testing data. It is common to have this. The model does not have problem of "Overfitting" or "Underfitting". 
# 

# # Kaggle Submission

# In[ ]:


df_test =  pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


df_test.pickup_datetime = df_test.pickup_datetime.str.slice(0,16)

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'], utc=True, format = '%Y-%m-%d %H:%M')


# In[ ]:


# Extracting Features for test set

df_test['hour'] = df_test.pickup_datetime.dt.hour
df_test['day'] = df_test.pickup_datetime.dt.day
df_test['month'] = df_test.pickup_datetime.dt.month
df_test['year'] = df_test.pickup_datetime.dt.year
df_test.drop('pickup_datetime', axis =1, inplace = True)

# Creating actual_distance column as measure of manhattan distance

df_test['actual_distance'] = distance_between_pickup_dropoff(df_test.pickup_latitude, df_test.pickup_longitude,
                                                             df_test.dropoff_latitude, df_test.dropoff_longitude)


# In[ ]:


X_test = df_test[final_features].values


# In[ ]:


y_pred_test_set = reg.predict(X_test)


# In[ ]:


submission =  pd.DataFrame({'key': df_test.key, 'fare_amount': y_pred_test_set},columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)


# In[ ]:


submission


# ### Further Improvement:
# I also came to know that there are many more algorithms available to improve RMSE score. Ex: Gradient Boosting Algorithms
# 
# There is no perfect solution to Machine Learning problem. One algorithm might works well for one problem while the same algorithm might not work for other but same kind of problem. We have to test many algorithm before reaching to any conclusion on problem. 

# In[ ]:




