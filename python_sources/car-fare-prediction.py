#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# In[ ]:


#Loading the dataset
train_data = pd.read_csv("../input/train-data/train_cab.csv", sep=",")
test_data  = pd.read_csv("../input/test-data/test.csv", sep=",")


# In[ ]:


#Missing Value Analysis
missing_val = pd.DataFrame(train_data.isnull().sum())
missing_val = missing_val.reset_index()
missing_val = missing_val.rename(columns={'index':'variables',0:'Missing_values'})
missing_val['Missing_Value_Percentage'] = (missing_val.Missing_values/len(train_data))*100
missing_val = missing_val.sort_values('Missing_Value_Percentage',ascending=False).reset_index(drop=True)
missing_val


# In[ ]:


#Dropping Missing values(NA values) very few viables have missing values and its better remove them.
train_data.drop(train_data[train_data.fare_amount.isnull()==True].index,axis=0,inplace=True)
train_data.drop(train_data[train_data.passenger_count.isnull()==True].index,axis=0,inplace=True)


# In[ ]:


#Rechecking Missing Value 
missing_val = pd.DataFrame(train_data.isnull().sum())
missing_val = missing_val.reset_index()
missing_val = missing_val.rename(columns={'index':'variables',0:'Missing_values'})
missing_val


# In[ ]:


#Checking the Data
train_data.head()


# In[ ]:


#Checking the datatypes
train_data.dtypes


# In[ ]:


#Reordering incurrect datatypes of Variables
train_data['fare_amount'] = pd.to_numeric(train_data['fare_amount'],errors='coerce')
train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'],infer_datetime_format=True,errors='coerce')
train_data['passenger_count'] = train_data['passenger_count'].astype('int')


# In[ ]:


train_data.dtypes


# > > Feature Engineering
# - Here pickup and drop locations are related to fare_amont of the data,
# - So we need to find the distance using pickup and drop location coordinates by using "Haversine distance formula" 

# In[ ]:


from math import radians, cos, sin, asin, sqrt
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def date_time_info(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], format="%Y-%m-%d %H:%M:%S UTC")
    
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day']  = data['pickup_datetime'].dt.day
    data['month'] = data['pickup_datetime'].dt.month
    data['weekday'] = data['pickup_datetime'].dt.weekday
    data['year']    = data['pickup_datetime'].dt.year
    
    return data

#Applying on train_data
train_data = date_time_info(train_data)
train_data['distance'] = distance(train_data['pickup_latitude'], 
                                     train_data['pickup_longitude'],
                                     train_data['dropoff_latitude'] ,
                                     train_data['dropoff_longitude'])
#Preprocessing on test data
#Applying distance and date_time_info function on test_data
test_data = date_time_info(test_data)
test_data['distance'] = distance(test_data['pickup_latitude'], 
                                     test_data['pickup_longitude'],
                                     test_data['dropoff_latitude'] ,
                                     test_data['dropoff_longitude'])

test_key = pd.DataFrame({'key_date':test_data['pickup_datetime']})
test_data = test_data.drop(columns=['pickup_datetime'],axis=1)

train_data.head()


# In[ ]:


#weekday starts from 0 to 6
train_data.describe()


# 1. 1. Checking data distribution Before Outlier Analysis

# In[ ]:


continuous_variables = ['year','month','fare_amount','passenger_count','pickup_longitude',
                        'pickup_latitude','dropoff_longitude','dropoff_latitude','distance']
for i in continuous_variables:
    plt.hist(train_data[i],bins=18)
    plt.title("Checking Distribution for Variable "+str(i))
    plt.ylabel("Density")
    plt.xlabel(i)
    plt.show()


# In[ ]:


train_data.dtypes


# Outlier Analysis
# - from above data fare_amount has negative values and it can't be -ve values 
# - Found negative value of fare amount.
#   Fare never be negative let's drop those rows which are having negative
#   fare amount and also remove outliers
# - passenger_count should be  > 0 and < =6
# - Latitude should be between min is 40.568973 and max is 41.709555
# - Longitude should be min is -74.263242 and max is  -72.986532
# 

# In[ ]:


#Outlier Visualizations
col = ['fare_amount', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude']
for i in col:
    sns.boxplot(y=train_data[i])
    fig=plt.gcf()
    fig.set_size_inches(5,5)
    plt.show()


# In[ ]:


#Fare_amount data distribution by using Scatter plot for all the observations
plt.scatter(x=train_data.fare_amount,y=train_data.index)
plt.ylabel('Index')
plt.xlabel('fare_amount')
plt.show()


# In[ ]:


#Fare_amount data distribution by using Scatter plot for selected observations (of x lim range from 1 to 70)
#because from 70 onwards all observations are extreme outliers.
plt.scatter(x=train_data.fare_amount,y=train_data.index)
plt.ylabel('Index')
plt.xlim(1,70)
plt.xlabel('fare_amount')
plt.show()


# In[ ]:


#passenger_count data distribution by using Scatter plot for all the observations,
#because from 70 onwards all observations are extreme outliers.
plt.scatter(x=train_data.passenger_count,y=train_data.index)
plt.ylabel('Index')
plt.xlabel('passenger_count')
plt.show()


# In[ ]:


#passenger_count data distribution by using Scatter plot for sected observations ( x lim range from 1 to 10)
#because from 70 onwards all observations are extreme outliers.
plt.scatter(x=train_data.passenger_count,y=train_data.index)
plt.ylabel('Index')
plt.xlim(1,10)
plt.xlabel('passenger_count')
plt.show()


# In[ ]:


#Scatter plot for distance variable for all the observations
plt.scatter(x=train_data.distance,y=train_data.index)
plt.ylabel('Index')
#plt.xlim(1,10)
plt.xlabel('ditsance')
plt.show()


# In[ ]:


#Scatter plot for distance variable for selected index range of observations
plt.scatter(x=train_data.distance,y=train_data.index)
plt.ylabel('Index')
plt.xlim(1,30)
plt.xlabel('ditsance')
plt.show()


# Manualy removing outliers based on the test data feature conditions

# In[ ]:


#Finding the longitude min and max of test data
lon_min=min(test_data.pickup_longitude.min(),test_data.dropoff_longitude.min())
lon_max=max(test_data.pickup_longitude.max(),test_data.dropoff_longitude.max())
print(lon_min,',',lon_max)

#Finding the latitude min and max of test data
lat_min=min(test_data.pickup_latitude.min(),test_data.dropoff_latitude.min())
lat_max=max(test_data.pickup_latitude.max(),test_data.dropoff_latitude.max())
print(lat_min,',',lat_max)


# In[ ]:


train_data.describe()


# In[ ]:


#1 - Removing -ve values from the fare_amount variable
train_data_new = train_data  
train_data=train_data.drop(train_data[(train_data.fare_amount<=0) | (train_data.fare_amount>=65)].index,axis=0)

#2 - Removing null values from passenger count
#From the the test data passenger count lies between min is 1 and max is 6 
train_data=train_data.drop(train_data[(train_data.passenger_count<=0) | (train_data.passenger_count>6)].index,axis=0)

#4 - Removing pickup_latitude,dropoff_latitude, pickup_longitude, and dropoff_longitude
train_data=train_data.drop(train_data[(train_data.pickup_latitude <lat_min)| (train_data.pickup_latitude >lat_max)].index,axis=0)
train_data=train_data.drop(train_data[(train_data.dropoff_latitude <lat_min) | (train_data.dropoff_latitude >lat_max)].index,axis=0)
train_data=train_data.drop(train_data[(train_data.pickup_longitude <lon_min) | (train_data.pickup_longitude >lon_max)].index,axis=0)
train_data=train_data.drop(train_data[(train_data.dropoff_longitude <lon_min) | (train_data.dropoff_longitude >lon_max)].index,axis=0)

#4 - Removing Outliers in the Distance variable outliers
train_data=train_data.drop(train_data[(train_data.distance <=0)].index,axis=0)


# In[ ]:


train_data.shape


# In[ ]:


train_data.isna().sum()


# In[ ]:


train_data.dropna(axis=0,inplace=True)
train_data.isnull().sum()


# In[ ]:


train_data.shape


# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# Checking data distribution after Outlier Analysis

# In[ ]:


continuous_variables = ['year','month','fare_amount','passenger_count','pickup_longitude',
                        'pickup_latitude','dropoff_longitude','dropoff_latitude','distance']
for i in continuous_variables:
    plt.hist(train_data[i],bins=18)
    plt.title("Checking Distribution for Variable "+str(i))
    plt.ylabel("Density")
    plt.xlabel(i)
    plt.show()


# In[ ]:


#Plot for fare_amount variation across distance
plt.scatter(y=train_data['distance'],x=train_data['fare_amount'])
plt.xlabel('fare')
plt.ylabel('distance')
plt.show()


# In[ ]:


sns.countplot(train_data['passenger_count'])


#     **Asumptions for Hypothesis**

# In[ ]:


train_data.columns


#     1)Check the pickup date and time affect the fare or not

# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_data['day'], y=train_data['fare_amount'], s=1.5)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.show()


# The fares throught the month mostly seem uniform
# 
# 

# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_data['hour'], y=train_data['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# The time of day definitely plays an important role. The frequency of cab rides seem to be the lowest at 5AM

# ![](http://)2 - Number of Passengers vs Fare

# In[ ]:


plt.figure(figsize=(15,7))
plt.hist(train_data['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


plt.figure(figsize=(15,7))
plt.scatter(x=train_data['passenger_count'], y=train_data['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# single passengers are the most frequent travellers, and the highest fare also seems to come from cabs which carry just 1 passenger.
# 
# 

#     3 - Does the day of the week affect the fare?
# 

# In[ ]:


plt.figure(figsize=(15,7))
plt.hist(train_data['weekday'], bins=100)
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.show()


# day of the week doesn't seem to have the effect on the number of cab rides
# 
# 

# **Feature Scaling**

# In[ ]:


#Normality check
#%matplotlib inline  
plt.hist(train_data['fare_amount'], bins='auto')


# In[ ]:


#taking copy of the data
train_data_df = train_data.copy()


# In[ ]:


#Nomalization
"""
cnames =['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour',
       'day', 'month', 'weekday', 'year', 'distance']
for i in cnames:
    print(i)
    train_data[i] = (train_data[i] - min(train_data[i]))/(max(train_data[i]) - min(train_data[i]))
"""


# In[ ]:


#train_data = train_data_df.copy()
#correlation between numerical variables
num = pd.DataFrame(train_data.select_dtypes(include=np.number))
cor = num.corr()        
cor


# In[ ]:


num


# In[ ]:


train_data.describe()


# In[ ]:


#Coorelation Plot to check the Coorelation
continuous_variables =['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour',
       'day', 'month', 'weekday', 'year', 'distance']

df_cor = train_data.loc[:,continuous_variables]
f, ax = plt.subplots(figsize=(10,10))

#Generate correlation matrix
cor_mat = df_cor.corr()

#Plot using seaborn library
sns.heatmap(cor_mat, mask=np.zeros_like(cor_mat, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=False),
            square=True, ax=ax)
plt.plot()

train_data.shape
# In[ ]:


train_data.passenger_count


# In[ ]:


#VIF to check the Correlation
#pick_up date is correlated to its extracted columns i.e day, year, month, weekday
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = add_constant(num)
pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)


# Feature Selection

# In[ ]:


#Removing variable 'Pickup datetime' beacause day,year,month carries all the information  from it
del train_data['pickup_datetime']

#Selected variables for model building
train_data.columns

"""
Index(['fare_amount', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour',
       'day', 'month', 'weekday', 'year', 'distance'],
      dtype='object')
"""


# In[ ]:


train_data.columns


# Model Building for Train data

# In[ ]:


#Splitting data into test and train
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#-> For Train data -> train_data

y = train_data['fare_amount']
X = train_data.drop(columns=['fare_amount'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def evaluate(model, test_features, test_actual):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_actual)
    rmse = np.sqrt(mean_squared_error(test_actual,predictions))
    mape = 100 * np.mean(errors / test_actual)
    accuracy = 100 - mape
    rsquared = r2_score(test_actual, predictions)
    df_pred = pd.DataFrame({'actual':test_actual,'predicted':predictions})
    print('<---Model Performance--->')
    print('R-Squared Value = {:0.2f}'.format(rsquared))
    print('RMSE = {:0.2f}'.format(rmse))
    print('MAPE = {:0.2f}'.format(mape))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return rmse


# In[ ]:


#Linear regression
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression().fit(X_train,y_train)

#predicting and testing on train data
evaluate(model_lr, X_test, y_test)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeRegressor
model_dt = DecisionTreeRegressor(random_state = 123).fit(X_train,y_train)

#predicting and testing on train data
evaluate(model_dt, X_test, y_test)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor().fit(X_train,y_train)

#predicting and testing on train data
evaluate(model_rf, X_test, y_test)


# In[ ]:


#Parameters of base model
model_rf.get_params()


# In[ ]:


X_train.columns


# In[ ]:


#Printing Feature importance of the model
feat_importances = pd.Series(model_rf.feature_importances_, index=X_train.columns)
feat_importances.plot(kind='barh')


# In[ ]:


X_train = x_train.copy()


# Modelling Using SVM Classifier

# In[ ]:


#SVM
from sklearn.svm import SVR
from sklearn.metrics import classification_report, accuracy_score
model = SVR()
model.fit(X_train, y_train)
#predicting and testing on train data
evaluate(model, X_test, y_test)


# Modelling Using XGBoost

# In[ ]:


from xgboost import XGBClassifier
clf = XGBClassifier()
clf
clf.fit(X_train, y_train)
#predicting and testing on train data
evaluate(clf, X_test, y_test)


# Modelling Using KNN

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
# Instantiate learning model (k = 3)
regresor = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)

#predicting and testing on train data
evaluate(regresor, X_test, y_test)


# **Hyper Parameter tuning**

# In[ ]:


#Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 25, 30],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [2,3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [300,500,600,800]
}
# Create a base model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


grid_search.fit(X_train,y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_

#Applying gridsearchcsv to test data
grid_accuracy = evaluate(best_grid,X_test, y_test)


# In[ ]:


#Getting the best Parameters
#grid_search.best_params_
#or
grid_search.best_estimator_


# In[ ]:


#Printing Feature importance by visualizations
feat_importances_hyp = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances_hyp.plot(kind='barh')


#     **Applying hyperparameters tuned  base model on test data **

# In[ ]:


#Building Random Forest model with hypertuned parameters
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=10,
           min_weight_fraction_leaf=0.0, n_estimators=800, n_jobs=None,
           oob_score=False, random_state=None, verbose=0, warm_start=False).fit(X_train,y_train)

#predicting and testing on train data
evaluate(model_rf, X_test, y_test)


# **Predicting on given test data **

# In[ ]:


#Checking the given test data
test_data.describe()


# In[ ]:


predictions = model_rf.predict(test_data)
predicted_test = pd.DataFrame({'pickup_datetime':test_key['key_date'],'Predicted_Fare':predictions})
predicted_test.head(10)

