#!/usr/bin/env python
# coding: utf-8

# # Google New York Taxi Fare Prediction

# This is a short kernel I made for myself and others as a reference guide/tutorial whilst I'm working out the best way to make accurate predictions. It's probably the most useful to beginners and I did try to describe/explain almost everything. I hope you find this useful, feel free to comment or contact me with any suggestions or questions.

# ![](https://kaggle2.blob.core.windows.net/competitions/kaggle/10170/logos/header.png?t=2018-07-12-22-07-30)

# # Import Libraries and Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle #calculate distances
from sklearn import metrics #evaluating models
#from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score #set splitting and validation
from sklearn.linear_model import LinearRegression 
import xgboost as xgb #XGBoost classifier
import matplotlib.pyplot as plt #plotting
import seaborn as sns #plotting
from math import sin, cos, sqrt, atan2, radians
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Check for the comptetition files
print(os.listdir("../input"))


# In[ ]:


#Load test dataset
test = pd.read_csv('../input/test.csv')


# In[ ]:


#Get types of values
test.dtypes


# As pointed out in this kernel https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows there is no need to use float64 for GPS coordinates, and float32 will provide just as accurate results and allow you to import and process data much quicker.

# In[ ]:


#Set lighter types to a dictionary to speed up train dataset (float64 is an overkill for GPS coordinates)
types = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}


# In[ ]:


#Load portion of the dataset in the defined types
train = pd.read_csv('../input/train.csv',nrows=100000,dtype=types)


# # Data Exploration and Cleanup

# Now let's look at what the data look like and some general statistics about it.

# In[ ]:


train.head()


# In[ ]:


train.describe()


# Let's plot distribution of couple of the values to get an easier insight into the data and the way it's distributed. You can see that there are a few values which are 0 dollars or less on fare_amount or 0 passengers on passenger_count. Feel free to check distributions of other values.

# In[ ]:


#Distribution plot of fares
sns.distplot(train['fare_amount'])


# In[ ]:


#Distribution plot of passanger count
sns.distplot(train['passenger_count'])


# Based on just a quick look at the data, we can see that it's not 100% clean and some entries will contribute to higher error rates. As there are more than enough entries, we can easily sacrifice any rows that have null data. Note that depending on how large of a slice of training data you took, you might not see any null values, but there are some in the 6 million rows provided. Just in case let's drop them. Additionally we can drop all entries where fare_amount or passanger_count is less or equal to 0. We can do a similar cleanup with coordinates, as the dataset is on New York taxi fares, and some of the values go beyond New York coordinates, and some of them are clearly errors.

# In[ ]:


#Check how many rows have null values
train.isnull().sum()


# In[ ]:


#Drop nulls if exist
train.dropna(inplace=True)


# In[ ]:


#Clean up the trian dataset to eliminate out of range values
train = train[train['fare_amount'] > 0]
train = train[train['pickup_longitude'] < -72]
train = train[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 44)]
train = train[train['dropoff_longitude'] < -72]
train = train[(train['dropoff_latitude'] > 40) & (train['dropoff_latitude'] < 44)]
train = train[(train['passenger_count'] > 0) & (train['passenger_count'] < 10)]


# Now we can see there are no obvious inconsitentcies with the data.

# In[ ]:


train.describe()


# # Feature Engineering

# We're only provided with a 8 columns of directly useable data, however we can extract much more information from it by engineering features from those columns or combinations of them. To start with we can get the distanct between the pickup and dropoff points, which should be a strong predictor for the fare. 

# In[ ]:


#Define function to calculate distance in km from coordinates
def dist_calc(df):
    for i,row in df.iterrows():
        df.at[i,'distance'] = great_circle((row['pickup_latitude'],row['pickup_longitude']),(row['dropoff_latitude'],row['dropoff_longitude'])).km


# In[ ]:


#Quicker but slightly less accurate
def quick_dist_calc(df):
    R = 6373.0
    for i,row in df.iterrows():

        lat1 = radians(row['pickup_latitude'])
        lon1 = radians(row['pickup_longitude'])
        lat2 = radians(row['dropoff_latitude'])
        lon2 = radians(row['dropoff_longitude'])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        df.at[i,'distance'] = distance


# In[ ]:


#Get distance for both sets
quick_dist_calc(train)
quick_dist_calc(test)


# Then if you look at the pickup_datetime column, we can extract quite a few datetime related features from it. First we need to format the string column into datetime format, after which it will be easy getting atomic features like hour, day, year, etc. of the pickup. 

# In[ ]:


#Get useable date for feature engineering
train['pickup_datetime'] = train['pickup_datetime'].str.replace(" UTC", "")
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

test['pickup_datetime'] = test['pickup_datetime'].str.replace(" UTC", "")
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


# In[ ]:


#Getting interger numbers from the pickup_datetime
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year


# Last but not least, we can look at fare correlation with the distance from certain hotspots around New York where prices will be higher or lower than usual. You can have a look at this great kernel https://www.kaggle.com/shaz13/simple-exploration-notebook-map-plots-v2 where the fare data is plotted geographically and gives an nice visual representation of such places. The most obvious ones would be airports and the center of Manhattan. Check out this kernel https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares for getting distance to and from higher fare hotspots. 

# In[ ]:


#Function for distance calculation between coordinates as mapped variables
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
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


# In[ ]:


#Function for calculating distance between newly obtained distances from the hotspots.
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
    
    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)
    
    return dataset


# In[ ]:


#Run the functions to add the features to the dataset
train = add_airport_dist(train)
test = add_airport_dist(test)


# Check the data after addition of all the new variables and make sure train and test datasets match.

# In[ ]:


train.head()


# In[ ]:


test.head()


# You can plot a correlation between all the features using a heatmap. The further away the values from 0, the more of an impact they play on the final fare prediciton.

# In[ ]:


#Plot heatmap of value correlations
plt.figure(figsize=(15,8))
sns.heatmap(train.drop(['key','pickup_datetime'],axis=1).corr(),annot=True,fmt='.4f')


# # Model Training

# As the coordinates columns seem to directly correlate with the fare_amount, I decided to leave them in the model fitting and prediction, along with all the newly generated features. Now we need to drop non-predicting columns and split the data into train and test sets for training the model. 

# In[ ]:


X = train.drop(['key','fare_amount','pickup_datetime'],axis=1)
y = train['fare_amount']


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


#Split train set into test and train subsets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# In[ ]:


#Drop columns from test dataset we're not going to use
test_pred = test.drop(['key','pickup_datetime'],axis=1)


# In[ ]:


#Scale values if needed for a particular model
#scaler = RobustScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.fit_transform(X_test)
#X_scaled = scaler.fit_transform(X)
#test_scaled = scaler.fit_transform(test_pred)


# ## Linear Regression

# Let's start with a simple linear regression model and see how well it does for predicting the fare_amount. 

# In[ ]:


#Initilise a linear regression model, fit the data and get scores
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.score(X_train,y_train))
print(lm.score(X_test,y_test))


# In[ ]:


#Predict fares and get a rmse for them
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse


# In[ ]:


#Predict final fares for submission
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions


# In[ ]:


#Check predictions have the correct dimensions
LinearPredictions.size


# In[ ]:


#Set up predictions for a submittable dataframe
linear_submission = pd.DataFrame({"key": test['key'],"fare_amount": LinearPredictions},columns = ['key','fare_amount'])


# In[ ]:


#Check the submissions look reasonable
linear_submission.head()


# ## Gradient Boosting

# Linear regression seemed to do reasonably well, but you can get much more accurate predictions using a more finely tuned model such as gradient boosting optimization. First we define the model and some basic parameters for it (feel free to play around to see if you can get better results). 

# In[ ]:


#Define a XGB model and parameters
def XGBoost(X_train,X_test,y_train,y_test):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=400, 
                    early_stopping_rounds=30,evals=[(dtest,'test')])


# In[ ]:


#Fit data and optimise the model, generate predictions
xgbm = XGBoost(X_train,X_test,y_train,y_test)
XGBPredictions = xgbm.predict(xgb.DMatrix(test_pred), ntree_limit = xgbm.best_ntree_limit)


# In[ ]:


#Check if predictions seem to be realistic looking
XGBPredictions


# In[ ]:


#Round predictions to 2 decimal numbers
XGBPredictions = np.round(XGBPredictions, decimals=2)
XGBPredictions


# # Submission

# Now seeing that the XGB model has done much better than linear regression, let's prepare the data for submission by putting it in the correct format and match each prediction fare with the corresponding key. After which generate a csv file which will be ready to be uploaded for final submission.

# In[ ]:


#Prepare predictions for submission
XGB_submission = pd.DataFrame({"key": test['key'],"fare_amount": XGBPredictions},columns = ['key','fare_amount'])
XGB_submission.head()


# In[ ]:


#submission = linear_submission
submission = XGB_submission


# In[ ]:


#Generate the final submission csv file
submission.to_csv('XGBSubmission.csv',index=False)


# That is it for this quick guide on some basic data exploration, analysis and prediction. Feel free to ask me any questions or comment any suggestions/corrections if you have any. I hope you've found this useful.
