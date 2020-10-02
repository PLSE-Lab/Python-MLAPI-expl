#!/usr/bin/env python
# coding: utf-8

# Hey there!!
# 
# Here is a list of things I have done so far:
# 

# In[ ]:


import matplotlib.pyplot as plt    #--- for plotting ---
import numpy as np                 #--- linear algebra ---
import pandas as pd                #--- data processing, CSV file I/O (e.g. pd.read_csv) ---
import seaborn as sns              #--- for plotting and visualizations ---

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#--- Let's peek into the data
print (train_df.head())
print (test_df.head())


# **Adding newer columns to the dataframe**

# In[ ]:


from math import radians, cos, sin, asin, sqrt   #--- for the mathematical operations involved in the function ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train_df['Haversine_dist'] = train_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
test_df['Haversine_dist'] = test_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
#train_df['Haversine_dist'] = haversine(train_df['pickup_longitude'], train_df['pickup_latitude'],train_df['dropoff_longitude'], train_df['dropoff_latitude'])
#print (train_df.head())


# In[ ]:


def arrays_bearing(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs1_rads = np.radians(lngs1)
    lngs2_rads = np.radians(lngs2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) - np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    
    return np.degrees(np.arctan2(y, x))

train_df['bearing_dist'] = arrays_bearing(
train_df['pickup_latitude'], train_df['pickup_longitude'], 
train_df['dropoff_latitude'], train_df['dropoff_longitude'])

test_df['bearing_dist'] = arrays_bearing(
test_df['pickup_latitude'], test_df['pickup_longitude'], 
test_df['dropoff_latitude'], test_df['dropoff_longitude'])

#print (train_df.head())
#print (test_df.head())


# In[ ]:


train_df['Manhattan_dist'] =     (train_df['dropoff_longitude'] - train_df['pickup_longitude']).abs() +     (train_df['dropoff_latitude'] - train_df['pickup_latitude']).abs()
    
test_df['Manhattan_dist'] =     (test_df['dropoff_longitude'] - test_df['pickup_longitude']).abs() +     (test_df['dropoff_latitude'] - test_df['pickup_latitude']).abs()    
    
#print(train_df.head())  
#print(test_df.head())  


# In[ ]:


#--- Taken from Part 2 ---
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)
train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)

train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)
train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)
train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)
#print (train_df.head())

#--- Doing the same for the test data excluding dropoff time ---
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

test_df['pickup_month'] = test_df.pickup_datetime.dt.month.astype(np.uint8)
test_df['pickup_day'] = test_df.pickup_datetime.dt.weekday.astype(np.uint8)
test_df['pickup_hour'] = test_df.pickup_datetime.dt.hour.astype(np.uint8)

#print (test_df.head())


# In[ ]:


train_df['trip_duration_mins'] = train_df['trip_duration'] / 60
train_df['trip_duration_hours'] = train_df['trip_duration_mins'] / 60
#print (train_df.head())


# **Removing Outliers**
# 
# **1. Very long trip durations**

# Hourly statitics of trip duration

# In[ ]:


print(max(train_df['trip_duration_hours']))
print(min(train_df['trip_duration_hours']))

print(train_df['trip_duration_hours'].describe())


# Let us remove all the trips that are above 5 hours. 

# In[ ]:


print (train_df[train_df['trip_duration_hours'] > 5].count()['id'])
print (len(train_df))


# In[ ]:


train_df.drop(train_df[train_df.trip_duration_hours > 5].index, inplace=True)
print (len(train_df))


# Now let us view the statistics in the trip duration for hours again

# In[ ]:


print(max(train_df['trip_duration_hours']))
print(min(train_df['trip_duration_hours']))

print(train_df['trip_duration_hours'].describe())


# We can see that the **mean** and **standard deviation** values have decreased.

# **2. Traveling with '0' passengers**

# In[ ]:


print(train_df['passenger_count'].unique())
print(test_df['passenger_count'].unique())
print(test_df[(test_df['passenger_count'] == 0)].count()['id'])


# The test data also contains rides having no passengers. Hence it is not advisable to remove them.

# **3. Extreme latitude and longitude pickups**

# Let us plot the pickup and dropoff latitudes and longitudes

# In[ ]:


plt.plot(train_df['pickup_longitude'], train_df['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight = 'bold')
plt.show()

plt.plot(train_df['dropoff_longitude'], train_df['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight = 'bold')
plt.show()


#  - In both the plots we can see a clutter of points, where majority of
#    the pickups and drop-offs are located.
#  - Every other point is a possible outlier.
#  - Among the outliers the are **3 outstanding** ones, in both the plots.
# 
# Let us first remove the **outstanding** outliers first, manually and visualize the plot again !!

# In[ ]:


train_df = train_df[train_df.pickup_latitude != 51.881084442138672]

train_df = train_df[train_df.pickup_longitude != -121.93334197998048]

train_df = train_df[train_df.dropoff_longitude != -121.93320465087892]

#train_df = train_df[train_df.dropoff_latitude != 32.181140899658203]

plt.plot(train_df['pickup_longitude'], train_df['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight = 'bold')
plt.show()

plt.plot(train_df['dropoff_longitude'], train_df['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight = 'bold')
plt.show()


# The clutter of points has magnified a bit. But we still do have some outliers present.
# 
# Now let us standardize the points are visualize them again.

# In[ ]:


#--- Mean of locations Lats and Longs ---
mean_p_lat = np.mean(train_df['pickup_latitude'])
mean_p_lon = np.mean(train_df['pickup_longitude'])

print (mean_p_lat)
print (mean_p_lon)


# In[ ]:


#--- Standard deviation of pickup & dropoff Lats and Longs ---
std_p_lat = np.std(train_df['pickup_latitude'])
std_p_lon = np.std(train_df['pickup_longitude'])

print (std_p_lat)
print (std_p_lon)


# In[ ]:


min_p_lat = mean_p_lat - std_p_lat
max_p_lat = mean_p_lat + std_p_lat
min_p_lon = mean_p_lon - std_p_lon
max_p_lon = mean_p_lon + std_p_lon

locations = train_df[(train_df.pickup_latitude > min_p_lat) & (train_df.pickup_latitude < max_p_lat) & (train_df.pickup_longitude > min_p_lon) & (train_df.pickup_longitude < max_p_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight = 'bold')
plt.show()


# We have zoomed into Manhattan quite too much!!
# 
# Let's zoom out a little

# In[ ]:


min_p_lat = mean_p_lat - (3 * std_p_lat)
max_p_lat = mean_p_lat + (3 * std_p_lat)
min_p_lon = mean_p_lon - (3 * std_p_lon)
max_p_lon = mean_p_lon + (3 * std_p_lon)
'''
locations = train_df[(train_df.pickup_latitude > min_p_lat) & (train_df.pickup_latitude < max_p_lat) & (train_df.pickup_longitude > min_p_lon) & (train_df.pickup_longitude < max_p_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight = 'bold')
plt.show()
'''


# In[ ]:


min_p_lat = mean_p_lat - (4 * std_p_lat)
max_p_lat = mean_p_lat + (4 * std_p_lat)
min_p_lon = mean_p_lon - (4 * std_p_lon)
max_p_lon = mean_p_lon + (4 * std_p_lon)
'''
locations = train_df[(train_df.pickup_latitude > min_p_lat) & (train_df.pickup_latitude < max_p_lat) & (train_df.pickup_longitude > min_p_lon) & (train_df.pickup_longitude < max_p_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight = 'bold')
plt.show()
'''


# In[ ]:


min_p_lat = mean_p_lat - (10 * std_p_lat)
max_p_lat = mean_p_lat + (10 * std_p_lat)
min_p_lon = mean_p_lon - (10 * std_p_lon)
max_p_lon = mean_p_lon + (10 * std_p_lon)
'''
locations = train_df[(train_df.pickup_latitude > min_p_lat) & (train_df.pickup_latitude < max_p_lat) & (train_df.pickup_longitude > min_p_lon) & (train_df.pickup_longitude < max_p_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight = 'bold')
plt.show()
'''


# I've decided to stick with **5 standard deviations** away from the mean. I will apply the same for the drop-off latitudes and longitudes.

# In[ ]:


min_p_lat = mean_p_lat - (5 * std_p_lat)
max_p_lat = mean_p_lat + (5 * std_p_lat)
min_p_lon = mean_p_lon - (5 * std_p_lon)
max_p_lon = mean_p_lon + (5 * std_p_lon)

locations = train_df[(train_df.pickup_latitude > min_p_lat) & (train_df.pickup_latitude < max_p_lat) & (train_df.pickup_longitude > min_p_lon) & (train_df.pickup_longitude < max_p_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight = 'bold')
plt.show()


# * Comparing size of the two dataframes
# * Creating duplicate dataframe to work with

# In[ ]:


print(len(train_df))
print(len(locations))

#--- making a duplicate copy of the df to work on ---
locations_1 = locations
print(locations_1.head())


# Modeling!!

# In[ ]:


#--- Assigning the target variable ---
labels = train_df['trip_duration']


# In[ ]:


# --- I forgot to convert the categorical variables to numerical variables ---
df_s_f_train = pd.get_dummies(train_df['store_and_fwd_flag'])
df_s_f_test = pd.get_dummies(test_df['store_and_fwd_flag'])

# --- Join the dummy variables to the main dataframe ---
train_df = pd.concat([train_df, df_s_f_train], axis=1)
test_df = pd.concat([test_df, df_s_f_test], axis=1)

# --- Drop the categorical column ---
train_df.drop('store_and_fwd_flag', axis=1, inplace=True)
test_df.drop('store_and_fwd_flag', axis=1, inplace=True)

#print (train_df.head())
#print (test_df.head())


# In[ ]:


train_df = train_df.loc[:,~train_df.columns.duplicated()]
test_df = test_df.loc[:,~test_df.columns.duplicated()]

#print (train_df.head())
#print (test_df.head())


# In[ ]:


train_df.drop('id', axis=1, inplace=True)
#test_df.drop('id', axis=1, inplace=True)

#print (train_df.head())
#print (test_df.head())


# In[ ]:


#train_df['pickup_longitude'] = train_df['pickup_longitude'].round(3)
#train_df['pickup_latitude'] = train_df['pickup_latitude'].round(3)
#train_df['dropoff_longitude'] = train_df['dropoff_longitude'].round(3)
#train_df['dropoff_latitude'] = train_df['dropoff_latitude'].round(3)
#train_df['Haversine_dist'] = train_df['Haversine_dist'].round(3)
#train_df['bearing_dist'] = train_df['bearing_dist'].round(3)
#train_df['Manhattan_dist'] = train_df['Manhattan_dist'].round(3)
#train_df['trip_duration_mins'] = train_df['trip_duration_mins'].round(3)
#train_df['trip_duration_hours'] = train_df['trip_duration_hours'].round(3)


# In[ ]:


print (train_df.isnull().values.any())
print (test_df.isnull().values.any())


# In[ ]:


print(train_df.head())
print(test_df.head())


# 

# In[ ]:


b_train = train_df.drop(['pickup_datetime','dropoff_datetime','dropoff_hour', 'dropoff_month', 'dropoff_day', 'trip_duration', 'trip_duration_mins', 'trip_duration_hours'], 1)
b_label = train_df['trip_duration']
print(b_train.head())
print(b_label.head())

test = test_df
test = test.drop(['pickup_datetime','id'], 1)
print(test.head())


# **Cross Validation** 
# 
# Adapted from [HERE](https://www.kaggle.com/jeru666/stacked-regressions-top-4-on-leaderboard)

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(b_train)
    rmse= np.sqrt(-cross_val_score(model, b_train, b_label, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# **Modeling**
# 
# * LASSO
# 
# 

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso = Lasso(alpha =0.01, random_state=1)


# In[ ]:


#lasso.predict()  


# In[ ]:


#score = rmsle_cv(lasso)
#print("\nLasso score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
#print('DONE!!')


# 

# **Gradient Boosting Regressor** 

# In[ ]:


#--- Setting up and training Gradient Boosting Regressor ---

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=0, loss='ls')
GBR.fit(b_train, b_label)

print (GBR)


# In[ ]:


#--- List of important features for Gradient Boosting Regressor ---

features_list = b_train.columns.values
feature_importance = GBR.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)


# In[ ]:



plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()


# In[ ]:


#--- Predicting Gradient boost result for test data ---
y_GBR = GBR.predict(test)


# In[ ]:



final = pd.DataFrame()
final['id'] = test_df['id']
final['trip_duration'] = y_GBR
final.to_csv('Gradient_Boost_1.csv', index=False)
print('DONE!!')


# **Adaptive Boosting** (Ada Boost gave me a **VERY** bad score on the LB, you can try and see it for yourself!!)

# In[ ]:


#--- Setting up and training Ada boost ---
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

Ada_R = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators = 300, random_state = np.random.RandomState(1))

Ada_R.fit(b_train, b_label)

print (Ada_R)
'''


# In[ ]:


#--- List of important features for Ada Boost ---
'''
features_list = b_train.columns.values
feature_importance = Ada_R.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)
'''


# In[ ]:


'''
plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
'''


# In[ ]:


#--- Predicting Ada boost result for test data ---
#y_Ada = Ada_R.predict(test)


# In[ ]:


'''
final = pd.DataFrame()
final['id'] = test_df['id']
final['trip_duration'] = y_Ada
final.to_csv('Ada_Boost_1.csv', index=False)
print('DONE!!')
'''


# **Random Forest Regressor** (Random Forest performed a lot better !!!)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor  
'''
RF = RandomForestRegressor()
RF.fit(b_train, b_label)

print(RF)
'''


# In[ ]:


#--- List of important features ---
'''
features_list = b_train.columns.values
feature_importance = RF.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)
'''


# In[ ]:


'''
plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
'''


# In[ ]:


#--- Predicting for the test data ---
##test_df = test_df.drop('pickup_datetime', 1)
## = test_df.drop('id', 1)
'''
print(test.head())

Y_pred = RF.predict(test)
'''


# In[ ]:


'''
final = pd.DataFrame()
final['id'] = test_df['id']
final['trip_duration'] = Y_pred
final.to_csv('RF_1.csv', index=False)
print('DONE!!')
'''


# In[ ]:


#print(final.head())


# Random Forest with Cross Validation

# In[ ]:


'''
from sklearn.ensemble import RandomForestRegressor  
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score   

sample_size = locations_1.shape[0]

max_estimators_options=[7]   

print("lets start cross validation")

test_results=[]

cv = KFold(sample_size, n_folds=5,shuffle=True, random_state=123)
result_r2=np.empty([cv.n_folds,len(max_estimators_options)],dtype=float)
CV_stacked=[]

#--- target columns to remove 
cols = [ 'dropoff_month', 'dropoff_day', 'dropoff_hour', 'trip_duration', 'trip_duration_mins', 'trip_duration_hours']
x_train = train_df.drop(cols, 1)

#--- assign target column to separate variable---
labels = train_df["trip_duration"].copy()

count=0
for alp in max_estimators_options:
    params = {'n_estimators':400, 'max_depth': 7, 'min_samples_split':50,"max_features":75,
          'random_state':0,"verbose":1, 'n_jobs' : -1 }
    model=RandomForestRegressor(**params)
    result=[]
    actual=[]
    pred_CV=[]

    for traincv, testcv in cv: 
        X_train = x_train.iloc[traincv]            #--- all independent variables
        labels_train = labels.iloc[traincv]        #--- dependent variable -> trip_duration
        
        X_CV = x_train.iloc[testcv]
        labels_test= labels.iloc[testcv]

        test_data = test_df.copy()
        
        final_pred = model.fit(X_train, labels_train).predict(X_CV)
        val_score = r2_score(labels_test,final_pred)
        
        actual+=labels_test.tolist()
        pred_CV+=final_pred.tolist()
        
        test_pred = model.predict(test_data)
        test_results.append(test_pred)
        #val_score =r2_score(data_set.ix[testcv,"y"],final_pred)
        #print(val_score)
        result.append(val_score)
    result_r2[:,count]=result
    stacked_CV = r2_score(actual, pred_CV)
    CV_stacked.append(stacked_CV)
    print(count)
    print(result)
    count=count+1

mean = result_r2.mean(axis=0)
std = result_r2.std(axis=0)
###fine best estimator size
print(mean)
print(std)
print(CV_stacked)
'''


# Normally distributing the output variable using log transformation

# In[ ]:


plt.hist(np.log(train_df['trip_duration']+25), bins = 25)


# **WORK STILL IN PROGRESS
# 
# **MANY MORE MODELS WILL BE BUILT AND TUNED**
# 
# STAY TUNED!!!**
