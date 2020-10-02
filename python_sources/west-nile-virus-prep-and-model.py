#!/usr/bin/env python
# coding: utf-8

# In[90]:


# Data Handling
import pandas as pd
import numpy as np
import math
import scipy.stats as sps
#from scipy import stats, integrate
from time import time


# sklearn and models
from sklearn import preprocessing, ensemble, metrics, feature_selection, model_selection, pipeline
import xgboost as xgb

#plotting and display
from IPython.display import display
from matplotlib import pyplot


# ### Load Data

# In[91]:


# create date parser
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

# create data type converters
dtype_map_weather = dict(Station = 'str')
dtype_map_test_train = dict(Block = 'str', Street = 'str')

# read data into PANDAS DataFrames with date parsing
test = pd.read_csv('../input/test.csv', parse_dates=['Date'], date_parser=dateparse, dtype= dtype_map_test_train)
train = pd.read_csv('../input/train.csv', parse_dates=['Date'], date_parser=dateparse, dtype= dtype_map_test_train)
weather = pd.read_csv('../input/weather.csv', parse_dates=['Date'], date_parser=dateparse, dtype= dtype_map_weather)
sample_sub = pd.read_csv('../input/sampleSubmission.csv')


# In[92]:


print('Train')
display(train.info())

print('Test')
display(test.info())


# In[93]:


print('Weather')
display(weather.info())


# ### Select Columns

# In[94]:


# weather
weather_exclude = ['Dewpoint', 'WetBulb', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'StnPressure',
                 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','DewPoint']
weather_cols = [col for col in weather.columns if col not in weather_exclude]
weather = weather[weather_cols]


# train
train_exclude = ['Address', 'AddressNumberAndStreet', 'AddressAccuracy', 'NumMosquitos']
train_cols = [col for col in train.columns if col not in train_exclude]
train = train[train_cols]

# test
test_exclude = ['Address', 'AddressNumberAndStreet', 'AddressAccuracy', 'Id']
test_cols = [col for col in test.columns if col not in test_exclude]
test = test[test_cols]


# In[95]:


weather.info()


# In[96]:


print('Weather')
display(weather.head())

print('Train')
display(train.head())


# In[97]:


# what species have been detected (note that according to the CDC each
# of these species can carry WNV)
set(train.Species)


# In[98]:


# does this correspond to the test set
set(test.Species)
# it looks like there is another category


# In[99]:


train.groupby('Species').sum().WnvPresent


# ### Examine and Handle missing Data

# **What is 'T' and 'M'?**
# - From http://www.nws.noaa.gov/om/csd/info/NOWdata/FAQ.php
# 
# > "M" stands for "Missing". Data for an element will be missing if the primary sensor for that weather element is inoperable (e.g., has an outage) or malfunctioning (e.g., producing errant data) AND any collocated backup sensor is also inoperable or malfunctioning. "T" stand for "Trace". This is a small amount of precipitation that will wet a raingage but is less than the 0.01 inch measuring limit.
# 
# The Precipitation Total column is the only column in the retained data that can contain this value. The value has leading whitespace so we should strip prior to counting.

# In[100]:


# strip whitespace
weather.PrecipTotal = weather.PrecipTotal.str.strip()


# In[101]:


miss_weather = ['M', '-']
trace_weather = ['T']


# In[102]:


cols_not_date = [col for col in weather.columns if col != 'Date']


# In[103]:


weather[cols_not_date].apply(pd.value_counts, axis=1)[miss_weather + trace_weather].fillna(0).sum()


# In[104]:


# Both stations
check = weather[cols_not_date].apply(pd.value_counts, axis=0).fillna(0)
check.loc[['M', '-', 'T']]


# In[105]:


# Station 1
check_stat1 = weather[cols_not_date][weather.Station == '1'].apply(pd.value_counts, axis=0).fillna(0)
check_stat1.loc[['M', '-', 'T']]


# In[106]:


# Station 2
check_stat2 = weather[cols_not_date][weather.Station == '2'].apply(pd.value_counts, axis=0).fillna(0)
check_stat2.loc[['M', '-', 'T']]


# In[107]:


# Both stations
check.loc[['M', '-', 'T']]/(len(weather)) * 100


# In[108]:


# Station 1
check_stat1.loc[['M', '-', 'T']]/(len(weather)) * 100


# In[109]:


# Station 2()
check_stat2.loc[['M', '-', 'T']]/(len(weather)) * 100


# In[110]:


weather = weather.replace('M', np.NaN)
weather = weather.replace('-', np.NaN)
weather = weather.replace('T', 0.005) # very small amounts of rain can impact mosquito hatches
weather.Tmax = weather.Tmax.fillna(method = 'ffill')
weather.Tmin = weather.Tmin.fillna(method = 'ffill')
weather.Depart = weather.Depart.fillna(method = 'ffill')
weather.Heat = weather.Heat.fillna(method = 'ffill')
weather.Cool = weather.Cool.fillna(method = 'ffill')
weather.PrecipTotal = weather.PrecipTotal.fillna(method = 'ffill')


# In[111]:


# convert datatpypes

to_numeric = ['Tmax','Tmin','Tavg', 'Depart', 'Heat', 'Cool', 'PrecipTotal']

for col in to_numeric:
    weather[col]= pd.to_numeric(weather[col])


# In[112]:


weather.Sunrise = weather.Sunrise.fillna(method = 'ffill')
weather.Sunset = weather.Sunset.fillna(method = 'ffill')


# In[113]:


# sunset has entries where instead of incrementing to the next hour after xx59 it incremented to xx60
# This causes an exception, let's take a look
counter = 0
tracker = []
for index, val in enumerate(weather.Sunset):
    try:
        pd.to_datetime(val, format = '%H%M').time()
    except:
        counter += 1
        tracker.append((index, val, val[2:], counter))

print(tracker[-1])

# there are 48 exceptions


# In[114]:


# let's deal with this by decrmenting by 1 for each invalid instance
weather.Sunset = weather.Sunset.replace('\+?60', '59', regex = True)


# In[115]:


# time conversion lambda function
time_func = lambda x: pd.Timestamp(pd.to_datetime(x, format = '%H%M'))


# In[116]:


weather.Sunrise = weather.Sunrise.apply(time_func)


# In[117]:


weather.Sunset = weather.Sunset.apply(time_func)


# In[118]:


# what is the range of values for sunrise and sunset (in hours)
minutes= (weather.Sunset - weather.Sunrise).astype('timedelta64[m]')


# In[119]:


hours = minutes/60


# In[120]:


set(np.round(hours.values))


# In[121]:


#create a DayLength column with minute level precsion
weather['DayLength_MPrec'] = (weather.Sunset - weather.Sunrise).astype('timedelta64[m]')/60


# In[122]:


#create a DayLength column with rounded to the nearest hour
weather['DayLength_NearH'] = np.round(((weather.Sunset - weather.Sunrise).astype('timedelta64[m]')/60).values)


# In[123]:


# length of night with minute level precision
weather['NightLength_MPrec']= 24.0 - weather.DayLength_MPrec


# In[124]:


# lenght of night rounded to nearest hour
weather['NightLength_NearH']= 24.0 - weather.DayLength_NearH


# In[125]:


# function to calculate sunset and sunrise times in hours
hours_RiseSet_func = lambda x: x.minute/60.0 + float(x.hour)


# In[126]:


# sunrise in hours
weather['Sunrise_hours'] = weather.Sunrise.apply(hours_RiseSet_func)


# In[127]:


# sunset in hours
weather['Sunset_hours'] = weather.Sunset.apply(hours_RiseSet_func)


# In[128]:


mean_func = lambda x: x.mean()

blend_cols = ['Tmax', 'Tmin', 'Depart' ,'Heat', 'Cool', 'PrecipTotal']


# In[129]:


blended_cols= ['blended_' + col for col in blend_cols]


# In[130]:


station_1 = weather[blend_cols][weather.Station == '1']
station_2 = weather[blend_cols][weather.Station == '2']


# In[131]:


station_blend = pd.DataFrame((station_1.values + station_2.values)/2, columns= blended_cols)


# In[132]:


extract_2 = weather[weather.Station == '2'].reset_index(drop = True)
extract_2.head()


# In[133]:


extract_1 = weather[weather.Station == '1'].reset_index(drop = True)
extract_1.head()


# In[134]:


joined_1 = extract_1.join(station_blend)
joined_2 = extract_2.join(station_blend)


# In[135]:


weather_blend = pd.concat([joined_1, joined_2])


# In[136]:


weather_blend.info()


# ### Create Month and Day columns

# In[137]:


month_func = lambda x: x.month
day_func= lambda x: x.day
day_of_year_func = lambda x: x.dayofyear
week_of_year_func = lambda x: x.week

# train
train['month'] = train.Date.apply(month_func)
train['day'] = train.Date.apply(day_func)
train['day_of_year'] = train.Date.apply(day_of_year_func)
train['week'] = train.Date.apply(week_of_year_func)

# test
test['month'] = test.Date.apply(month_func)
test['day'] = test.Date.apply(day_func)
test['day_of_year'] = test.Date.apply(day_of_year_func)
test['week'] = test.Date.apply(week_of_year_func)

### Create integer latitude and longitude columns

train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)
# In[138]:


train.describe()


# In[139]:


test.describe()


# In[140]:


# remove sunrise and sunset since we have extracted critical information into other fields
weather_blend = weather_blend.drop(['Sunrise', 'Sunset'], axis= 1)


# ### Merge Data

# In[141]:


train = train.merge(weather_blend, on='Date')
test = test.merge(weather_blend, on='Date')


# ### Inspect DFs

# In[142]:


weather_blend.ix[:,:12].describe()


# In[143]:


weather_blend.ix[:,12:].describe()


# In[144]:


train.describe()


# ### Handle Weather Stations 1

# In[145]:


# columns to write
cols_to_write = [col for col in train.columns if col != 'Date'] # exclude 'Date'


# In[146]:


# split the data into two dataframes by station

train_station_1= train[train.Station == '1']
train_station_2= train[train.Station == '2']

test_station_1= test[test.Station == '1']
test_station_2= test[test.Station == '2']


# In[147]:


# export to JSON for external use
#train_station_1.to_json('train_station_1.json')
#train_station_2.to_json('train_station_2.json')
#train.to_json('train.json')

# epxort to csv for external use
#train_station_1.to_csv('train_station_1.csv')
#train_station_2.to_csv('train_station_2.csv')
train.to_csv('train.csv')


# # Prepare Data Set for Model Building

# In[148]:


# set up a merge for stations 1 and 2
# keep unique cols from station 2
keep_cols = ['Date', u'Tmax', u'Tmin', u'Tavg',u'PrecipTotal']
train_station_2 = train_station_2[keep_cols]
test_station_2 = test_station_2[keep_cols]

# rename cols with prefix
prefix_s2 = 'stat_2_'
rename_cols_s2 = [prefix_s2 + col for col in train_station_2.columns]
train_station_2.columns = rename_cols_s2
test_station_2.columns = rename_cols_s2


# In[149]:


# drop cols from station 1 that won't be used in model
drop_cols = ['Heat', 'Cool', 'Depart', 'NightLength_MPrec', 'NightLength_NearH',
            'blended_Depart', 'blended_Heat', 'blended_Cool']

train_station_1 = train_station_1.drop(drop_cols, axis= 1)
test_station_1 = test_station_1.drop(drop_cols, axis= 1)   


# In[150]:


# raname uniqe station 1 columns
prefix_s1 = 'stat_1_'
rename_cols_s1 = [prefix_s1 + col for col in keep_cols]
cols_to_rename= [col for col in train_station_1.columns if col in keep_cols]

# setup name mapping
s1_name_map = dict(zip(cols_to_rename, rename_cols_s1))

train_station_1 = train_station_1.rename(columns= s1_name_map)
test_station_1 = test_station_1.rename(columns= s1_name_map)


# In[151]:


# concat (outer join)
train_station_1 =  train_station_1.reset_index(drop= True)
train_station_2 = train_station_2.reset_index(drop = True)
train_merge = pd.concat([train_station_1, train_station_2], axis= 1)

test_station_1 =  test_station_1.reset_index(drop= True)
test_station_2 = test_station_2.reset_index(drop = True)
test_merge = pd.concat([test_station_1, test_station_2], axis= 1)


# ### Create Dummies from Categorical Variables

# In[152]:


train_merge.columns


# In[153]:


test_merge.columns


# In[154]:


# get label
labels = train_merge.pop('WnvPresent').values


# In[155]:


# remove dates
train_merge = train_merge.drop(['stat_1_Date', 'stat_2_Date'], axis = 1)

test_merge = test_merge.drop(['stat_1_Date', 'stat_2_Date' ], axis = 1)


# In[156]:


# add lat and long integer columns

train_merge['Lat_int'] = train_merge.Latitude.astype(int)
train_merge['Long_int'] = train_merge.Longitude.astype(int)

test_merge['Lat_int'] = test_merge.Latitude.astype(int)
test_merge['Long_int'] = test_merge.Longitude.astype(int)


# In[157]:


# Create dummies from the categorical species, block, trap, and streetname
train_merge = pd.get_dummies(train_merge, columns= ['Species'])
train_merge = pd.get_dummies(train_merge, columns= ['Block'])
train_merge = pd.get_dummies(train_merge, columns= ['Street'])
train_merge = pd.get_dummies(train_merge, columns= ['Trap'])

test_merge = pd.get_dummies(test_merge, columns= ['Species'])
test_merge = pd.get_dummies(test_merge, columns= ['Block'])
test_merge = pd.get_dummies(test_merge, columns= ['Street'])
test_merge = pd.get_dummies(test_merge, columns= ['Trap'])


# In[158]:


#train_merge= train_merge.drop(['Street', 'Trap', 'Station'], axis= 1)
#test_merge= test_merge.drop(['Street', 'Trap', 'Station'], axis= 1)

train_merge= train_merge.drop('Station', axis= 1)
test_merge= test_merge.drop('Station', axis= 1)


# In[159]:


#drops= ['Block', 'Street', 'Trap', 'Latitude', 'Longitude']

#train_merge= train_merge.drop(drops, axis= 1)
#test_merge= test_merge.drop(drops, axis= 1)


# In[160]:


len(train_merge.columns)


# In[161]:


len(test_merge.columns)


# In[162]:


unique_test_cols = [col for col in test_merge.columns if col not in train_merge.columns]


# In[163]:


test_merge= test_merge.drop(unique_test_cols, axis= 1)


# In[164]:


# epxort to csv for external use
#train_merge.to_csv('train_merge.csv')
#train_merge.to_csv('test_merge.csv')


# In[165]:


clf = ensemble.RandomForestClassifier(n_estimators=1000, min_samples_split= 2, random_state= 42)
clf.fit(train_merge, labels)


# In[166]:


# create predictions and submission file
predictions_randfor = clf.predict_proba(test_merge)[:,1]


# AUC for Random forest models on private test data (private leaderboard) tends to fall in the range of 0.655 and of 0.687

# In[167]:


# fit model no training data
xgbc = xgb.XGBClassifier(seed= 42)
xgbc.fit(train_merge, labels)
# feature importance
#print(xgb.feature_importances_)

# plot feature importance
fig, ax = pyplot.subplots(figsize=(10, 15))
xgb.plot_importance(xgbc, ax=ax)
#pyplot.show()


# In[168]:


xgbc.get_fscore()


# In[169]:


# feature importance
xgbc.get_fscore()
#print(xgbc.feature_importances_)


# ## Sandbox Setup "Validation" from training set
# 
# - Strategy:
#     + break train merge 67, 33 call train_split, val_split
#     + fit to train_split
#     + feature selection and cross val (tuning) on val_split

# In[170]:


def calc_roc_auc(y, predict_probs):
    
    """
    Function accepts labels (matrix y) and predicted probabilities
    Function calculates fpr (false positive rate), tpr (true postivies rate), thresholds and auc (area under
    the roc curve)
    Function returns auc
    """
    fpr, tpr, thresholds = metrics.roc_curve(y, predict_probs)
    roc_auc = metrics.auc(fpr, tpr)
    
    return roc_auc


# In[171]:


train_split, val_split, label_train_split, label_val_split = model_selection.train_test_split(train_merge, 
                                      labels, test_size = 0.33, random_state = 42, stratify= labels)


# In[172]:


def select_features_by_importance_threshold(model, X_train, y_train, selection_model, X_test, y_test,
                                           minimum = False):

    # Fit model using each importance as a threshold
    if minimum:
        thresholds= np.unique(model.feature_importances_[model.feature_importances_ > minimum])
        # include 0 for all features
        thresholds = np.insert(thresholds, 0, 0.)
    else:
        thresholds= np.unique(model.feature_importances_)
        
    
    print(thresholds)
    for thresh in thresholds:
	    # select features using threshold
        selection = feature_selection.SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
	    # train model
        selection_model = selection_model
        selection_model.fit(select_X_train, y_train)
	    # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict_proba(select_X_test)[:,1]
        predictions = y_pred
        #predictions = [round(value) for value in y_pred]
        auc = calc_roc_auc(y_test, predictions)
        print("Thresh=%.3f, n=%d, AUC: %.2f%%" % (thresh, select_X_train.shape[1], auc))


# ```select_features_by_importance_threshold(xgb, train_split, label_train_split,
#                                        XGBClassifier(seed= 42), val_split, label_val_split)```

# ```python
# [ 0.          0.00173913  0.00347826  0.00521739  0.00695652  0.00869565
#   0.01043478  0.01217391  0.01565217  0.0226087   0.02434783  0.02782609
#   0.03130435  0.03304348  0.03478261  0.04869565  0.05043478  0.0626087
#   0.1373913   0.19652174]
# 
# Thresh=0.000, n=358, AUC: 0.83%
# Thresh=0.002, n=56, AUC: 0.83%
# Thresh=0.003, n=48, AUC: 0.83%
# Thresh=0.005, n=38, AUC: 0.83%
# Thresh=0.007, n=31, AUC: 0.83%
# Thresh=0.009, n=26, AUC: 0.83%
# Thresh=0.010, n=20, AUC: 0.83%
# Thresh=0.012, n=18, AUC: 0.83%
# Thresh=0.016, n=16, AUC: 0.83%
# Thresh=0.023, n=14, AUC: 0.83%
# Thresh=0.024, n=12, AUC: 0.83%
# Thresh=0.028, n=11, AUC: 0.83%
# Thresh=0.031, n=9, AUC: 0.82%
# Thresh=0.033, n=7, AUC: 0.82%
# Thresh=0.035, n=6, AUC: 0.82%
# Thresh=0.049, n=5, AUC: 0.82%
# Thresh=0.050, n=4, AUC: 0.82%
# Thresh=0.063, n=3, AUC: 0.81%
# Thresh=0.137, n=2, AUC: 0.80%
# Thresh=0.197, n=1, AUC: 0.78%
# ```

# ```select_features_by_importance_threshold(clf, train_split, label_train_split,
#                                        ensemble.RandomForestClassifier(random_state= 42), val_split, 
#                                         label_val_split, minimum= 3.0e-3 )```

# ```python
# [ 0.          0.00301567  0.00305084  0.00307623  0.00308812  0.00311346
#   0.00311898  0.00315303  0.00315906  0.00319348  0.00325467  0.00329564
#   0.00334669  0.00338802  0.00340644  0.00345389  0.00356917  0.00366348
#   0.00369682  0.00382194  0.00391003  0.00401982  0.00407744  0.00437668
#   0.00472272  0.00554023  0.00556184  0.00588063  0.00931316  0.00969602
#   0.01031569  0.01153792  0.01159319  0.01641062  0.01707728  0.0175769
#   0.01763611  0.01786111  0.01793691  0.01830143  0.0185359   0.01863073
#   0.0191287   0.02313415  0.02382371  0.02395312  0.02430192  0.03390532
#   0.04137335  0.04780691  0.06432936  0.07285084]
# 
# Thresh=0.000, n=358, AUC: 0.72%
# Thresh=0.003, n=51, AUC: 0.70%
# Thresh=0.003, n=50, AUC: 0.74%
# Thresh=0.003, n=49, AUC: 0.71%
# Thresh=0.003, n=48, AUC: 0.72%
# Thresh=0.003, n=47, AUC: 0.70%
# Thresh=0.003, n=46, AUC: 0.71%
# Thresh=0.003, n=45, AUC: 0.72%
# Thresh=0.003, n=44, AUC: 0.72%
# Thresh=0.003, n=43, AUC: 0.70%
# Thresh=0.003, n=42, AUC: 0.71%
# Thresh=0.003, n=41, AUC: 0.71%
# Thresh=0.003, n=40, AUC: 0.70%
# Thresh=0.003, n=39, AUC: 0.71%
# Thresh=0.003, n=38, AUC: 0.71%
# Thresh=0.003, n=37, AUC: 0.70%
# Thresh=0.004, n=36, AUC: 0.71%
# Thresh=0.004, n=35, AUC: 0.72%
# Thresh=0.004, n=34, AUC: 0.71%
# Thresh=0.004, n=33, AUC: 0.72%
# Thresh=0.004, n=32, AUC: 0.70%
# Thresh=0.004, n=31, AUC: 0.71%
# Thresh=0.004, n=30, AUC: 0.69%
# Thresh=0.004, n=29, AUC: 0.70%
# Thresh=0.005, n=28, AUC: 0.70%
# Thresh=0.006, n=27, AUC: 0.71%
# Thresh=0.006, n=26, AUC: 0.73%
# Thresh=0.006, n=25, AUC: 0.72%
# Thresh=0.009, n=24, AUC: 0.71%
# Thresh=0.010, n=23, AUC: 0.73%
# Thresh=0.010, n=22, AUC: 0.72%
# Thresh=0.012, n=21, AUC: 0.71%
# Thresh=0.012, n=20, AUC: 0.71%
# Thresh=0.016, n=19, AUC: 0.70%
# Thresh=0.017, n=18, AUC: 0.72%
# Thresh=0.018, n=17, AUC: 0.72%
# Thresh=0.018, n=16, AUC: 0.73%
# Thresh=0.018, n=15, AUC: 0.73%
# Thresh=0.018, n=14, AUC: 0.73%
# Thresh=0.018, n=13, AUC: 0.72%
# Thresh=0.019, n=12, AUC: 0.69%
# Thresh=0.019, n=11, AUC: 0.70%
# Thresh=0.019, n=10, AUC: 0.71%
# Thresh=0.023, n=9, AUC: 0.68%
# Thresh=0.024, n=8, AUC: 0.70%
# Thresh=0.024, n=7, AUC: 0.69%
# Thresh=0.024, n=6, AUC: 0.66%
# Thresh=0.034, n=5, AUC: 0.60%
# Thresh=0.041, n=4, AUC: 0.60%
# Thresh=0.048, n=3, AUC: 0.60%
# Thresh=0.064, n=2, AUC: 0.64%
# Thresh=0.073, n=1, AUC: 0.51%
# ```

# In[173]:


train_merge.shape


# In[174]:


# Set a minimum threshold of 0.023
sfm = feature_selection.SelectFromModel(xgbc, threshold=0.023, prefit= True)
sfm_train= sfm.transform(train_merge)
n_features = sfm_train.shape[1]
print(n_features)


# In[175]:


# initialize and fit model
xgb_clf= xgb.XGBClassifier(seed= 42)
xgb_clf.fit(sfm_train, labels)


# In[176]:


sfm_test = sfm.transform(test_merge)
predictions_xgb = xgb_clf.predict_proba(sfm_test)[:,1]


# In[177]:


# plot single tree
xgb.plot_tree(xgb_clf, rankdir= 'LR')
pyplot.show()


# ### Address overfittting through early stopping

# In[178]:


X_train= train_split
X_test= val_split
y_train= label_train_split
y_test= label_val_split
model= xgb.XGBClassifier(seed= 42)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=True)


# In[179]:


results = model.evals_result()
print(results)


# In[180]:


model.fit(X_train, y_train, eval_metric=["auc", "logloss", "error"], eval_set=eval_set)
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# plot auc
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
pyplot.ylabel('AUC')
pyplot.title('XGBoost AUC by Epoch')
pyplot.show()

# plot logloss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Logloss')
pyplot.title('XGBoost Logloss by Epoch')
pyplot.show()

# plot error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Error')
pyplot.title('XGBoost Error by Epoch')
pyplot.show()


# In[181]:


eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["auc"], eval_set=eval_set, early_stopping_rounds=10)
results = model.evals_result()
print(results)


# xgb.cv(, num_round, nfold=5,
#        metrics={'error'}, seed = 0,
#        callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

# In[182]:


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[183]:


#n_estimators_dist= np.random.randint(1, 500)# number of trees, could use a discrete list or np.random.exponential(scale=0.1, size= 100)
#colsample_bytree_dist= np.random.uniform(0.2,0.6) # should be 0.3 - 0.5
#max_depth_dist = np.random.randint(2, 12) # typical values 3 - 10
#learning_rate_dist= np.random.uniform(0.01, 0.3) # default 0.3, typical values 0.01 - 0.2

#learning_rate_dist= scipy.stats.expon(scale=100)
#learning_rate_dist= 10. ** np.arange(-3, -2)
n_estimators_dist= sps.randint(1, 300)
learning_rate_dist = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]


# In[184]:


#cv = model_selection.StratifiedShuffleSplit(n_splits = 10, random_state = 42)  

param_dist = dict(learning_rate= learning_rate_dist, n_estimators= n_estimators_dist) 

# run randomized search
n_iter_search = 20
random_search = model_selection.RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring= 'roc_auc')

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


# In[185]:


sample_sub['WnvPresent'] = predictions_xgb
sample_sub.to_csv('sub_xgb.csv', index=False)

#sample_sub['WnvPresent'] = predictions_randfor
#sample_sub.to_csv('sub_randfor.csv', index=False)


# predictions.to_csv('predictions.csv', header= True, index_label= 'id')
