#!/usr/bin/env python
# coding: utf-8

# # Filling Missing Values with Machine Learning 
# 
# * If you inspect the missing rows int the weather data, there are many instances where there is a missing value on one element of the row but not in the other elements of the same row. Example: value at column A in Row 5 is missing, but the other columns in row 5 are not missing. 
# * Using this pattern, we can predict on the missing column using the non missing columns.
# 

# In[ ]:


import pandas as pd
import numpy as np 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics, sklearn.ensemble
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from pathlib import Path


# # Loading data
# In this Kernal, we'll be filling the missing values in weather_train, also by combining values from the weather.test.csv. To predict for test weather data, just swap the two weather paths below.

# In[ ]:


data_dir = Path('../input/ashrae-energy-prediction')
# this is for train. Just switch below to paths to fill missing data in test weather
weather = pd.read_csv(data_dir/'weather_train.csv', parse_dates=['timestamp'], 
                      dtype={'site_id': np.uint16})
weather_test = pd.read_csv(data_dir/'weather_test.csv', parse_dates=['timestamp'], 
                      dtype={'site_id': np.uint16})

weather.head()


# # Adding in missing the timestamps
# After merging weather and train, there are many additional missing values created, because the weather file skips an hour's reading here and there which the train set contains (weather merges on train timestamp). Will be better if we can fill in these values too, whilst filling up the others with ML

# In[ ]:


# running df that will be appended to 
running_batch = weather[weather['site_id'] == 1].set_index('timestamp').resample('h').mean().copy()
running_batch['site_id'] = 1
# for each site, resampling weather every one hour
for site in weather['site_id'].unique():
    if site == 1:
        continue

    site_batch = weather[weather['site_id'] == site].set_index('timestamp').resample('1h').mean()   
    site_batch['site_id'] = site
    running_batch = running_batch.append(site_batch)
print(running_batch.isna().sum())
print('Weather has increased by {} samples'.format(len(running_batch)-len(weather)))
    


# # Creating time features

# In[ ]:


weather = running_batch.reset_index(level=0).copy()
weather = weather.sort_values(['timestamp'])

weather['hour']=weather['timestamp'].apply(lambda x: x.hour).astype(np.uint8)
weather['month'] = weather['timestamp'].apply(lambda x: x.month).astype(np.uint8)
weather['day']=weather['timestamp'].apply(lambda x: x.day).astype(np.uint8)
weather['year']=(weather['timestamp'].apply(lambda x: x.year) - 2015).astype(np.uint8)


weather_test['hour']=weather_test['timestamp'].apply(lambda x: x.hour).astype(np.uint8)
weather_test['month'] = weather_test['timestamp'].apply(lambda x: x.month).astype(np.uint8)
weather_test['day']=weather_test['timestamp'].apply(lambda x: x.day).astype(np.uint8)
weather_test['year']=(weather_test['timestamp'].apply(lambda x: x.year) - 2015).astype(np.uint8)


# In[ ]:


corr = weather.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# # Functions to train model on nonmissing data and predict on missing data
# 
# Procedure of fill_col with LightGBM:
# * A target and feature list is provided. After which, train and test sets are created.
# * Train sets are created using Non missing values of the features and target. 
# * The rows in which feature columns have missing values, those rows are not considered in train or test
# * Rows in which features are present, but not the target, that becomes test set
# 
# 
# If you have more compute available, you can experiment with adding larger and smaller n_estimators and min_child_samples to the grid search. Because some trainsets, in the process of filling, can contain few features and less data. Grid search would most likely choose lower n_est and min_child_samples for smaller trainsets.
# 

# In[ ]:


# label encode columns [list] of df 
def encode_df(df, columns):
        onehot_features = []
        for i, column in enumerate(columns):
            lenc = LabelEncoder()
            df[column] = lenc.fit_transform(df[column].values)
            
        return df
    
# get data from X columns test data 
def extract_x_from_test(df, cols):
    df = df[cols]
    df.dropna(inplace=True)
    return df 

# inputs: Dataframe, list, list, string. Fills rows in target column that are missing, by
# training on non missing values of target and features 
# features only include continous features
def fill_col(df, X_from_testset, features, target,  cat_features_names):    
    
    # updating feature set names 
    features += cat_features_names
    # onehot encoding cat columns 
    train_df  = df.copy()
    train_df = encode_df(train_df, cat_features_names)
    # extracting non-missing features and missing target rows for test 
    x_test =  train_df[(~train_df[features].isna()).all(axis=1)][train_df[target].isna()][features]
    # extracting non-missing features and target rows for train
    x_train =  train_df[(~train_df[features].isna()).all(axis=1)][~train_df[target].isna()]
    if len(X_from_testset) !=0:
        a = X_from_testset[features+[target]].dropna()
        x_train = x_train.append(a)
    
    # data from test set     
    # dataset specs 
    print('Training on {0:.5f} fraction, {1:} samples'.format(len(x_train)/len(train_df), len(x_train)))
    print('Filling up {0:.5f} fraction, {1:} samples'.format(len(x_test)/len(train_df),len(x_test)))
    
    if len(x_train) == 0 or len(x_test) == 0:
        print('Cannot fill any missing values.')
        return df

    y_train = x_train[target]
    x_train=x_train[features]

    # grid search cv
    param_grid = {'num_leaves': [15], 'learning_rate':[0.25],
                 'min_child_samples':[70], 'n_estimators':[45],
                  'lambda_l2':[20], 'max_bin':[50], 'objective':['regression']}
    
    gbm = LGBMRegressor(categorical_features=cat_features_names)
    gc = sklearn.model_selection.GridSearchCV(gbm, param_grid=param_grid, cv=4, verbose=1,
        n_jobs=6, refit='r2', scoring=['neg_mean_absolute_error', 'r2'], 
                                              return_train_score=True)
    # fits best scoring model 
    gc.fit(x_train, y_train)
    
    train_preds2 = gc.predict(X=x_train)
    test_preds = gc.predict(X=x_test)
    df.at[x_test.index, target] = test_preds
    
                  
    results = pd.DataFrame.from_dict(gc.cv_results_)
    results = results.sort_values(['rank_test_r2'])
    metrics=['mean_test_r2', 'mean_test_neg_mean_absolute_error', 'mean_train_r2', 'mean_train_neg_mean_absolute_error' ]
    eval_results = results.iloc[0][metrics]
    
    print(gc.best_params_)
    print(eval_results)

    return df


# In[ ]:


print('Missing values at start')
print(weather.isna().sum())


# # Fitting, predicting on missing values 
# 
# * Target -> Col we want to fillna
# * Features -> Features we want to use to predict on Target

# In[ ]:


target = 'dew_temperature'
features = ['air_temperature', 'hour', 'month', 'year', 'day']
cat_features = ['site_id']

weather_f1 = fill_col(weather.copy(), weather_test.copy(), features, target, cat_features)  


# In[ ]:


target = 'cloud_coverage'
features=['dew_temperature', 'hour', 'month', 'wind_speed', 'year', 'day']
cat_features = ['site_id']
weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  
weather_f1.isna().sum()


# In[ ]:


target = 'precip_depth_1_hr'
features=['dew_temperature', 'hour', 'month', 'wind_speed', 'cloud_coverage',
          'year', 'day']

cat_features = ['site_id']
weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  
weather_f1.isna().sum()


# In[ ]:


target = 'sea_level_pressure'
features=['air_temperature', 'hour', 'month', 'wind_speed', 'cloud_coverage', 'precip_depth_1_hr', 
         'wind_direction', 'year', 'day']
cat_features = ['site_id']
weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  
weather_f1.isna().sum()


# In[ ]:


# predicting on due temperature again with missing values filled in
target = 'dew_temperature'
features=['hour', 'month', 'wind_speed', 'cloud_coverage', 'precip_depth_1_hr', 'year', 'day']
cat_features = ['site_id']
weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  
weather_f1.isna().sum()


# In[ ]:


target = 'wind_direction'
features=['hour', 'month', 'wind_speed', 'cloud_coverage', 
          'precip_depth_1_hr', 'dew_temperature', 'year', 'day']
cat_features = ['site_id']
weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  


# In[ ]:


weather_f1.isna().sum()


# In[ ]:


target = 'wind_direction'
features=['hour', 'month', 'year', 'day']
cat_features = ['site_id']
weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  


# # Filling Nans with a mean filter
# * Value of missing value at position i, is `(val[i-1] + val[i+1])/2`. Window size used is three. 

# In[ ]:


# getting df series with col names and if they cocntain missing values 
cols_ismissing = weather_f1.isna().any(axis=0).reset_index(level=0)
# getting missing column names 
missing_cols = cols_ismissing[cols_ismissing[0] == True]['index'].values
weather_f1 = weather_f1.sort_values(['timestamp'])
weather_f2 = weather_f1.copy()

for site_id in weather_f1['site_id'].unique():
    df = weather_f1[weather['site_id']==site_id].copy()
    if df.isna().any(axis=0).sum() == 0:
        continue
        
    weather_f2.at[df.index, missing_cols] = (df[missing_cols].fillna(method='bfill',limit =1) + 
                                             df[missing_cols].fillna(method='ffill', limit =1))/2
    
weather_f2.isna().sum()


# In[ ]:



previous_targets = []
for target in ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
                'sea_level_pressure', 'wind_direction','wind_speed']:
    print('Filling ', target)
    features=['hour', 'month', 'year'] + previous_targets
    cat_features = ['site_id', 'day'] 

    weather_f2 = fill_col(weather_f2.copy(), weather_test.copy(), features, target, cat_features)
    if target not in ['sea_level_pressure', 'wind_direction']:
        previous_targets.append(target)


# In[ ]:


weather_f2.isna().sum()


# In[ ]:


weather_f2.drop(['day', 'month', 'hour', 'year'], axis=1, inplace=True)
print(weather_f2.columns)
assert len(weather) == len(weather_f2)
weather_f2.to_csv('weather_train_filled.csv', index=False)


# # Conclusion:
# * The performance of filling certain weather items are good and some are not great (eg. wind direction has mae of ~80 degrees). Prediction of precip_depth_1_hr had the worst performance with an r2 of 0.0399225 (Predictions explain variance of 3.9% of variance) and mae of 1.15764. Meanwhile, dew temperature showed r2 of ~0.840953 and mae of ~ 2.9 and cloud_coverage r2 of ~0.52 and mae of ~1.28957. 
# * This approach sort of gives you more claritiy on how accurate your fills for NaNs are 
# * Filling of missing values based on seasonality and temporal promixity  
# * I did recieve a slight boost in my performance using this filled dataset as compared to mean filling, but much less of an improvement than I expected to see after this. If weather had more of an effect of target readings then this approach might have held more value.
# * This method might be a little overkill to fill missing values compared to other approaches seen, but let me know what you think!
# 
# 
