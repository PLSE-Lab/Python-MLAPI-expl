#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import sys
import gc
import random
pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
pd.options.display.float_format

from sklearn.model_selection import train_test_split

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# ****Data loading and Preprocessing****

# In[ ]:


def load_properties_data(file_name):

    # Helper function for parsing the flag attributes
    def convert_true_to_float(df, col):
        df.loc[df[col] == 'true', col] = '1'
        df.loc[df[col] == 'Y', col] = '1'
        df[col] = df[col].astype(float)

    prop = pd.read_csv(file_name, dtype={
        'propertycountylandusecode': str,
        'hashottuborspa': str,
        'propertyzoningdesc': str,
        'fireplaceflag': str,
        'taxdelinquencyflag': str
    })

    for col in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']:
        convert_true_to_float(prop, col)

    return prop

train_2016 = pd.read_csv('/kaggle/input/zillow-prize-1/train_2016_v2.csv' , parse_dates=["transactiondate"])
train_2017 = pd.read_csv('/kaggle/input/zillow-prize-1/train_2017.csv' , parse_dates=["transactiondate"])
prop_2016 = load_properties_data('/kaggle/input/zillow-prize-1/properties_2016.csv')
prop_2017 = load_properties_data('/kaggle/input/zillow-prize-1/properties_2017.csv')
test = pd.read_csv('/kaggle/input/zillow-prize-1/sample_submission.csv')
print("Training 2016 transaction: " + str(train_2016.shape))
print("Training 2017 transaction: " + str(train_2017.shape))
print("Number of Property 2016: " + str(prop_2016.shape))
print("Number of Property 2017: " + str(prop_2017.shape))
print("Sample Size: " + str(test.shape))


# **Feature Engineering**

# In[ ]:


# Basic feature engineering + Drop duplicate columns
for prop in [prop_2016, prop_2017]:
    prop['avg_garage_size'] = prop['garagetotalsqft'] / prop['garagecarcnt']
    
    prop['property_tax_per_sqft'] = prop['taxamount'] / prop['calculatedfinishedsquarefeet']
    
    # Rotated Coordinates
    prop['location_1'] = prop['latitude'] + prop['longitude']
    prop['location_2'] = prop['latitude'] - prop['longitude']
    prop['location_3'] = prop['latitude'] + 0.5 * prop['longitude']
    prop['location_4'] = prop['latitude'] - 0.5 * prop['longitude']
    
    # finished_area_sqft and 'total_area' cover only a subset of 'calculatedfinishedsquarefeet', when both fields are not null, the values are always the same 
    # So we can probably drop 'finished_area_sqft' and 'total_area' since they are redundant
    # If there're some patterns in when the values are missing, we can add two isMissing binary features
    
    prop['missing_finished_area'] = prop['finishedsquarefeet12'].isnull().astype(np.float32)
    prop['missing_total_area'] = prop['finishedsquarefeet15'].isnull().astype(np.float32)
    prop.drop(['finishedsquarefeet12', 'finishedsquarefeet15'], axis=1, inplace=True)
    
    # Same as above, 'bathroomcnt' covers everything that 'bathroom_cnt_calc' has
    # So we can safely drop 'bathroom_cnt_calc' and optionally add an isMissing feature
    prop['missing_bathroom_cnt_calc'] = prop['calculatedbathnbr'].isnull().astype(np.float32)
    prop.drop(['calculatedbathnbr'], axis=1, inplace=True)
    
    # 'room_cnt' has many zero or missing values
    # On the other hand, 'bathroom_cnt' and 'bedroom_cnt' have few zero or missing values
    # Add an derived room_cnt feature by adding bathroom_cnt and bedroom_cnt
    prop['derived_room_cnt'] = prop['bedroomcnt'] + prop['bathroomcnt']
    
    # Average area in sqft per room
    mask = (prop.roomcnt >= 1)  # avoid dividing by zero
    prop.loc[mask, 'avg_area_per_room'] = prop.loc[mask, 'calculatedfinishedsquarefeet'] / prop.loc[mask, 'roomcnt']
    
    # Use the derived room_cnt to calculate the avg area again
    mask = (prop.derived_room_cnt >= 1)
    prop.loc[mask,'derived_avg_area_per_room'] = prop.loc[mask,'calculatedfinishedsquarefeet'] / prop.loc[mask,'derived_room_cnt']
    

prop_2017.head()


# In[ ]:


train_2016 = train_2016.merge(prop_2016, how='left', on='parcelid')
train_2017 = train_2017.merge(prop_2017, how='left', on='parcelid')
train = pd.concat([train_2016, train_2017], axis=0, ignore_index=True)


# In[ ]:


print("\nCombined training set size: {}".format(len(train)))
train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()

for c in train.columns:
    train[c]= train[c].fillna(0)
    if train[c].dtype == 'object':
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))


# In[ ]:


def add_simple_datetime_features(df):
    dt = pd.to_datetime(df.transactiondate).dt
    df['year'] = (dt.year - 2016).astype(int)
    df['month'] = (dt.month).astype(int)
    df['quarter'] = (dt.quarter).astype(int)
    df.drop(['transactiondate'], axis=1, inplace=True)
add_simple_datetime_features(train)
train.head()


# In[ ]:


"""
    Drop id and label columns + Feature selection for Cast Boot
"""        
def drop_features(features):
    # id and label (not features)
    unused_feature_list = ['parcelid', 'logerror']

    # too many missing (LightGBM is robust against bad/unrelated features, so this step might not be needed)
    missing_list = ['buildingclasstypeid', 'architecturalstyletypeid', 'storytypeid', 'finishedsquarefeet13', 'basementsqft', 'yardbuildingsqft26']
    unused_feature_list += missing_list

    # not useful
    bad_feature_list = ['fireplaceflag', 'decktypeid', 'pooltypeid10', 'typeconstructiontypeid', 'regionidcounty', 'fips']
    unused_feature_list += bad_feature_list

    # really hurts performance
    unused_feature_list += ['propertycountylandusecode','propertyzoningdesc', 'taxdelinquencyflag']

    return features.drop(unused_feature_list, axis=1, errors='ignore')


# In[ ]:


castboot_features = drop_features(train)
print("Number of features for CastBoot: {}".format(len(castboot_features.columns)))
castboot_features.head(5)


# In[ ]:


# Prepare training and cross-validation data
castboot_label = train.logerror.astype(np.float32)
print(castboot_label.head())

# Transform to Numpy matrices
lgb_X = castboot_features.values
lgb_y = castboot_label.values

# Perform shuffled train/test split
np.random.seed(42)
random.seed(10)
X_train, X_val, y_train, y_val = train_test_split(lgb_X, lgb_y, test_size=0.2)

# Remove outlier examples from X_train and y_train; Keep them in X_val and y_val for proper cross-validation
outlier_threshold = 0.4
mask = (abs(y_train) <= outlier_threshold)
X_train = X_train[mask, :]
y_train = y_train[mask]

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))


# In[ ]:


feature_names = [s for s in castboot_features.columns]
categorical_features = ['airconditioningtypeid', 'heatingorsystemtypeid', 'propertylandusetypeid', 'year', 'month', 'quarter']

categorical_indices = []
for i, n in enumerate(castboot_features.columns):
    if n in categorical_features:
        categorical_indices.append(i)
print(categorical_indices)


# In[ ]:


# CatBoost parameters
params = {}
params['loss_function'] = 'MAE'
params['eval_metric'] = 'MAE'
params['nan_mode'] = 'Min'  # Method to handle NaN (set NaN to either Min or Max)
params['random_seed'] = 0

params['iterations'] = 1000  # default 1000, use early stopping during training
params['learning_rate'] = 0.015  # default 0.03

params['border_count'] = 254  # default 254 (alias max_bin, suggested to keep at default for best quality)

params['max_depth'] = 6  # default 6 (must be <= 16, 6 to 10 is recommended)
params['random_strength'] = 1  # default 1 (used during splitting to deal with overfitting, try different values)
params['l2_leaf_reg'] = 5  # default 3 (used for leaf value calculation, try different values)
params['bagging_temperature'] = 1  # default 1 (higher value -> more aggressive bagging, try different values)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from catboost import CatBoostRegressor, Pool\n# Train CatBoost Regressor with cross-validated early-stopping\nval_pool = Pool(X_val, y_val, cat_features=categorical_indices)\n\nnp.random.seed(42)\nrandom.seed(36)\nmodel = CatBoostRegressor(**params)\nmodel.fit(X_train, y_train,\n          cat_features=categorical_indices,\n          use_best_model=True, eval_set=val_pool, early_stopping_rounds=50, verbose=False)\n\n# Evaluate model performance\nprint("Train score: {}".format(abs(model.predict(X_train) - y_train).mean() * 100))\nprint("Val score: {}".format(abs(model.predict(X_val) - y_val).mean() * 100))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\nparams = {\'depth\': [4, 7, 10],\n          \'learning_rate\' : [0.03, 0.1, 0.15],\n         \'l2_leaf_reg\': [1,4,9],\n         \'iterations\': [300, 1000, 1500],\n         \'eval_metric\' : [\'MAE\']}\nmodel = CatBoostRegressor()\ngrid = GridSearchCV(estimator= model, param_grid= params, cv= 3, n_jobs=-1)\ngrid.fit(X_train, y_train)\nprint("\\n========================================================")\nprint(" Results from Grid Search " )\nprint("========================================================")    \nprint("\\n The best estimator across ALL searched params:\\n", grid.best_estimator_)\nprint("\\n The best score across ALL searched params:\\n", grid.best_score_)\nprint("\\n The best parameters across ALL searched params:\\n", grid.best_params_)')


# In[ ]:


# CatBoost feature importance
feature_importance = [(feature_names[i], value) for i, value in enumerate(model.get_feature_importance())]
feature_importance.sort(key=lambda x: x[1], reverse=True)
for k, v in feature_importance[:10]:
    print("{}: {}".format(k, v))


# In[ ]:


def transform_test_features(features_2016, features_2017):
    test_features_2016 = drop_features(features_2016)
    test_features_2017 = drop_features(features_2017)
    
    test_features_2016['year'] = 0
    test_features_2017['year'] = 1
    
    # 11 & 12 lead to unstable results, probably due to the fact that there are few training examples for them
    test_features_2016['month'] = 10
    test_features_2017['month'] = 10
    
    test_features_2016['quarter'] = 4
    test_features_2017['quarter'] = 4
    
    return test_features_2016, test_features_2017

"""
    Helper method that makes predictions on the test set and exports results to csv file
    'models' is a list of models for ensemble prediction (len=1 means using just a single model)
"""
def predict_and_export(models, features_2016, features_2017, file_name):
    # Construct DataFrame for prediction results
    submission_2016 = pd.DataFrame()
    submission_2017 = pd.DataFrame()
    submission_2016['ParcelId'] = features_2016.parcelid
    submission_2017['ParcelId'] = features_2017.parcelid
    
    test_features_2016, test_features_2017 = transform_test_features(features_2016, features_2017)
    
    pred_2016, pred_2017 = [], []
    for i, model in enumerate(models):
        print("Start model {} (2016)".format(i))
        pred_2016.append(model.predict(test_features_2016))
        print("Start model {} (2017)".format(i))
        pred_2017.append(model.predict(test_features_2017))
    
    # Take average across all models
    mean_pred_2016 = np.mean(pred_2016, axis=0)
    mean_pred_2017 = np.mean(pred_2017, axis=0)
    
    submission_2016['201610'] = [float(format(x, '.4f')) for x in mean_pred_2016]
    submission_2016['201611'] = submission_2016['201610']
    submission_2016['201612'] = submission_2016['201610']

    submission_2017['201710'] = [float(format(x, '.4f')) for x in mean_pred_2017]
    submission_2017['201711'] = submission_2017['201710']
    submission_2017['201712'] = submission_2017['201710']
    
    submission = submission_2016.merge(how='inner', right=submission_2017, on='ParcelId')
    
    print("Length of submission DataFrame: {}".format(len(submission)))
    print("Submission header:")
    print(submission.head())
    submission.to_csv(file_name, index=False)
    return submission, pred_2016, pred_2017 


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
def fillnan(train):    
    for c in train.columns:
        train[c]= train[c].fillna(0)
        if train[c].dtype == 'object':
            lbl.fit(list(train[c].values))
            train[c] = lbl.transform(list(train[c].values))
    return train
prop2016 = fillnan(prop_2016)
prop2017 = fillnan(prop_2017)
        
file_name = 'final_castboot_single_21092019.csv'
submission, pred_2016, pred_2017 = predict_and_export([model], prop_2016, prop_2017, file_name)


# **Ensemble Training & Prediction**

# In[ ]:


# Remove outliers (if any) from training data
outlier_threshold = 0.4
mask = (abs(lgb_y) <= outlier_threshold)
catboost_X = lgb_X[mask, :]
catboost_y = lgb_y[mask]
print("catboost_X: {}".format(catboost_X.shape))
print("catboost_y: {}".format(catboost_y.shape))

####################3
bags = 8
models = []
params['iterations'] = 1000
for i in range(bags):
    print("Start training model {}".format(i))
    params['random_seed'] = i
    np.random.seed(42)
    random.seed(36)
    model = CatBoostRegressor(**params)
    model.fit(catboost_X, catboost_y, cat_features=categorical_indices, verbose=False)
    models.append(model)
    
# Sanity check (make sure scores on a small portion of the dataset are reasonable)
for i, model in enumerate(models):
    print("model {}: {}".format(i, abs(model.predict(X_val) - y_val).mean() * 100))


# In[ ]:


# Make predictions and export results
lbl = LabelEncoder()
def fillnan(train):    
    for c in train.columns:
        train[c]= train[c].fillna(0)
        if train[c].dtype == 'object':
            lbl.fit(list(train[c].values))
            train[c] = lbl.transform(list(train[c].values))
    return train
prop2016 = fillnan(prop_2016)
prop2017 = fillnan(prop_2017)

file_name = 'final_catboost_ensemble_x8_20192209.csv'
submission, pred_2016, pred_2017 = predict_and_export(models, prop_2016, prop_2017, file_name)

