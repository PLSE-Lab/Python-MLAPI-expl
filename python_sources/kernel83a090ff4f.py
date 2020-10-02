#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp

columns_required = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
path= '~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/'
# Load subset of the training data
X_train = pd.read_csv('../input/train.csv',usecols=columns_required,parse_dates=['click_time'])

# Show the head of the table
X_train.head()
X_train.shape


# In[ ]:


# segregating the click_time column into day, hour, minute and second


# In[ ]:


X_train['day'] = X_train['click_time'].dt.day.astype('uint8')
X_train['hour'] = X_train['click_time'].dt.hour.astype('uint8')
X_train['minute'] = X_train['click_time'].dt.minute.astype('uint8')
X_train['second'] = X_train['click_time'].dt.second.astype('uint8')
X_train.head()


# In[ ]:


X_train.loc[:, 'device'] = X_train.loc[:, 'device'].astype(np.int16)
X_train.loc[:, 'os'] = X_train.loc[:, 'os'].astype(np.int16)
X_train.loc[:, 'channel'] = X_train.loc[:, 'channel'].astype(np.int16)
X_train.loc[:, 'is_attributed'] = X_train.loc[:, 'is_attributed'].astype(np.int8)
X_train.loc[:, 'ip'] = X_train.loc[:, 'ip'].astype(np.int8)
X_train.loc[:, 'app'] = X_train.loc[:, 'app'].astype(np.int8)


# In[ ]:


pp.ProfileReport(X_train)


# In[ ]:


columns_required = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

# Load subset of the training data
X_test = pd.read_csv('../input/test.csv',nrows=100000,usecols=columns_required,parse_dates=['click_time'])
X_test.fillna(X_test.mean(), inplace=True)
# Show the head of the table
X_test.head()
X_test.shape


# In[ ]:


X_test['day'] = X_test['click_time'].dt.day.astype('uint8')
X_test['hour'] = X_test['click_time'].dt.hour.astype('uint8')
X_test['minute'] = X_test['click_time'].dt.minute.astype('uint8')
X_test['second'] = X_test['click_time'].dt.second.astype('uint8')
X_test.head()


# In[ ]:


X_test.loc[:, 'device'] = X_test.loc[:, 'device'].astype(np.int16)
X_test.loc[:, 'os'] = X_test.loc[:, 'os'].astype(np.int16)
X_test.loc[:, 'channel'] = X_test.loc[:, 'channel'].astype(np.int16)
X_test.loc[:, 'ip'] = X_test.loc[:, 'ip'].astype(np.int8)
X_test.loc[:, 'app'] = X_test.loc[:, 'app'].astype(np.int8)


# In[ ]:


ATTRIBUTION_CATEGORIES = [        
    # Group-1 Features 
    ['ip'], ['app'], ['device'], ['os'], ['channel'],
    
    # Group-2 Features
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
    
    # Group-3 Features
    ['channel', 'os'],
    ['channel', 'device'],
    ['os', 'device']
]


# In[ ]:


# Find frequency of is_attributed for each unique value in column in train data
freqs = {}
for cols in ATTRIBUTION_CATEGORIES:
    
    # New feature name
    new_feature = '_'.join(cols)+'_confRate'    
    
    # Perform the groupby
    group_object = X_train.groupby(cols)
    
    # Group sizes    
    group_sizes = group_object.size()
    log_group = np.log(100000) 
    print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
        cols, new_feature, 
        group_sizes.max(), 
        np.round(group_sizes.mean(), 2),
        np.round(group_sizes.median(), 2),
        group_sizes.min()
    ))
    
    # Aggregation function
    def rate_calculation(x):
        """Calculate the attributed rate. Scale by confidence"""
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_group])
        return rate * conf
    
    # Merge operation
    X_train = X_train.merge(
        group_object['is_attributed']. \
            apply(rate_calculation). \
            reset_index(). \
            rename( 
                index=str,
                columns={'is_attributed': new_feature}
            )[cols + [new_feature]],
        on=cols, how='left'
    )
    
X_train.head()


# In[ ]:


#Test Data
freqs = {}
for cols in ATTRIBUTION_CATEGORIES:
    
    # New feature name
    new_feature = '_'.join(cols)+'_confRate'    
    
    # Perform the groupby
    group_object = X_test.groupby(cols)
    
    # Group sizes    
    group_sizes = group_object.size()
    log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
    print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
        cols, new_feature, 
        group_sizes.max(), 
        np.round(group_sizes.mean(), 2),
        np.round(group_sizes.median(), 2),
        group_sizes.min()
    ))


# In[ ]:


# Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    
    # Group-1 - GroupBy Features    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # Group-2 - GroupBy Features 
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # Group-3 - GroupBy Features     
    # Reference from https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}   
    
]


# In[ ]:


# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
     # Perform the groupby
    gp = X_train[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        X_train[new_feature] = gp[0].values
    else:
        X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()

X_train.head()


# In[ ]:


#Test Data 
#Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
     # Perform the groupby
    gp = X_test[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        X_test[new_feature] = gp[0].values
    else:
        X_test= X_test.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()

X_test.head()


# In[ ]:


# Train Data
GROUP_BY_NEXT_CLICKS = [
    
    # Group-1
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
    
    # Group-3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
]

# Calculate the time to next click for each group
for t in GROUP_BY_NEXT_CLICKS:
    
    # Name of new feature
    new_feature = '{}_nextClick'.format('_'.join(t['groupby']))    
    
    # Unique list of features to select
    all_features = t['groupby'] + ['click_time']
    
    # Run calculation
    print(f">> Grouping by {t['groupby']}, and saving time to next click in: {new_feature}")
    X_train[new_feature] = X_train[all_features].groupby(t['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    
X_train.head()


# In[ ]:


#Test Data
GROUP_BY_NEXT_CLICKS = [
    
    # V1
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
]

# Calculate the time to next click for each group
for t in GROUP_BY_NEXT_CLICKS:
    
    # Name of new feature
    new_feature = '{}_nextClick'.format('_'.join(t['groupby']))    
    
    # Unique list of features to select
    all_features = t['groupby'] + ['click_time']
    
    # Run calculation
    print(f">> Grouping by {t['groupby']}, and saving time to next click in: {new_feature}")
    X_test[new_feature] = X_test[all_features].groupby(t['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    
X_test.head()


# In[ ]:


HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    
    # Clicks in the past
    X_train['prev_'+fname] = X_train.         groupby(fset).         cumcount().         rename('prev_'+fname)
        
    # Clicks in the future
    X_train['future_'+fname] = X_train.iloc[::-1].         groupby(fset).         cumcount().         rename('future_'+fname).iloc[::-1]

# Count cumulative subsequent clicks
X_train.head()


# In[ ]:


# Test Data
HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    
    # Clicks in the past
    X_test['prev_'+fname] = X_test.         groupby(fset).         cumcount().         rename('prev_'+fname)
        
    # Clicks in the future
    X_test['future_'+fname] = X_test.iloc[::-1].         groupby(fset).         cumcount().         rename('future_'+fname).iloc[::-1]

# Count cumulative subsequent clicks
X_test.head()


# In[ ]:


# Split into X and y
X_train.fillna(X_train.mean(), inplace=True)
y_train = X_train['is_attributed']
X_train= X_train.drop('is_attributed', axis=1).select_dtypes(include=[np.number])
# Oversampling to decrease imbalance in labels
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train, y_train = sm.fit_sample(X_train, y_train)
#Shuffle the data to train well
from sklearn.utils import shuffle
shuffle(X_train)


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_train = SelectKBest(chi2, k=42).fit_transform(abs(X_train), y_train)
X_train.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid= train_test_split(X_train,y_train,test_size=0.3, random_state=0)


# In[ ]:


# Create a model
# Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
import xgboost as xgb
clf_xgBoost = xgb.XGBClassifier(max_depth = 4,subsample = 0.8,colsample_bytree = 0.7,colsample_bylevel = 0.7,scale_pos_weight = 9,
    min_child_weight = 0,reg_alpha = 0.01,n_jobs = -1, objective = 'binary:logistic')
# Fit the models
clf_xgBoost.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import roc_auc_score
y_pred=clf_xgBoost.predict(X_valid)


# In[ ]:


roc_auc_score(y_valid, y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf1=RandomForestClassifier(n_jobs=-1,criterion="entropy",min_samples_leaf=1,min_samples_split=8,                                                 n_estimators=15,max_features=None,random_state=100)
clf1.fit(X_train,y_train)
                        
    
    
y_pred_rf=clf1.predict(X_valid)
print(y_pred_rf)
    



# In[ ]:


roc_auc_score(y_valid, y_pred_rf)


# In[ ]:


X_test.drop('click_time',axis=1,inplace=True)


# In[ ]:


X_test.fillna(X_test.mean(), inplace=True)


# In[ ]:


y_pred_t=clf1.predict(X_test)


# In[ ]:


# Create submission file
submission = pd.DataFrame({'click_id':[i for i in range(len(y_pred_t))],'is_attributed':y_pred_t})
submission.to_csv('submission.csv', index=False)

