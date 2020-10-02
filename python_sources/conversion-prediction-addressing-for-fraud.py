#!/usr/bin/env python
# coding: utf-8

# # Conversion Prediction #
# 
# Goal of this analysis is to predict conversion - a user downloading an app after clicking a mobile app ad. 
# 
# Even though this challenge is titled "Fraud Detection", the data does not identify "gold standard" fraudulent clicks to support a true fraud analysis. This distinction - whether we are doing conversion prediction or fraud detection - could be of significance during feature engineering. For a conversion prediction, what intuitively makes a good predictor is app popularity (expect a long tail). For a fraud detection, that would be something indicating abnormal behaviors, such as the number of clicks within a short time period from the same type of device with the same IP address. Both types of features will be Incorporated in this analysis. Having planned that, I still believe it is desirable business acumen to attempt only one goal with one analysis. Ad channel implementation then combines findings from multiple analyses with various purposes. 
# 
# Conversion is expected to be low even if there were not any fraud. That 90% of TalkingData's daily clicks are potentially fraudulent makes the event even more rare. Potential approaches include resampling and ensemble techniques. I will first look at how rare the event is. In case of extreme unbalancy, over-sampling and under-sampling would not seem appropriate. I will build simple random forest and gradient boosting models to quickly look at feature importance. 

# ## 1. Load data ##

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import gc # Garbage Collector


# Load the provided subset of training data. Loading the entire training dataset here would lead to kernel death!

# In[2]:


df_train = pd.read_csv("../input/train_sample.csv", parse_dates=['click_time', 'attributed_time'])
print('Data frame column types:')
print(df_train.dtypes)
print("\n")
print('Glimpse:')
print(df_train.head())


# In[4]:


df_test = pd.read_csv("../input/test.csv", parse_dates=['click_time'])


# ## 2. Look at data and create features ##

# ### 2.1 Check missing ###

# In[6]:


print(f'Data has no missing value? {df_train.isnull().values.any()}')


# ### 2.2 Conversion rate ###

# How rare is the event?

# In[7]:


print('app download frequency (0 - no, 1 - yes):')
print(df_train['is_attributed'].value_counts())
print('percentage:')
print(df_train['is_attributed'].value_counts(normalize=True))


# Conversion rate is extremely low!

# ### 2.3 Build time features ###

# The overview states that the clicks are over 4 days. Extract day, hour, minute, and second and discard year/month. 

# In[8]:


def handle_time(df_name):
    df_name['click_day'] = df_name['click_time'].dt.day
    df_name['click_hour'] = df_name['click_time'].dt.hour
#     df_train['click_minute'] = df_train['click_time'].dt.minute
#     df_train['click_second'] = df_train['click_time'].dt.second
    return df_name

df_train = handle_time(df_train)
df_test = handle_time(df_test)


# ### 2.4 Split training and testing datasets ###

# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(df_train.drop(['attributed_time', 'click_time'], axis=1),                                                                 df_train['is_attributed'], test_size=0.33, random_state=0)

del df_train
gc.collect()


# ### 2.5 Conversion-oriented features ###

# My hypothesis is that, assuming no fraud, some apps on some oses/devices are more popular than others and naturally will get more downloads. Some marketing channels are more successful than others. Your IP address alone probably does not contribute much to your download decision. From the platform's and advertising company's perspective, channels may perform better if they are targeted, e.g., show an educational app when detecting an IP from an education institute. Time has been observed to have a relatively high impact on conversion. However, do platforms want to rule out the time effect when testing the effectiveness of channels? 
# 
# App/os/device/etc. popularity can be represented as "conversion rate": 
# $$\frac{\text{is_attributed count}}{\text{click count in that category (or multiple-category combination)}}$$

# In[10]:


feature_combinations = [        
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
    ['app', 'device', 'os']
]

for cols in feature_combinations:
    calc_df = X_train.groupby(cols)['is_attributed'].apply(lambda x: x.sum() / float(x.count())) 
    calc_df = calc_df.to_frame()
    calc_df.rename(columns={'is_attributed': '_'.join(cols)+'_conv_rate'}, inplace=True)

    X_train = X_train.join(calc_df, on=cols, how='left', sort=False)
    X_validation = X_validation.join(calc_df, on=cols, how='left', sort=False)
    X_validation.fillna(0, inplace=True)

    df_test = df_test.join(calc_df, on=cols, how='left', sort=False)
    df_test.fillna(0, inplace=True)

    del calc_df

gc.collect()


# My training dataset is small so I have many app-channel-etc combinations that exist in the validation and test datasets respectively that are absent from the training data. 

# [NanoMathias's kernel](https://www.kaggle.com/nanomathias/feature-engineering-importance-testing) defines a confidence level to the rate above based on the number of views. I should make such adjustment too. 

# ### 2.6 Fraud-detection-oriented features ###

# A large total number of clicks from the same IP could potentially be a good indicator for fraud. Frequent clicks within a short time sounds suspicious too but the total clicks may be lower to avoid detection. 
# For fraudsters IP address is easy to change. I assume device and OS are difficult to change. Let's count the number of clicks within the same ip-device-os-etc combinations during a time window. 

# In[11]:


# https://www.kaggle.com/nanomathias/feature-engineering-importance-testing. Simplified. Only keep some combinations. 
# Define all the groupby transformations
click_aggregations = [
    # Variance in hour, for ip-app-os
    # {'groupby': ['ip','app','os'], 'select': 'click_hour', 'agg': 'var'},
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'click_hour', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','click_day','click_hour'], 'select': 'channel', 'agg': 'count'},    
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'}    
]


# Apply all the groupby transformations
for spec in click_aggregations:

    # Name pattern of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), spec['agg'], spec['select'])

    # Info
#     print("Grouping by {}, and aggregating {} with {}".format(
#         spec['groupby'], spec['select'], spec['agg']
#     ))

    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))

    # Perform the groupby
    gp = X_train[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(columns={spec['select']: new_feature})

    # Merge back to total data frame
    if 'cumcount' == spec['agg']:
        X_train[new_feature] = gp[0].values
        X_validation[new_feature] = gp[0].values
        X_validation.fillna(0, inplace=True)
        
        df_test[new_feature] = gp[0].values
        df_test.fillna(0, inplace=True)
    else:
        X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        X_validation = X_validation.merge(gp, on=spec['groupby'], how='left')
        X_validation.fillna(0, inplace=True)
        
        df_test = df_test.merge(gp, on=spec['groupby'], how='left')
        df_test.fillna(0, inplace=True)
     # Clear memory
    del gp
    gc.collect()

gc.collect()


# ### 2.5 Categorical features ###

# In[12]:


for col in ['os', 'app', 'device', 'channel']:
    print(f'Number of unique {col} in training data: {X_train[col].nunique()}')


# In[13]:


for col in ['os', 'app', 'device', 'channel']:
    print(f'Number of unique {col} in testing data: {df_test[col].nunique()}')


# One-hot encoding would be too sparse. Let's apply the hashing trick to these high-cardinality features.

# In[14]:


from sklearn.feature_extraction import FeatureHasher 

FH = FeatureHasher(n_features=6, input_type='string') # device will have hash collision
for col in ['os', 'app', 'device', 'channel']:
    newcolnm = col+'_FH'
    newcolnm = pd.DataFrame(FH.transform(X_train[col].astype('str')).toarray()).add_prefix(col)
    X_train = X_train.join(newcolnm)
    X_validation = X_validation.join(newcolnm)
    X_validation.fillna(0, inplace=True)
    
    df_test = df_test.join(newcolnm)
    df_test.fillna(0, inplace=True)
    del newcolnm
    gc.collect()
del FH
    
gc.collect()


# ## 3. Fit Model ##

# In[15]:


# from h2o.estimators import H2ORandomForestEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[16]:


X_train.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], axis=1, inplace=True)
X_validation.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], axis=1, inplace=True)

df_test.drop(['ip', 'app', 'device', 'os', 'channel', 'click_time'], axis=1, inplace=True)


# ### 3.2 Random Forest ###

# In[17]:


# # Define model
# clf_rf = RandomForestClassifier(random_state=0)
# # Train model
# clf_rf.fit(X_train, y_train)


# ### 3.3 Adaptive Boosting ###

# In[18]:


clf_adab = AdaBoostClassifier(n_estimators=200, random_state=0)
clf_adab.fit(X_train, y_train)


# ## 4. Performance ##

# ### 4.1 Random forest ###

# In[19]:


# importances = clf_rf.feature_importances_
 
# print ("Random Forest Sorted Feature Importance:")
# sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
# print (sorted_feature_importance)
# print ('\n')
# print(f'Random Forest AUC: {roc_auc_score(y_validation, clf_rf.predict_proba(X_validation)[:, 1])}')

# del importances


# ### 4.2 AdaBoost ###

# #### 4.2.1 Feature importance ####

# In[20]:


importances = clf_adab.feature_importances_

print ("AdaBoost Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
print (sorted_feature_importance)


# #### 4.2.2 AUC ####

# In[21]:


print(f'AdaBoost AUC: {roc_auc_score(y_validation, clf_adab.predict_proba(X_validation)[:, 1])}')
del importances


# ## 5. Prediction ##

# In[22]:


del X_train, y_train, X_validation, y_validation
gc.collect()


# In[23]:


split_size = 20
test_df_list = np.array_split(df_test, split_size, axis=0)


# In[24]:


submission_df_list = []
for i, test_df_chunk in reversed(list(enumerate(test_df_list))):
    test_df_chunk['is_attributed'] = clf_adab.predict_proba(test_df_chunk.drop('click_id', axis=1))[:, 1]
    submission_df_list.append(test_df_chunk[['click_id', 'is_attributed']])
    del test_df_list[i]
    gc.collect()


# In[26]:


del df_test
gc.collect()


# In[27]:


result = pd.concat(submission_df_list)
del submission_df_list
gc.collect()

result.sort_values(by='click_id', inplace=True)
result.to_csv('adaboost_submission.csv', header=True, index=False)

