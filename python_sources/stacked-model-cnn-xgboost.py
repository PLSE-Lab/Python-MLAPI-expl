#!/usr/bin/env python
# coding: utf-8

# # Meta Text Features. 

# ## CNN on title + description -> 't' Score + 'd' Score or 'text' Score

# The model I used for text classification problem is CNN. 
# 
# You can check this paper for a detailed explaination.
# https://arxiv.org/abs/1408.5882
# 
# Also check this post for an idea of implementation.
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

# I split the train dataset to 3 folds evenly, as the size of test dataset is 1/3 of the size of train dataset. For each fold, it's score is predicted by applying CNN on the other two folds. This is to pretend overfitting. 
# 
# Check these for more information. 
# 
# https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
# 
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

# However, I don't recommended this implementation for real problem because it is time killer (Approx. 20 hours for CV) and have to run over all train data again while new data comes in. 
# 
# If any of you know any idea on how to apply this model without training all  over again when new data comes in, PLZ COMMENT BELOW, I'd love to know it.
# 
# Below are some details on time and rmse. 

# # Result
# Fold 1 Start!
# 
# Epoch 1/3
#  - 4620s - loss: 0.0581 - mean_squared_error: 0.0581
# 
# Epoch 2/3
#  - 3155s - loss: 0.0547 - mean_squared_error: 0.0547
# 
# Epoch 3/3
#  - 3323s - loss: 0.0534 - mean_squared_error: 0.0534
# 
# Title RMSE: 0.234442
# 
# Epoch 1/3
#  - 6731s - loss: 0.0580 - mean_squared_error: 0.0580
# 
# Epoch 2/3
#  - 4412s - loss: 0.0546 - mean_squared_error: 0.0546
# 
# Epoch 3/3
#  - 4420s - loss: 0.0524 - mean_squared_error: 0.0524
# 
# Description RMSE: 0.236288
# 
# Fold 2 Start!
# 
# Epoch 1/3
#  - 2436s - loss: 0.0578 - mean_squared_error: 0.0578
# 
# Epoch 2/3
#  - 2414s - loss: 0.0546 - mean_squared_error: 0.0546
# 
# Epoch 3/3
#  - 2281s - loss: 0.0534 - mean_squared_error: 0.0534
# 
# Title RMSE: 0.235876
# 
# Epoch 1/3
#  - 4878s - loss: 0.0600 - mean_squared_error: 0.0600
# 
# Epoch 2/3
#  - 4345s - loss: 0.0547 - mean_squared_error: 0.0547
# 
# Epoch 3/3
#  - 9811s - loss: 0.0526 - mean_squared_error: 0.0526
# 
# Description RMSE: 0.235445
# 
# Fold 3 Start!
# 
# Epoch 1/3
#  - 3731s - loss: 0.0575 - mean_squared_error: 0.0575
# 
# Epoch 2/3
#  - 3685s - loss: 0.0545 - mean_squared_error: 0.0545
# 
# Epoch 3/3
#  - 3645s - loss: 0.0533 - mean_squared_error: 0.0533
# 
# Title RMSE: 0.235326
# 
# Epoch 1/3
#  - 7152s - loss: 0.0579 - mean_squared_error: 0.0579
# 
# Epoch 2/3
#  - 6924s - loss: 0.0544 - mean_squared_error: 0.0544
# 
# Epoch 3/3
#  - 6815s - loss: 0.0523 - mean_squared_error: 0.0523
# 
# Description RMSE: 0.236635
# 
# Finish!

# I will skip the steps on how to generate meta text features. The code for it can be found here. 
# 
# https://github.com/JingqiL/Stacked-models-CNN-XGBoost-/blob/master/Keras%20CNN(title%2Bdescription).ipynb
# 
# You can also first merge two scores of title and description to get an overall score ('text' score) and then feed this score and other features to your ML algorithms. The code of how to merge the score of title and description is in this kernel. (This is a tensorflow version, you can easily transfer it to a Keras version.)
# 
# https://www.kaggle.com/jingqliu/fasttext-conv2d-with-tf-on-title-and-description/notebook

# #### This version is the result for 'text' score. I have updated the script in github link for both 't'+'d' and 'text'. If you want check the result of 't' + 'd', please go version 6. The rmse is 0.2266.

# # XGBoost features on meta text + grouped features.

# In[ ]:


import pandas as pd
import numpy as np
import re
import gc
from keras.preprocessing import text, sequence
import math
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/readyforuse/trainforuse_mix.csv')
test = pd.read_csv('../input/readyforuse/testforuse_mix.csv')
submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')

train = train.drop(columns = ['item_id'])
train['date'] = pd.to_datetime(train['activation_date']).dt.weekday.astype('int')
train = train.drop(columns = ['activation_date','image'])

test = test.drop(columns = ['item_id'])
test['date'] = pd.to_datetime(test['activation_date']).dt.weekday.astype('int')
test = test.drop(columns = ['activation_date','image'])


# In[ ]:


encoding_predictors = ['user_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'item_seq_number', 'user_type', 'date']


# In[ ]:


for i in encoding_predictors:
    print(str(i) +': ' + str(len(set(train[i]))))
    print('Na contains? ' + str(train[i].isnull().values.any()))


# In[ ]:


for i in encoding_predictors:
    print(str(i) +': ' + str(len(set(test[i]))))
    print('Na contains? ' + str(test[i].isnull().values.any()))


# In[ ]:


l_region = dict(zip(list(set(train['region'])),range(1,29)))
l_parent_category_name = dict(zip(list(set(train['parent_category_name'])),range(1,10)))
l_category_name = dict(zip(list(set(train['category_name'])),range(1,48)))
l_user_type = dict(zip(list(set(train['user_type'])),range(1,4)))
l_date = dict(zip(list(set(train['date'])),range(1,8)))
l_param1 = dict(zip(list(set(train['param_1'])),range(1,373)))
l_param2 = dict(zip(list(set(train['param_2'])),range(1,273)))
l_city = dict(zip(list(set(train['city'])),range(1,1734)))
l_param3 = dict(zip(list(set(train['param_3'])),range(1,1221)))


# In[ ]:


train['region'] = train['region'].replace(l_region)
train['parent_category_name'] = train['parent_category_name'].replace(l_parent_category_name)
train['category_name'] = train['category_name'].replace(l_category_name)
train['user_type'] = train['user_type'].replace(l_user_type)
train['date'] = train['date'].replace(l_date)
train['param_1'] = train['param_1'].replace(l_param1)
train['param_2'] = train['param_2'].replace(l_param2)
train['city'] = train['city'].replace(l_city)
train['param_3'] = train['param_3'].replace(l_param3)


# In[ ]:


test['region'] = test['region'].map(l_region)
test['parent_category_name'] = test['parent_category_name'].map(l_parent_category_name)
test['category_name'] = test['category_name'].map(l_category_name)
test['user_type'] = test['user_type'].map(l_user_type)
test['date'] = test['date'].map(l_date)
test['param_1'] = test['param_1'].map(l_param1)
test['param_2'] = test['param_2'].map(l_param2)
test['city'] = test['city'].map(l_city)
test['param_3'] = test['param_3'].map(l_param3)


# In[ ]:


def modelfit(alg,dtrain,predictors,useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain['deal_probability'].values, feature_names = predictors)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds, metrics = 'rmse', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators = cvresult.shape[0])
        print('Best n_estimator = ' + str(cvresult.shape[0]))
    alg.fit(dtrain[predictors], dtrain['deal_probability'], eval_metric = 'rmse')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    print('\nModel Report:')
    print('RMSE: %f' % math.sqrt(metrics.mean_squared_error(dtrain['deal_probability'].values, dtrain_predictions)))


# In[ ]:


train.loc[:,'image_item_n'] = train.groupby(['image_top_1','item_seq_number']).user_id.transform('nunique')
train.loc[:,'image_city_n'] = train.groupby(['city','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_region_n'] = train.groupby(['region','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_categoryname_n'] = train.groupby(['category_name','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_param2_n'] = train.groupby(['image_top_1','param_2']).user_id.transform('nunique')
train.loc[:,'image_parentcategoryname_n'] = train.groupby(['parent_category_name','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_date_n'] = train.groupby(['image_top_1','date']).user_id.transform('nunique')
train.loc[:,'image_usertype_n'] = train.groupby(['image_top_1','user_type']).user_id.transform('nunique')
train.loc[:,'image_param1_n'] = train.groupby(['image_top_1','param_1']).user_id.transform('nunique')
train.loc[:,'image_param3_n'] = train.groupby(['image_top_1','param_3']).user_id.transform('nunique')


# In[ ]:


test.loc[:,'image_item_n'] = test.groupby(['image_top_1','item_seq_number']).user_id.transform('nunique')
test.loc[:,'image_city_n'] = test.groupby(['city','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_region_n'] = test.groupby(['region','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_categoryname_n'] = test.groupby(['category_name','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_param2_n'] = test.groupby(['image_top_1','param_2']).user_id.transform('nunique')
test.loc[:,'image_parentcategoryname_n'] = test.groupby(['parent_category_name','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_date_n'] = test.groupby(['image_top_1','date']).user_id.transform('nunique')
test.loc[:,'image_usertype_n'] = test.groupby(['image_top_1','user_type']).user_id.transform('nunique')
test.loc[:,'image_param1_n'] = test.groupby(['image_top_1','param_1']).user_id.transform('nunique')
test.loc[:,'image_param3_n'] = test.groupby(['image_top_1','param_3']).user_id.transform('nunique')


# In[ ]:


def clean(string):
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub('[\W]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string.lower())
    return string

def find_punc(string):
    string = re.sub(r'\s','',string)
    string = re.findall('[\W]',string)
    l = len(string)
    return l


# In[ ]:


train_t = train['title'].apply(clean)
test_t = test['title'].apply(clean)
train_d = train['description'].astype(str).apply(clean)
test_d = test['description'].astype(str).apply(clean)


# In[ ]:


train_t_len = []
for line in train_t:
    train_t_len.append(len(line.split()))
    
train_d_len = []
for line in train_d:
    train_d_len.append(len(line.split()))
    
test_t_len = []
for line in test_t:
    test_t_len.append(len(line.split()))
    
test_d_len = []
for line in test_d:
    test_d_len.append(len(line.split()))


# In[ ]:


#train['t_n'] = train_t_len
train['t_per'] = np.array(train['title'].apply(find_punc))/np.array(train_t_len)
#test['t_n'] = test_t_len
test['t_per'] = np.array(test['title'].apply(find_punc))/np.array(test_t_len)
train['d_n'] = train_d_len
#train['d_per'] = np.array(train['description'].astype(str).apply(find_punc))/np.array(train_d_len)
test['d_n'] = test_d_len
#test['d_per'] = np.array(test['description'].astype(str).apply(find_punc))/np.array(test_d_len)


# In[ ]:


train = train.drop(columns = ['title','description'])
test = test.drop(columns = ['title','description'])


# In[ ]:


ready_train = train[['price','image_top_1','param_1','item_seq_number','city','region','parent_category_name','category_name','user_type','date','param_2','deal_probability','param_3','mix'] + list(train.columns[15:])]
ready_test = test[['price','image_top_1','param_1','item_seq_number','city','region','parent_category_name','category_name','user_type','date','param_2','param_3','mix'] + list(test.columns[14:])]


# In[ ]:


predictors = ready_train.columns[ready_train.columns != 'deal_probability']


# In[ ]:


len(predictors)


# In[ ]:


xgb1 = XGBRegressor(objective = 'reg:logistic', learning_rate = 0.1, n_estimators = 1000, max_depth = 5, min_child_weight = 6, gamma = 0, subsample = 0.9, colsample_bytree = 0.7, reg_alpha = 1.4, seed = 2018)


# In[ ]:


modelfit(xgb1, ready_train, predictors, useTrainCV = False)


# In[ ]:


xgb.plot_importance(xgb1)
plt.show()


# In[ ]:


pred = xgb1.predict(ready_test[predictors])


# In[ ]:


submission['deal_probability'] = pred
submission.to_csv('submission.csv',index=False)


# Feature engineering to be continued...... (These features are just for test in order to get a slightly better result than using the original predictors, don't take it seriously.) @.@

# 
