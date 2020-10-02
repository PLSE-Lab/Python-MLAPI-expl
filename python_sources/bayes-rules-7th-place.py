#!/usr/bin/env python
# coding: utf-8

# # Energy consumption
# 
# Congrats to everyone who participated. This is my way of takling this competition. It was quite easy to reach the 613K mark on LB, as pure parameter optimization with no feature engineering, except OHE, was enough to get there.
# TL;DR 
# - Impute nans(-2)
# - Create feature interactions to boost expected gain with xgbfir
# - Add features which indicate the frequency of each value of each column
# - Bayesian optimization
# - validate on kfold

# Some of the things tried which failed:
# - PCA transformation as features
# - Overfit a DNN on a projection onto 10 PCA components
# - Remove outliers with z-score
# - Isolation Forest to create outlier score as a feature
# - Remove columns which comprise mostly of one value
# - Remove 'Id', which is almost the most important feature, lol

# ### Data preprocessing

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


data = pd.read_csv('/kaggle/input/infopulsehackathon/train.csv')
test = pd.read_csv('/kaggle/input/infopulsehackathon/test.csv')


# In[ ]:


ohe = OneHotEncoder(sparse=False)
ohe.fit(data.select_dtypes(object))
data = pd.concat((data,pd.DataFrame(ohe.transform(data.select_dtypes(object)))),axis=1)
test = pd.concat((test,pd.DataFrame(ohe.transform(test.select_dtypes(object)))),axis=1)

data.drop(data.select_dtypes(object).columns,axis=1,inplace=True)
test.drop(test.select_dtypes(object).columns,axis=1,inplace=True)


# In[ ]:


target = data['Energy_consumption']
data = data.drop(['Energy_consumption'],axis=1)


# In[ ]:


cols = data.columns


# Impute -2 values with median, as the column they are in are mostly categorical and imputing with mean would not suffice.

# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=-2,strategy="median")
data_imp = imp.fit_transform(data)
test_imp = imp.fit_transform(test)


# Use xgbfir to get the features which interract the most to get most Expected gain. Just have to cycle through all the possible operators

# In[ ]:


def trip_featurize(df,feat1,feat2, feat3):
    df[f'{feat1}-{feat2}-{feat3}'] = df[feat1] - df[feat2] - df[feat3]
    df[f'{feat1}-{feat2}+{feat3}'] = df[feat1] - df[feat2] + df[feat3]
    df[f'{feat1}-{feat2}*{feat3}'] = df[feat1] - df[feat2] * df[feat3]
    df[f'{feat1}-{feat2}/{feat3}'] = df[feat1] - df[feat2] / df[feat3]
    
    df[f'{feat1}+{feat2}-{feat3}'] = df[feat1] + df[feat2] - df[feat3]
    df[f'{feat1}+{feat2}+{feat3}'] = df[feat1] + df[feat2] + df[feat3]
    df[f'{feat1}+{feat2}*{feat3}'] = df[feat1] + df[feat2] * df[feat3]
    df[f'{feat1}+{feat2}/{feat3}'] = df[feat1] + df[feat2] / df[feat3]
    
    df[f'{feat1}*{feat2}-{feat3}'] = df[feat1] * df[feat2] - df[feat3]
    df[f'{feat1}*{feat2}+{feat3}'] = df[feat1] * df[feat2] + df[feat3]
    df[f'{feat1}*{feat2}*{feat3}'] = df[feat1] * df[feat2] * df[feat3]
    df[f'{feat1}*{feat2}/{feat3}'] = df[feat1] * df[feat2] / df[feat3]
    
    df[f'{feat1}/{feat2}-{feat3}'] = df[feat1] / df[feat2] - df[feat3]
    df[f'{feat1}/{feat2}+{feat3}'] = df[feat1] / df[feat2] + df[feat3]
    df[f'{feat1}/{feat2}*{feat3}'] = df[feat1] / df[feat2] * df[feat3]
    df[f'{feat1}/{feat2}/{feat3}'] = df[feat1] / df[feat2] / df[feat3]
def double_featurize(df,feat1,feat2):
    df[f'{feat1}-{feat2}'] = df[feat1] - df[feat2]
    df[f'{feat1}+{feat2}'] = df[feat1] + df[feat2]
    df[f'{feat1}*{feat2}'] = df[feat1] * df[feat2]
    df[f'{feat1}/{feat2}'] = df[feat1] / df[feat2]


# In[ ]:


double_featurize(data,'feature_5','feature_122')
double_featurize(test,'feature_5','feature_122')

double_featurize(data,'feature_122','feature_33')
double_featurize(test,'feature_122','feature_33')

double_featurize(data,'feature_129','feature_5')
double_featurize(test,'feature_129','feature_5')

double_featurize(data,'feature_122','feature_248')
double_featurize(test,'feature_122','feature_248')

trip_featurize(data,'feature_5','feature_122', 'feature_33')
trip_featurize(test,'feature_5','feature_122', 'feature_33')

trip_featurize(data,'feature_122','feature_264', 'feature_33')
trip_featurize(test,'feature_122','feature_264', 'feature_33')

trip_featurize(data,'feature_122','feature_248', 'feature_5')
trip_featurize(test,'feature_122','feature_248', 'feature_5')

trip_featurize(data,'feature_229','feature_250', 'feature_5')
trip_featurize(test,'feature_229','feature_250', 'feature_5')


# In[ ]:


data_val = data.values
test_val = test.values


# In[ ]:


counts = []
for column in cols:
    counts.append(data[column].value_counts())


# In[ ]:


bins = np.zeros(data.shape)
for i in range(data.shape[0]):
    for j in range(len(cols)):
        bins[i][j] = counts[j][data_val[i][j]]


# In[ ]:


test_bins = np.zeros(test.shape)
for i in range(test.shape[0]):
    for j in range(len(cols)):
        try:
            test_bins[i][j] = counts[j][test_val[i][j]]
        except:
            test_bins[i][j] = 1


# In[ ]:


data_val = np.concatenate((data_imp,bins),axis=1)
test = np.concatenate((test_imp,test_bins),axis=1)


# ### Model and cross-validation

# In[ ]:


from sklearn.model_selection import KFold
import lightgbm as lgb


# In[ ]:


folds = KFold(shuffle=True,random_state=40,n_splits=5)


# These parameters are a result of Bayesian Optimization.

# In[ ]:


params = {
 'min_data_in_leaf': 47,
 'num_leaves': 61,
 'lr': 0.06172399472541107,
 'min_child_weight': 0.0070492703809497,
 'colsample_bytree': 0.06169,
 'bagging_fraction': 0.3809,
 'min_child_samples': 35,
 'subsample': 0.6077180918189186,
 'max_depth': 4,
 'objective': 'regression',
 'seed': 1337,
 'feature_fraction_seed': 1337,
 'bagging_seed': 1337,
 'drop_seed': 1337,
 'data_random_seed': 1337,
 'boosting_type': 'gbdt',
 'verbose': 1,
 'boost_from_average': True,
 'metric': 'mse',
 'cat_l2': 24.38,
 'cat_smooth': 18.49,
 'feature_fraction': 0.1045,
 'lambda_l1': 2.306,
 'lambda_l2':18.02,
 'max_cat_threshold':50,
 'min_gain_to_split':  0.2585,
 'min_sum_hessian_in_leaf':0.001923,         
}


# In[ ]:


target = target.values


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


pred = np.zeros((1,test.shape[0]))
scores = 0
features = range(data_val.shape[1])
for fold, (trn_idx, val_idx) in enumerate(folds.split(data_val)):
    x_trn, y_trn = data_val[trn_idx], target[trn_idx]
    x_val, y_val = data_val[val_idx], target[val_idx]
    
    train = lgb.Dataset(x_trn,label=y_trn)
    valid = lgb.Dataset(x_val,label=y_val)
    
    model = lgb.train(params,train,num_boost_round=10000,early_stopping_rounds=100,valid_sets=(train,valid),verbose_eval=False)
    
    valid_prediction = model.predict(x_val,num_iteration=model.best_iteration)
    test_prediction = model.predict(test,num_iteration=model.best_iteration)
    
    score = mean_squared_error(y_val, valid_prediction)
    scores += score
    pred += test_prediction
    
    print('Validation fold {} : '.format(fold),score)
    
print("SCORE",scores/5)


# ### Submission

# In[ ]:


pred /= 5


# In[ ]:


sub = pd.read_csv('/kaggle/input/infopulsehackathon/sample_submission.csv')


# In[ ]:


sub['Energy_consumption'] = pred[0]
sub.to_csv('submission.csv',index=False)


# And in conclusion: you shouldn't have trusted your validation on this competition :)
# Although I do think, that the way the data was split into private and public leaderboards is somewhat wrong. If you remove only 400 rows with extreme values through z-score, your local validation will go up. Sure thing, because it is way easier for the model and you are plainly overfitting. But the results from this removal on LB were way too big. Just because of those outliers you'd gain around 200-300K MSE on LB.
# And a constant theme on this competition was public LB score being much better, than local validation score. You can see above that some folds score have a drastic MSE, when others have a very low score. I'd like to hear more from the organizers on how exactly they split their data, that public and private LBs score much better, than local validation. Usually it's the other way around.
