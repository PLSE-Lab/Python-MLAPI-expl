#!/usr/bin/env python
# coding: utf-8

# # Santander Value Prediction Challenge
#    Predict the value of transactions for potential customers.

# # Loading libraries and Dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')


# In[ ]:


print('Shape of training dataset: ',df_train.shape)
df_train.head()


# In[ ]:


print('Shape of test dataset: ',df_test.shape)
df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# ## Checking Missing Values

# Defining function to check missing values and percentage of missing values in each column.

# In[ ]:


def check_missing_data(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percent=((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending=False)
    return pd.concat([total,percent],axis=1,keys=['Total','Percent'])


# In[ ]:


check_missing_data(df_train).head()


# In[ ]:


check_missing_data(df_test).head()


# # Checking Unique Value in each column
# Column with only one unique value is useless. There for we can drop these columns

# In[ ]:


df_tmp=pd.DataFrame(df_train.nunique().sort_values(),columns=['num_unique_values']).reset_index().rename(columns={'index':'Column_name'})
df_tmp.head()


# In[ ]:


def col_name_with_n_unique_value(df,n):
    df1=pd.DataFrame(df.nunique().sort_values(),columns=['num_unique_values']).reset_index()
    col_name=list(df1[df1.num_unique_values==1]['index'])
    print('number of columns with only',n,'unique values are: ',len(col_name))
    return col_name


# In[ ]:


col_to_drop=col_name_with_n_unique_value(df_train,1)


# ### Droping unneccessary columns from train and test dataset

# In[ ]:


df_train.drop(columns=col_to_drop,inplace=True)
df_test.drop(columns=col_to_drop,inplace=True)
print('Shape of train dataset after droping columns: ',df_train.shape)
print('Shape of test dataset after droping columns: ',df_test.shape)


# # Getting Dataset in numpy ndarray format

# In[ ]:


train=df_train.iloc[:,2:].values
test=df_test.iloc[:,1:].values
target=df_train.target.values
print('Shape of train: ',train.shape)
print('Shape of target: ',target.shape)
print('Shape of test: ',test.shape)


# In[ ]:


del df_train,df_test,df_tmp
gc.collect()


# # Visualization of target Column

# In[ ]:


plt.hist(target, bins=20, color='c')
plt.title('Histogram : Target')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()


# Target values ranges from 0 to 4*1e7. there for taking log will reduce the range

# In[ ]:


t=np.log1p(target)
plt.hist(t, bins=20, color='c')
plt.title('Histogram : Target')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)


# # D. Splitting dataset into Train, val and Test set
# We split the dataset into train and val sets using an 80/20 split.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_sc, t, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(test_sc.shape, test_sc.shape)


# In[ ]:


del train_sc,train,test
gc.collect()


# # Machine Learning Models

# In[ ]:


Model_Summary=pd.DataFrame()


# ## 1. Lightgbm

# In[ ]:


import lightgbm
train_data=lightgbm.Dataset(X_train,y_train)
valid_data=lightgbm.Dataset(X_val,y_val)


# In[ ]:


params={'learning_rate':0.01,
        'boosting_type':'gbdt',
        'objective':'regression',
        'metric':'rmse',
        'sub_feature':0.5,
        'num_leaves':180,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'min_data':50,
        'max_depth':-1,
        'reg_alpha': 0.3, 
        'reg_lambda': 0.1, 
        'min_child_weight': 10, 
        'verbose': 1,
        'nthread':5,
        'max_bin':512,
        'subsample_for_bin':200,
        'min_split_gain':0.0001,
        'min_child_samples':5
       }


# In[ ]:


lgbm = lightgbm.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 80,
                 verbose_eval= 10
                 )


# In[ ]:


model_name='lightgbm'
RMSLE=np.sqrt(mean_squared_error(y_val,lgbm.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary


# In[ ]:


pred_lgbm=np.expm1(lgbm.predict(test_sc))
pred_lgbm


# ## 2. CatBoostRegressor

# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.1,
                             depth=5,
                             l2_leaf_reg=20,
                             bootstrap_type='Bernoulli',
                             subsample=0.6,
                             eval_metric='RMSE',
                             random_seed = 42,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=45)


# In[ ]:


cb_model.fit(X_train, y_train,
             eval_set=(X_val, y_val),
             use_best_model=True,
             verbose=True)


# In[ ]:


model_name='CatBoostRegressor'
RMSLE=np.sqrt(mean_squared_error(y_val,cb_model.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary


# In[ ]:


pred_cb=np.expm1(cb_model.predict(test_sc))
pred_cb


# ## 3. RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf_model = RandomForestRegressor(n_estimators=100)


# In[ ]:


rf_model.fit(X_train, y_train)


# In[ ]:


model_name='RandomForestRegressor'
RMSLE=np.sqrt(mean_squared_error(y_val,rf_model.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary


# In[ ]:


pred_rf=np.expm1(rf_model.predict(test_sc))
pred_rf


# ## 4. XGBoost

# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgb_model=XGBRegressor(max_depth=9)


# In[ ]:


xgb_model.fit(X_train, y_train)


# In[ ]:


model_name='xgboost'
RMSLE=np.sqrt(mean_squared_error(y_val,xgb_model.predict(X_val)))
Model_Summary=Model_Summary.append({'Model_Name':model_name,'RMSLE':RMSLE},ignore_index=True)
Model_Summary


# In[ ]:


pred_xgb=np.expm1(xgb_model.predict(test_sc))
pred_xgb


# In[ ]:





# # Generating Submision File

# **generating submission file for 4 model**

# In[ ]:


Model_Summary


# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv')
sub.target=(pred_lgbm+pred_cb+pred_rf+pred_xgb)/4.0
sub.head()


# In[ ]:


sub.to_csv('sub1.csv',index=False)


# In[ ]:




