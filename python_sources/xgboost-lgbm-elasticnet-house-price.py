#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.shape


# Deleting 'PoolQC' feature since its fully null

# In[ ]:


del train_data['PoolQC']


# In[ ]:


del test_data['PoolQC']


# In[ ]:


del train_data['Id']


# Extracting text cols

# In[ ]:


list_of_categorical = []
list_of_numerical = []
for i in train_data.columns:
    if train_data[i].dtypes == 'object':
        list_of_categorical.append(i)
    else:
        if i == 'SalePrice':
            pass
        else:
            list_of_numerical.append(i)


# In[ ]:


len(list_of_categorical)


# In[ ]:


len(list_of_numerical)


# # Pipeline building

# In[ ]:


from sklearn.feature_extraction import DictVectorizer


# In[ ]:


from sklearn_pandas import DataFrameMapper, CategoricalImputer


# In[ ]:


from sklearn.pipeline import Pipeline, FeatureUnion


# In[ ]:


import xgboost as xgb


# In[ ]:


from sklearn.preprocessing import Imputer, FunctionTransformer


# In[ ]:


from sklearn.impute import SimpleImputer


# In[ ]:


# idea taken from https://stackoverflow.com/a/52090830/7886239

transformers = []

transformers.extend([([num_feat],SimpleImputer(strategy='median')) for num_feat in list_of_numerical])
transformers.extend([(cat_feat,CategoricalImputer()) for cat_feat in list_of_categorical])

combined_pipeline = DataFrameMapper(transformers,
                                   input_df=True,
                                   df_out=True)


# In[ ]:


def return_dict(blob):
    return blob.to_dict("records")


# # Model

# **using xgboost to predict prices**

# In[ ]:


pipeline = Pipeline([
    ('featureunion',combined_pipeline),
    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),
    ('vectorizer',DictVectorizer(sort=False)),
    ('reg',xgb.XGBRegressor(objective="reg:squarederror"))
])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop(['SalePrice'],axis=1), train_data['SalePrice'], test_size=0.15, random_state=42)


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


predictions = pipeline.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_log_error


# In[ ]:


print(np.sqrt(mean_squared_log_error(y_test,predictions)))


# **Our base model is working great so lets use RandomizedSearchCv to furthur improve it**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params = {
    'reg__max_depth' : [5,10,15,20],
    'reg__gamma' : [.1,.2,.3,.4,.5,.6],
    'reg__colsample_bytree' : [.1,.2,.3,.4,.5,.6,.7],
    'reg__reg_alpha' : np.arange(0,0.6,0.1)
}


# In[ ]:


grid_pipe = RandomizedSearchCV(estimator=pipeline,param_distributions=params,cv=4,
                               scoring="neg_mean_squared_log_error",n_iter=15)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid_pipe.fit(X_train,y_train)')


# In[ ]:


grid_pipe.best_params_


# In[ ]:


grid_pipe.best_score_


# In[ ]:


pipe_predict = grid_pipe.predict(X_test)


# In[ ]:


print(np.sqrt(mean_squared_log_error(y_test,pipe_predict)))


# **Lets use lgbm to predict prices**

# In[ ]:


import lightgbm as lgb


# Building a pipeline for lgbm

# In[ ]:


pipeline_lgm = Pipeline([
    ('featureunion',combined_pipeline),
    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),
    ('vectorizer',DictVectorizer(sort=False)),
    ('reg_lgb',lgb.LGBMRegressor())
])


# In[ ]:


pipeline_lgm.fit(X_train,y_train)


# In[ ]:


pipeline_lgm_predict = pipeline_lgm.predict(X_test)


# In[ ]:


print(np.sqrt(mean_squared_log_error(y_test,pipeline_lgm_predict)))


# Lets fine tune this regressor too

# In[ ]:


params_lgb = {
    'reg_lgb__learning_rate' : [.1,.2,.3,.4,.5,.6],
    'reg_lgb__n_estimators' : [50,100,200],
    'reg_lgb__colsample_bytree' : [.1,.2,.3,.4,.5,.6,.7],
    'reg_lgb__reg_lambda' : np.arange(0,0.6,0.1)
}


# In[ ]:


grid_pipe_lgm = RandomizedSearchCV(estimator=pipeline_lgm,param_distributions=params_lgb,cv=4,
                               scoring="neg_mean_squared_log_error",n_iter=15)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid_pipe_lgm.fit(X_train,y_train)')


# In[ ]:


grid_pipe_lgm.best_params_


# In[ ]:


grid_pipe_lgm.best_score_


# In[ ]:


lgm_pipe_predict = grid_pipe_lgm.predict(X_test)


# In[ ]:


print(np.sqrt(mean_squared_log_error(y_test,lgm_pipe_predict)))


# **Now predicting prices with ElasticNet Regressor**

# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


pipeline_elasticnet = Pipeline([
    ('featureunion',combined_pipeline),
    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),
    ('vectorizer',DictVectorizer(sort=False)),
    ('reg_en',ElasticNet())
])


# In[ ]:


pipeline_elasticnet.fit(X_train,y_train)


# In[ ]:


predicted_base_elastic_net = pipeline_elasticnet.predict(X_test)


# In[ ]:


print(np.sqrt(mean_squared_log_error(y_test,predicted_base_elastic_net)))


# Fine tuning elastic net

# In[ ]:


params_elasticnet = {
    'reg_en__l1_ratio' : [0.2,0.4,0.6,0.8],
    'reg_en__alpha' : [0.5,1,1.5,2],
}


# In[ ]:


grid_pipe_elastic = RandomizedSearchCV(estimator=pipeline_elasticnet,param_distributions=params_elasticnet,cv=4,
                               scoring="neg_mean_squared_error")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid_pipe_elastic.fit(X_train,y_train)')


# In[ ]:


grid_pipe_elastic.best_params_


# In[ ]:


grid_pipe_elastic.best_score_


# In[ ]:


print(np.sqrt(mean_squared_log_error(y_test,grid_pipe_elastic.predict(X_test))))


# # Ensemble of regressors

# **Building final model with best parameters and calculating on full train data with voting regressor**

# In[ ]:


from sklearn.ensemble import VotingRegressor


# In[ ]:


xgb_final = xgb.XGBRegressor(objective="reg:squarederror",
                        reg_alpha = 0.1,
                        reg__max_depth =  5,
                        reg__gamma =  0.6,
                        reg__colsample_bytree =  0.4)


# In[ ]:


lgm_final = lgb.LGBMRegressor(reg_lambda = 0.2,
                             n_estimators = 200,
                             learning_rate = 0.1,
                             colsample_bytree = 0.3)


# In[ ]:


elastic_final = ElasticNet(alpha=.5,l1_ratio=.6)


# In[ ]:


final_regressor = VotingRegressor(estimators=[
    ('xgb',xgb_final),
    ('lgm',lgm_final),
    ('elastic',elastic_final)
])


# In[ ]:


pipeline_final = Pipeline([
    ('featureunion',combined_pipeline),
    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),
    ('vectorizer',DictVectorizer(sort=False)),
    ('regressor',final_regressor)
])


# In[ ]:


X = train_data.drop(['SalePrice'],axis=1)
y = train_data['SalePrice']


# In[ ]:


Id = test_data['Id']


# # Submission

# In[ ]:


test_final = test_data.drop(['Id'],axis=1)


# In[ ]:


pipeline_final.fit(X,y)


# In[ ]:


final_preds = pipeline_final.predict(test_final)


# In[ ]:


submission = pd.DataFrame({ 'Id': Id,
                            'SalePrice': final_preds })
submission.to_csv(path_or_buf ="Housing_Regression.csv", index=False)


# In[ ]:




