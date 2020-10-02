#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBRFRegressor


# In[ ]:


train=pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test=pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
submission=pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


RANDOM_STATE = 0


# In[ ]:


def FeatureEngineering(df):
  
    df.Date = pd.to_datetime(df.Date).dt.strftime("%Y%m%d").astype(int)
    df.Country_Region = df.apply(lambda x: x.Country_Region if pd.isnull(x.Province_State) else x.Province_State, axis=1)
    
    return df

df = FeatureEngineering(train)

df.head()


# In[ ]:


X_train,X_val,y_train,y_val = train_test_split(train.iloc[:, 3:-1]
                                  , train['TargetValue']
                                  , test_size = 0.2
                                  , random_state = RANDOM_STATE)

X_train.head()


# In[ ]:


num_pipe = Pipeline([   
        ('imputer', SimpleImputer(strategy = 'median')), 
        ('scaler' , StandardScaler())
])

cat_pipe = Pipeline([ 
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')), 
        ('encoder' , OneHotEncoder(handle_unknown = 'ignore'))
])

Transformer = ColumnTransformer(
    n_jobs = -1
    , transformers = [
        ('num', num_pipe, X_train.select_dtypes(include = ['int64','float64']).columns)
        , ('cat', cat_pipe, X_train.select_dtypes(include = ['object']).columns)
    ])

Model = Pipeline([ 
        ('preprocessor', Transformer)
        , ('predictor', XGBRegressor(n_jobs = -1
                                        , random_state = RANDOM_STATE)) 
])

Model.fit(X_train, y_train)

Model.score(X_val, y_val)


# In[ ]:


df = FeatureEngineering(test)

df.head()


# In[ ]:


pred = Model.predict(df[X_train.columns.tolist()])

df = pd.DataFrame({'Id': df.index, 'TargetValue': pred.tolist()})


# In[ ]:


q = ['0.05', '0.5', '0.95']
dfq = None

for i in q:
    k = (df.groupby(['Id'])['TargetValue'].quantile(q=float(i)).reset_index()
         .rename(columns = {'TargetValue':i})
        )
    if dfq is None:
        dfq = k
    else:
        dfq = pd.concat([dfq, k[i]], 1)
        
dfq.Id = dfq.Id + 1

dfq.head()


# In[ ]:


df = (pd.melt(dfq, id_vars = ['Id'], value_vars = q)
      .rename(columns = {'value':'TargetValue'})
     )

df['ForecastId_Quantile'] = df['Id'].astype(str) + '_' + df['variable']

df[['ForecastId_Quantile','TargetValue']].to_csv('submission.csv', index = False)

