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


# In[ ]:


PATH_DATA = '../input/covid19-global-forecasting-week-5/'
RANDOM_STATE = 0


# In[ ]:


def MakeDataFrame(file, Path = PATH_DATA):
    df = (pd.read_csv(f'{PATH_DATA}{file}.csv'
                      , sep = ','
                      , header = 0)
         )
    
    df.Date = pd.to_datetime(df.Date).dt.strftime("%Y%m%d").astype(int)
    df.Country_Region = df.apply(lambda x: x.Country_Region if pd.isnull(x.Province_State) else x.Province_State, axis=1)
    
    return df

df = MakeDataFrame('train')

df.head()


# In[ ]:


XE, XT, ye, yt = train_test_split(df.iloc[:, 3:-1]
                                  , df['TargetValue']
                                  , test_size = 0.2
                                  , random_state = RANDOM_STATE)

XE.head()


# ## Make Pipeline
# 
# * `StandardScaler`to standardize numericals features.
# * `OneHotEncoder` to make dummies in categorical features.
# * `ColumnTransformer` to run previus two steps.
# * The estimator `RandomForestRegressor`.

# In[ ]:


Nums = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'median')), 
        ('scaler' , StandardScaler())
    ])

Text = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')), 
        ('onehot' , OneHotEncoder(handle_unknown = 'ignore'))
    ])

Transformer = ColumnTransformer(
    n_jobs = -1
    , transformers = [
        ('num', Nums, XE.select_dtypes(include = ['int64','float64']).columns)
        , ('cat', Text, XE.select_dtypes(include = ['object']).columns)
    ])

Model = Pipeline(
    steps = [
        ('Prepo', Transformer)
        , ('Clf', RandomForestRegressor(n_jobs = -1
                                        , random_state = RANDOM_STATE)) 
    ])

Model.fit(XE, ye)

Model.score(XT, yt)


# ## Make Submission

# In[ ]:


df = MakeDataFrame('test')

df.head()


# In[ ]:


pred = Model.predict(df[XE.columns.tolist()])

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

