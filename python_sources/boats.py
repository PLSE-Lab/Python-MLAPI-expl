#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


from pymongo import MongoClient
from pymongo.errors import CursorNotFound, DuplicateKeyError
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import pickle, random, re, os


# In[2]:


def connect():
    client = MongoClient(host='mongodb://<dbuser>:<dbpassword>@ds237072.mlab.com:37072/gemi',
                         port=47450,
                         username='roxy',
                         password='gemicik1',
                         authSource='gemi',
                         authMechanism='SCRAM-SHA-1')

    db_name = 'gemi'
    db = client[db_name]
    collection_name = 'boats'
    collection = db[collection_name]
    
    return collection

collection = connect()


# In[3]:


collection.count()


# In[4]:


def get_ready(df):
        
    def clean_fields(df):
        df = df.dropna()
        
        df = df[df['engine hours'] != df['length']]

        df = df[df['engine hours'] != 'missing']
        df = df[df['country'] != '']

        df = df[pd.to_numeric(df['total power'], errors='coerce').notnull()]     
        df = df[pd.to_numeric(df['price'], errors='coerce').notnull()]   
        
        return df 
    
    df = clean_fields(df)
    
    def make_int(field):
        try:
            df[field] = df[field].astype(int)
        except TypeError as e:
            print(e)   
    
    integer_fields = [ 'price', 'year', 'length', 'total power', 'engine hours'] 
    for field in integer_fields:
        make_int(field)

    return df 


# In[156]:


query = {'year': {"$exists": True},
         'country': {"$ne": ""}, 
         'total power': {"$exists": True},
         'engine/fuel type':{"$exists": True},
         'engine hours': {"$exists": True},
        }
projection = {'_id':0, 'link':1, 'year':1, 'country':1, 'price':1, 'total power':1, 'length':1, 'model':1, 'engine/fuel type':1, 'engine hours':1, 'first_seen':1, 'removed':1}
cursor = collection.find(query, projection)


# In[157]:


df = pd.DataFrame(list(cursor))


# In[158]:


df = get_ready(df)


# In[159]:


df.count()


# In[9]:


df.head()


# In[10]:


model_counts = df.model.value_counts().rename_axis('model').reset_index(name='freq')
model_counts = model_counts[model_counts['freq'] >= 10]
model_counts.count()


# In[11]:


model_counts.head(10)


# In[12]:


sum(model_counts.freq.values)


# In[13]:


df.dtypes


# In[14]:


len(df.model.unique())


# In[15]:


from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             explained_variance_score,
                             mean_absolute_error,
                             mean_squared_log_error,
                             median_absolute_error)

from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold


# In[160]:


def get_X_y(boat_model):
    model_rows = df[df['model'] == boat_model]
    y = model_rows[['price']]
    X = model_rows[["country", "engine hours", "engine/fuel type", "year", "total power"]]
    X = pd.get_dummies(X, columns=['country', 'engine/fuel type'])
    return X, y 


# In[169]:


def get_regressors():
    regs = list()
    
    xgbr = XGBRegressor(loss ='lad', max_depth=5, n_estimators=200)
    regs.append((xgbr, 'XGBRegressor'))

    baggins = BaggingRegressor(base_estimator=xgbr, max_features=4)
    regs.append((baggins, 'BaggingRegressor'))

    rf = RandomForestRegressor()
    regs.append((rf, 'RandomForestRegressor'))
    
    return regs


# In[174]:


kf = KFold(n_splits=10)

regs = get_regressors()

for i in range(100):
    boat_model = model_counts.loc[i].model
    print(boat_model)
    
    X, y = get_X_y(boat_model)
    
    for train_index, test_index in kf.split(X):
        splits = (X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])
        X_train, X_test, y_train, y_test = splits
        
        errors = list()
        scores = list()
        preds = list()
        
        for reg in regs:
            regressor, name = reg
            
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            percentage_error = (np.median(np.abs( (y_test.price.values - y_pred) /  y_test.price.values) ) * 100).round(2)

            score = regressor.score(X_test, y_test).round(2)  

            print(percentage_error, score)
            
            errors.append(percentage_error)
            scores.append(score)
            preds.append(y_pred)
            
        min_index = errors.index(min(errors))
        y_pred = preds[min_index]
        error = errors[min_index]

        y_test['prediction'] = y_pred.astype(int)
        y_test['diff'] = (y_test['prediction'] - y_test['price'])/y_test['price']*100
        y_test['diff'] = y_test['diff'].astype(int)

        for index, row in tqdm(y_test.iterrows()):
            df.at[index,'prediction'] = row["prediction"]
            df.at[index,'diff' ] = row["diff"]
            df.at[index,'score' ] = score
            df.at[index,'error' ] = error

        
        


# In[236]:


def get_deals(good=True, rate=1.2, score_limit=0.9):
    deals = df[df["diff"].notnull()]
    
    deals['prediction'] = deals['prediction'].astype(int)
    
    if good:
        deals = deals[deals["diff"] > rate*deals["error"]]
    else:
        deals = deals[deals["diff"] < -rate*deals["error"]]
        
    deals = deals[deals.score > score_limit]
    
    on_sale = deals[deals["removed"] == False]
    sold = deals[deals["removed"] == True]
    
    print("deals", len(deals) )
    print("on_sale", len(on_sale) )
    print("rate", len(on_sale)/len(deals))
    
    deals = on_sale
    
    deals = deals[deals["first_seen"] > "2019-01-01"]
        
    deals = deals.sort_values('error', ascending=True)
            
    return deals


def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)

def cur(val):
    return "$ {:,}".format(val)
    
deals = get_deals(good=True, rate=1.5)

deals = deals[['year', 'model', 'length', 'country', 'engine hours', 'price', 'prediction', 'link', "score", "error", "diff", "first_seen", "removed"]]

deals.style.format({'link': make_clickable, 'price':cur, 'prediction':cur})


# In[235]:


deals = deals[['year', 'model', 'length', 'country', 'engine hours', 'price', 'prediction', 'link']]

deals.style.format({'link': make_clickable, 'price':cur, 'prediction':cur})


# In[218]:


len(deals)


# In[150]:


def train_test(reg, X, y ):
    regressor, name = reg
    
    X_train, X_test, y_train, y_test = X, X, y, y
        
    regressor.fit(X_train, y_train)
        
    y_pred = regressor.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test.price.values, y_pred)).round(2)
    
    percentage_error = (np.mean(np.abs( (y_test.price.values - y_pred) /  y_test.price.values) ) * 100).round(2)
    
    r2_score = regressor.score(X_test, y_test).round(2)  
    
    print(percentage_error, r2_score)
    

    def save_to_df():
        y_test['prediction'] = y_pred
        y_test['diff'] = (y_test['prediction'] - y_test['price'])/y_test['price']*100
        y_test['diff'] = y_test['diff'].astype(int)

        for index, row in tqdm(y_test.iterrows()):
            df.at[index,'prediction'] = row["prediction"]
            df.at[index,'diff' ] = row["diff"]
            df.at[index,'score' ] = r2_score
            df.at[index,'error' ] = r2_score
            
    save_to_df()
    
    return percentage_error, r2_score


# In[ ]:


def predict_model(i):
    boat_model = model_counts.loc[i].model
    print(boat_model)
    X, y = get_X_y(boat_model)

    regs = get_regressors()
    
    for reg in regs:
        train_test(reg, X, y)
    
        
def predict_prices(n): 
    for i in range(n):
        predict_model(i)
        
predict_prices(10)


# In[137]:


def k():
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        splits = (X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])


# In[ ]:


np.median(list(scores.values()))
np.median(list(errors.values()))


# In[ ]:


def stats(n):
    scores = model_counts['best r2 score'][:n]
    print('\n median r2 %', np.nanmedian(scores))
    print('\n mean r2 %', np.mean(scores))
    
    
    errors = model_counts['% error'][:n]
    print('\n median error %', np.nanmedian(errors))
    print('\n mean error %', np.mean(errors))
    

    fig, axs = plt.subplots(ncols=2)

    sns.boxplot(sorted(scores), ax=axs[0])
    sns.boxplot(sorted(errors), ax=axs[1])

