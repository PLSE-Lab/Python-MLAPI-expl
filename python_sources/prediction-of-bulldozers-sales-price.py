#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import sklearn


# # Predicting the sales Price of Bulldozers

# In[ ]:


df=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False)


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


fig,ax =plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])


# In[ ]:


df.SalePrice.plot.hist()


# In[ ]:


df.columns


# ## parsing dates

# In[ ]:


df=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])


# In[ ]:


df.saledate.head()


# In[ ]:


fig,ax =plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])


# In[ ]:


df.head()


# In[ ]:


df.saledate.head(20)


# In[ ]:


df.sort_values(by='saledate',inplace=True,ascending=True)


# In[ ]:


df_temp=df.copy()


# In[ ]:


df_temp['saleYear']=df_temp.saledate.dt.year


# In[ ]:


df_temp['saleMonth']=df_temp.saledate.dt.month
df_temp['saleDay']=df_temp.saledate.dt.day
df_temp['saleDayofWeek']=df_temp.saledate.dt.dayofweek
df_temp['saleDayofYear']=df_temp.saledate.dt.dayofyear


# In[ ]:


df_temp.head().T


# Removing sale Date

# In[ ]:


df_temp.drop('saledate',axis=1,inplace=True)


# In[ ]:


#find columns which contain string
for label , content in df_temp.items():
    if pd.api.types.is_string_dtype(content):
        df_temp[label]=content.astype('category').cat.as_ordered()
       


# In[ ]:


df_temp.info()


# In[ ]:


df_temp.state.cat.categories


# In[ ]:


df_temp.isnull().sum()/len(df_temp)


# In[ ]:


#fill numeric rows with median
for label , content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_temp[label]=content.fillna(content.median())
       


# In[ ]:


#Turn categories into number and fill missing values
for label , content in df_temp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_temp[label]=pd.Categorical(content).codes +1
        
            


# In[ ]:


df_temp.isna().sum() #no  more missing values


# In[ ]:


len(df_temp)


# In[ ]:



from sklearn.ensemble import RandomForestRegressor


# In[ ]:



get_ipython().run_cell_magic('time', '', "model=RandomForestRegressor(n_jobs=-1,random_state=42)\n\nmodel.fit(df_temp.drop('SalePrice',axis=1),df_temp['SalePrice'])")


# In[ ]:


#Score the model
model.score(df_temp.drop('SalePrice',axis=1),df_temp['SalePrice'])


# ## Splitting data into training and validation set
# 

# In[ ]:


df_val = df_temp[df_temp.saleYear==2012]
df_train = df_temp[df_temp.saleYear !=2012]
len(df_val),len(df_train)


# In[ ]:


#Split data into x and Y
X_train,y_train=df_train.drop('SalePrice',axis=1),df_train['SalePrice']
X_valid,y_valid=df_val.drop('SalePrice',axis=1),df_val['SalePrice']


# In[ ]:


X_train.shape,y_train.shape,X_valid.shape,y_valid.shape


# ## Builing an evaluation function

# In[ ]:


#Creating an evalutaion function (competion uses RMSLE)
from sklearn.metrics import mean_squared_log_error,mean_absolute_error,r2_score
def rmsle(y_test,y_pred):
    """
    Calculates root mena squared log error
    """
    return np.sqrt(mean_squared_log_error(y_test,y_pred))
#Creating function to evaluate model on few different levels
def show_scores(model):
    train_preds=model.predict(X_train)
    val_preds=model.predict(X_valid)
    scores={"Training MAE":mean_absolute_error(y_train,train_preds)
            ,"Valid MaE":mean_absolute_error(y_valid,val_preds),
           "Training RMSLE":rmsle(y_train,train_preds),
           "valid RMSLE": rmsle(y_valid,val_preds),
           "Training R^2":r2_score(y_train,train_preds),
           "Valid R^2":r2_score(y_valid,val_preds)}
    return scores


# ## testing our model on a subset(to tune hyperparameters)

# In[ ]:





# In[ ]:


#Change max samples values
model=RandomForestRegressor(n_jobs=-1,
                           random_state=42,
                           max_samples=10000)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train,y_train)')


# In[ ]:


show_scores(model)


# ## Hyperparameter tuning wiwth RandomisedSearchCV

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n#Different RandomForestRegressor hyberparameters\nrf_grid={"n_estimators":np.arange(10,100,10),\n         "max_depth":[None,3,5,10],\n         "min_samples_split":np.arange(2,20,2),\n         "min_samples_leaf":np.arange(1,20,2),\n         "max_features":[0.5,1,"sqrt","auto"],\n         "max_samples":[10000]\n    \n}\nrs_model=RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                 random_state=42),\n                                                 param_distributions=rf_grid,\n                           n_iter=50,\n                           cv=5,\n                           verbose=True)\nrs_model.fit(X_train,y_train)')


# In[ ]:


rs_model.best_params_


# In[ ]:


show_scores(rs_model)


# In[ ]:


ideal_model=RandomForestRegressor(n_estimators=40,
                                 min_samples_leaf=1,
                                 min_samples_split=14,
                                 max_features=0.5,
                                 n_jobs=-1,
                                 max_samples=None)
ideal_model.fit(X_train,y_train)


# In[ ]:


#trained on all the data
show_scores(ideal_model)


# ### Make prediction on test data

# In[ ]:


df_test=pd.read_csv("/kaggle/input/bluebook-for-bulldozers/Test.csv",
                   low_memory=False,
                   parse_dates=["saledate"])
df_test.head()


# In[ ]:


test_preds=ideal_model.predict(df_test)


# In[ ]:


df_test.isna().sum()


# ## geting test dataset same as training dataset

# In[ ]:


def preprocess_data(df):
    """
    Perform trasnsormation on df and returns transformed df.
    """
    df['saleYear']=df.saledate.dt.year
    df['saleMonth']=df.saledate.dt.month
    df['saleDay']=df.saledate.dt.day
    df['saleDayofWeek']=df.saledate.dt.dayofweek
    df['saleDayofYear']=df.saledate.dt.dayofyear
    df.drop('saledate',axis=1,inplace=True)
    #find columns which contain string
    for label , content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label]=content.astype('category').cat.as_ordered()
    #fill numeric rows with median
    for label , content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                 df[label]=content.fillna(content.median())	
    #Turn categories into number and fill missing values
    for label , content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            df[label]=pd.Categorical(content).codes +1
    
    return df


# In[ ]:


preprocess_data(df_test)


# In[ ]:


df_test.isna().sum()


# In[ ]:


test_preds=ideal_model.predict(df_test)


# In[ ]:


test_preds


# In[ ]:


#formatting predictions
df_preds=pd.DataFrame()
df_preds['SalesID']=df_test['SalesID']
df_preds['SalePrice']=test_preds
df_preds


# In[ ]:


#export
df_preds.to_csv("/kaggle/working/test_predictions.csv",index=False)


# In[ ]:




