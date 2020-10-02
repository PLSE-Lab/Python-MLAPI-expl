#!/usr/bin/env python
# coding: utf-8

# ## ASHRAE - Greate Energy Predictor
# EXPLORATORY DATA ANALYSIS NOTEBOOK <br/>
# with Random Forest Modeling
# 
# Assessing the value of energy efficiency improvements can be challenging as there's no way to truly know how much energy a building would have used without the improvements. We will be examining the four energy types based on historic usage rates and observed weather. The dataset includes three years of hourly meter readings from over one thousand buildings at several different sites around the world.
# 
# To see fill data description click this link: https://www.kaggle.com/c/ashrae-energy-prediction/data
# 
# 
# *Author notes: The analysis will include the energy types, usage rates, weather and building sites (?).*

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

from sklearn.model_selection import train_test_split as train_valid_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import eli5
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Energy
# This section will include data cleaning for the energy dataset. Also, included in this section is the analysis of meter reading in realtion to buildings
# 
# For data cleaning we need to change timestamp from object to datetime. Change meter to catgories, since its the 4 energy types that the data description was talking about

# In[ ]:


energy = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
energy.timestamp = pd.to_datetime(energy.timestamp)
energy.meter = energy.meter.astype('category')
energy.info()


# We want to know if the rows for each building are equally distributed. As we can see below that around 700 buildings have around 8000 rows, but we can also see some building ids that have 35000 rows in the dataframe

# In[ ]:


g = energy.building_id.value_counts()
plt.hist(g.values,bins=100)
plt.xlabel('Total rows of a building id')
plt.ylabel('Number building ids')
plt.show()


# Meter is the 4 types of energy that the data description was talking about... As we can see here Meter Type 0 is the dominant type for all buildings.

# In[ ]:


g = energy.meter.value_counts()
plt.bar(g.index,g.values)
plt.xlabel('Meter Type')
plt.ylabel('Count')
plt.show()


# Since the meter reading varies a lot. We can see the distribution by getting the log of the values to remove the huge differences.

# In[ ]:


g = energy[['meter','meter_reading']]
g['meter_reading'] = np.log1p(g['meter_reading'])
sns.boxplot(x='meter',y='meter_reading',data=g)
plt.plot();


# ## Weather
# This section will include data cleaning for the weather dataset. Also, included in this section is the analysis of weather variables
# 
# For data cleaning we need to change timestamp from object to datetime.

# In[ ]:


weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
weather.timestamp = pd.to_datetime(weather.timestamp)
weather.info()


# In terms of correlation of weather variables, only the air and dew temperature have significant correlation

# In[ ]:


g = weather.drop(['site_id','timestamp'],axis=1).corr()
plt.figure(figsize=(12,10))
sns.heatmap(g,annot=True,center=0,cmap='Blues');


# ## Building Information
# This section will include the analysis of buildings and its variables

# In[ ]:


building_info = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
building_info.info()


# Wow! The dataset contains buildings that were made in the 1900s. I thought the buildings will be just newer ones.

# In[ ]:


g = building_info.year_built.value_counts()
plt.bar(g.index,g.values)
plt.xlabel('Year Built')
plt.ylabel('Count')
plt.show()


# Majority of the buildings' primary usage is for Education

# In[ ]:


g = building_info.primary_use.value_counts()
plt.barh(g.index,g.values)
plt.xlabel('Building Primary Use')
plt.ylabel('Count')
plt.show()


# ## Random Forest Pipeline
# 
# This section will include modeling preparation, modeling, modeling evaluation

# In[ ]:


train = pd.merge(energy,building_info,on='building_id',how='left')
train = pd.merge(train,weather,on=['site_id','timestamp'],how='left')
del energy,weather
train.tail()


# ### Data prep for Modeling
# 
# Target variable: meter_reading<br/>
# Encode primary_use variables for modeling<br/>
# Convert the timestamp to datetime

# In[ ]:


def dt_parts(df,dt_col):
    if(df[dt_col].dtype=='O'):
        df[dt_col] = pd.to_datetime(df[dt_col])
    df['year'] = df[dt_col].dt.year.astype(np.int16)
    df['month'] = df[dt_col].dt.month.astype(np.int8)
    df['day'] = df[dt_col].dt.day.astype(np.int8)
    df['hour'] = df[dt_col].dt.hour.astype(np.int8)
    df['minute'] = df[dt_col].dt.minute.astype(np.int8)
    df['second'] = df[dt_col].dt.second.astype(np.int8)
    df.drop(dt_col,axis=1,inplace=True)
    return df

#optimizing the column types to consume less space
def df_type_optimize(df):
    df['building_id'] = df['building_id'].astype(np.uint16)
    df['meter'] = df['meter'].astype(np.uint8)
    df['site_id'] = df['site_id'].astype(np.uint8)
    df['square_feet'] = df['square_feet'].astype(np.uint32)
    
    df['year_built'] = df['year_built'].astype(np.uint16)
    df['floor_count'] = df['floor_count'].astype(np.uint8)
    
    df['air_temperature'] = df['air_temperature'].astype(np.int16)
    df['cloud_coverage'] = df['cloud_coverage'].astype(np.int16)
    df['dew_temperature'] = df['dew_temperature'].astype(np.int16)
    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].astype(np.int16)
    df['sea_level_pressure'] = df['sea_level_pressure'].astype(np.int16)
    df['wind_direction'] = df['wind_direction'].astype(np.int16)
    df['wind_speed'] = df['wind_speed'].astype(np.int16)
    
    return df


# In[ ]:


train['primary_use'] = train['primary_use'].astype('category').cat.codes
train = dt_parts(train,'timestamp')
train.fillna(0,inplace=True)
train=df_type_optimize(train)
train.head()


# In[ ]:


target_col = 'meter_reading'
y = train[target_col]
Xs = train.drop(target_col,axis=1)

X_train, X_valid, y_train, y_valid = train_valid_split(Xs, y, test_size=0.2, random_state=0)
del train
X_train.shape,X_valid.shape


# ### Modeling
# 
# Due to Memory Error problems, I used this solution to reducing sub-sample size of random forest in sklearn <br/> [How can I set sub-sample size in Random Forest Classifier in Scikit-Learn? Especially for imbalanced data](https://stackoverflow.com/questions/44955555/how-can-i-set-sub-sample-size-in-random-forest-classifier-in-scikit-learn-espec#50914280)

# In[ ]:


#code reference above
from sklearn.ensemble import forest
def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))
set_rf_samples(130000)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = RandomForestRegressor(n_estimators=60,\n                              random_state=0,n_jobs=-1)\nmodel.fit(X_train,y_train)')


# ### Evaluation
# https://www.kaggle.com/c/ashrae-energy-prediction/overview/evaluation
# 
# The competition used RMSLE but for the mean time we will be using RMSE to evaluate the model

# In[ ]:


def RMSE(actual,preds):
    return np.sqrt(mean_squared_error(actual,preds))

def get_evaluations(model):
    preds = model.predict(X_train)
    plt.hist(np.log1p(preds),bins=100)
    plt.show();
    print('train_rmse: ',RMSE(y_train,preds))
                    
    preds = model.predict(X_valid)
    plt.hist(np.log1p(preds),bins=100)
    plt.show()
    print('valid_rmse: ',RMSE(y_valid,preds))
    
get_evaluations(model)


# ## Model Interpretation
# 
# This section we will utilize different technique to know how the model make its decision based from the data.
# Insights from this section can be useful for optimal feature engineering and improving the model accuracy.
# 
# 
# ### Feature Weights
# We can see here that the month,meter,dew_temperature,square_feet have top weight in the random forest

# In[ ]:


eli5.show_weights(model,feature_names=list(X_train.columns))


# ### Explaining model predictions
# 
# Out sample will be from out validation set. We can see here the contribution of each feature to the model's prediction of meter reading.

# In[ ]:


test_row = X_valid.loc[15256244,:]
test_row


# In[ ]:


eli5.show_prediction(model,test_row,feature_names=list(X_train.columns))


# ## Submission Pipeline

# In[ ]:


energy_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
test = pd.merge(energy_test,building_info,on='building_id',how='left')
test = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')
del energy_test,weather_test
test.tail()


# In[ ]:


test['primary_use'] = test['primary_use'].astype('category').cat.codes
test = dt_parts(test,'timestamp')
test.fillna(0,inplace=True)
test=df_type_optimize(test)
ids = test['row_id']
test.drop('row_id',axis=1,inplace=True)
test.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "preds = model.predict(test)\n\nsub_df = pd.DataFrame()\nsub_df['row_id'] = ids\nsub_df['meter_reading'] = preds\nsub_df.to_csv('the-sub-mission.csv',index=False)\nsub_df.head()")


# In[ ]:


plt.hist(np.log1p(sub_df['meter_reading']),bins=100)
plt.show()


# ## Notebook in progress
# 
# Do UPVOTE if this notebook is helpful to you in some way :) <br/>
# Comment below any suggetions that can help improve this notebook. TIA
