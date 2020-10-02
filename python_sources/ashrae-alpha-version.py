#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import datetime as dt 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from tqdm import tqdm
import lightgbm as lgb
import datetime as dt

pd.options.display.float_format = '{:.4f}'.format


# In[ ]:


root = "../input/ashrae-energy-prediction/"

#read data from csv files
train = pd.read_csv(root + 'train.csv')
test = pd.read_csv(root + 'test.csv')
w_train = pd.read_csv(root + 'weather_train.csv', index_col=False)
w_test = pd.read_csv(root + 'weather_test.csv', index_col=False)
building = pd.read_csv(root + 'building_metadata.csv')

#change the categorical primary use column to numerical with label encoder
le = LabelEncoder()
building.primary_use = le.fit_transform(building.primary_use)


# In[ ]:


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))


# In[ ]:


reduce_mem_usage(train, verbose=True)
reduce_mem_usage(test, verbose=True)
reduce_mem_usage(w_train, verbose=True)
reduce_mem_usage(building, verbose=True)
reduce_mem_usage(w_test, verbose=True)


# # Merge

# In[ ]:


train = train.merge(building, on='building_id', how='left')
train = train.merge(w_train, on=['site_id', 'timestamp'], how='left')


# In[ ]:


test = test.merge(building, on='building_id', how='left')
test = test.merge(w_test, on=['site_id', 'timestamp'], how='left')


# In[ ]:


# We will need this row for our submission, 
#but it is not in the training dataset.
#We want to drop it for our analysisso we will assign it a variable to recall later 
row_id = test.row_id


# In[ ]:


test.drop(columns=['row_id'], axis=1, inplace=True)


# # EDA

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# The dataset contains measurement error. We will need to look for ouliers produced by error. Over 9% of the meter readings are zero, but a building in use should not have a meter reading of zero. Either the reading is an error, or the meter was not read at that particular time. In the latter case, we will need to gather the weather data between non-zero readings to arrive at an accurate assessment of the readings.

# In[ ]:


print("Percent equal to zero:", (train[train.meter_reading == 0].shape[0] / len(train.meter_reading))*100)


# The meter_reading column records the difference energy use, not the aggregate as domestic electric meter does, where any reading must be taken as the difference between the prior reading to arrive at interval usage. We will take a look at buiding_id 2 to get a sense of the readings.

# In[ ]:


# An example of meter readings for building_id 2
train[(train.building_id==2) & train.meter_reading>0].iloc[:100]


# While considering the outliers, we will take a look at the percentage of meter readings above the mean.

# In[ ]:


print("Percent above mean:", (train[train.meter_reading > train.meter_reading.mean()].shape[0] / train.shape[0]) * 100)


# In[ ]:


train.groupby('building_id').meter_reading.mean().sort_values()


# A better assessment of outliers would be to take the meter reading as a proportion of the building size, and we will tackle this in the Beta version.

# The following shows the percent of nan's in each column for the train and test dataframes. We will drop those with a percentage.

# In[ ]:


train.isnull().sum() * 100 / len(test)


# In[ ]:


test.isnull().sum() * 100 / len(test)


# # Visualize data
# ##### just a brief word on a few variables
# The scatter matrices that follow are samples of 5000 data points. The target variable is the meter reading column. This version includes all meter types (elec, chilledwater). In the Beta version, we will explore the correlations within each meter type.

# Dew temperature is the air temperature at which water in the air condenses. The higher the dew temperature the more "muggy" it feels. The data approaches a normal distribution (see, https://www.csemag.com/articles/controlling-dew-point/)

# Sea-level pressure lowers as altitude rises. Because the air is thinner at higher altitudes, it is less effective at heating at building. It takes more energy to heat and cool a building at a higher altitude than it does at sea level. For sea-level pressure, see https://www.britannica.com/science/atmospheric-pressure; https://w1.weather.gov/glossary/index.php?word=sea+level+pressure. For altitude and energy costs, see https://www.achrnews.com/articles/110408-selecting-heating-cooling-units-for-high-altitude-homes; http://www.comairrotron.com/solving-high-altitude-cooling-problems

# In[ ]:


#create a sample dataframe that filters our values greater than 5000 and 
# not equal to 0, so that our vizualizations are to scale
sample = train[(train.meter_reading < 5000)&(train.meter_reading > 0)] 
sample.sample(n=1000, random_state=1).hist(bins=50, figsize=(20,15)) 


# In[ ]:


# output a scatter matrix to see correlation between variables
# meter reading is the target variable
# the independent variables differ between each
from pandas.plotting import scatter_matrix
attributes = ['meter_reading', 'building_id', 'meter', 'site_id', 'primary_use']
scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))


# In[ ]:


attributes = ['meter_reading','square_feet', 'year_built', 'floor_count']
scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))


# In[ ]:


attributes = ['meter_reading', 'air_temperature', 'cloud_coverage', 'dew_temperature']
scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))


# In[ ]:


attributes = ['meter_reading', 'sea_level_pressure', 'wind_direction', 'wind_speed']
scatter_matrix(sample[attributes].sample(n=100, random_state=1), figsize=(12, 8))


# The above illustrations shows that the square_feet column has right skew.

# In[ ]:


# the distribution of a sample from the square-feet column
s = train['square_feet'].dropna(axis=0)
s = s.sample(n=1000000, random_state=1)
sns.distplot(a=s, kde=True)


# We can cube the data to get us closer a normal distribution (better than log(1+x)) and then replace the original values with the transformation.

# In[ ]:


# the distribution of the data behind the illustration cubed
s = np.cbrt(s)
sns.distplot(a=s, kde=True)


# In[ ]:


#transform column by cubing all values
train.square_feet = np.cbrt(train.square_feet)
test.square_feet = np.cbrt(test.square_feet)


# # Feature Selection

# Sea-pressure lowers with altitdue. More energy is needed to heat air that is less dense. We divide air_temp by sea_level so that lower sea level pressures translates to a higher output. We square the results to remove the negative values.

# In[ ]:


train['sea_temp'] = np.sqrt(np.square(train['air_temperature'] / train.sea_level_pressure * 100))
test['sea_temp'] = np.sqrt(np.square(test['air_temperature'] / test.sea_level_pressure * 100))


# Here we will try to approximate the air temperature that best regulates the building temperature without energy use in colder months. Sun exposure and heat produced naturally within the building (electronics, humans, etc. ) will lower the air temp needed to regulate a building. Eighteen degrees celcius is an comfortable indoor temperature. So, to get at a true zero temp we will shift the measurements down by 22 degrees then multiply these numbers by square feet so that any variance will be exagerrated. 

# In[ ]:


train['internal'] = np.square(train.air_temperature - 22)
test['internal'] = np.square(test.air_temperature - 22)


# We divide the new variable, internal temperature, by the square feet of the building to approximate a ratio between building size and temperature. We square the results to remove negative values.

# In[ ]:


train['air_sq'] = np.square(train.internal / train.square_feet)
test['air_sq'] = np.square(test.internal / test.square_feet)


# In[ ]:


train.columns


# We will take a look at the correlation of these new columns to the target variable (meter_reading). 

# In[ ]:


# update the sample variable to include the new features
sample = train[(train.meter_reading < 5000)&(train.meter_reading > 0)] 


attributes = ['meter_reading', 'sea_temp', 'air_sq']
scatter_matrix(sample[attributes].sample(n=10000, random_state=1), figsize=(12, 8))


# # Drop Features

# The buildings represented in the dataset are located around the world (15 site_ids). This means that meterological and astronomical seasons of the northern and souther hemispheres differ by two seasons (summer = winter). Seasons, months, days, and years are inherently linked, so we will remove them. Hours and weekend column provide no additional correlation between readings. Other columns with a large number of missing values are also dropped. 

# In[ ]:


drop_train = ['floor_count', 'timestamp', 'year_built', 'cloud_coverage', 'precip_depth_1_hr']
drop_test = ['floor_count', 'timestamp', 'year_built', 'cloud_coverage', 'precip_depth_1_hr']

train.drop(drop_train, axis=1, inplace=True)
test.drop(drop_test, axis=1, inplace=True)


# In[ ]:


print(train.shape)
print(test.shape)


# # Model

# In[ ]:


#X = train.drop(columns=['meter_reading'])
#y = train['meter_reading']


# In[ ]:


#from sklearn.model_selection import train_test_split, GridSearchCV
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1)


# In[ ]:


#categorical_features = ["building_id", "site_id", "meter", "primary_use", "hour", "weekday"]

#lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)

#lgb_test = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, free_raw_data=False)

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 30,
    "learning_rate": 0.1,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"
}

#model = lgb.train(params, train_set=lgb_train,  num_boost_round=1000, valid_sets=[lgb_train, lgb_test], verbose_eval=200, early_stopping_rounds=200)

#predictions = model.predict(X_test, num_iteration=model.best_iteration)


# In[ ]:


#lgb_train_full = lgb.Dataset(X, label=y, categorical_feature=categorical_features, free_raw_data=False)

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.1,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"
}

#model2 = lgb.train(params, train_set=lgb_train_full,  num_boost_round=2000)

#predictions = model2.predict(test, num_iteration=model.best_iteration)


# # Submission

# In[ ]:


#submission = pd.DataFrame({'row_id':row_id, 'meter_reading':predictions})
#submission.loc[submission.meter_reading < 0, 'meter_reading'] = 0
#submission.to_csv('/Users/DataScience/energy/lgbm4.csv', index=False)


# In[ ]:


#pd.read_csv('/Users/DataScience/energy/lgbm2.csv').head()


# In[ ]:


#submission.head(10)

