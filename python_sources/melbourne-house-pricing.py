#!/usr/bin/env python
# coding: utf-8

# ## Melbourne House Princing 99% Accuracy

# This is my first Kernel done from start to end on Kaggle after doing the Kaggle courses and some others from Udemy.
# 
# This kernel pre processes the dataset of the Melbourne Housing Dataset and does the prediction using the Random Forest algorithm.
# 
# The maximum calculated error of the algorithm is +/- 2.8%
# The prediction mean accuracy is over 99%.
# 
# If you can review it, I would appreciate!
# 
# Hope you enjoy it.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Import the Dataset

# In[ ]:


#path = '../input/melbourne-housing-market/Melbourne_housing_FULL.csv' 
# Not working for the above dataset at the moment. Not all the preprocessing needed have been done yet.

path = '../input/melbourne-housing-snapshot/melb_data.csv'
raw_data = pd.read_csv(path)


# ## Checking the columns**

# In[ ]:


raw_data.head(5)


# In[ ]:


columns_of_interest = ['Rooms', 'Type', 'Distance', 'Bedroom2', 
                        'Bathroom', 'Car', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude', 'Regionname', 'Price']


# In[ ]:


df = raw_data.copy()
df.info()


# In[ ]:


df = df[columns_of_interest]
df.head()


# ## Checking the Ratio of Landsize vs BuildingArea

# In[ ]:


column_names = ['Landsize', 'BuildingArea']
df_ls_ba = df[column_names]
df_ls_ba = df_ls_ba.dropna(axis = 0)
df_ls_ba.info()


# In[ ]:


landsize_buildingarea_ratio = df_ls_ba['Landsize'].median()/df_ls_ba['BuildingArea'].median()
landsize_buildingarea_ratio


# In[ ]:


df_buildingarea = df.copy()


# In[ ]:


def remove_nan_buildingarea(row):
    return row['Landsize']/landsize_buildingarea_ratio if                 pd.isnull(row['BuildingArea']) else row['BuildingArea']


# In[ ]:


df_buildingarea['BuildingArea'] = df_buildingarea.apply(remove_nan_buildingarea, axis = 1)


# In[ ]:


df_buildingarea.head()


# In[ ]:


df_buildingarea.info()


# In[ ]:


for col in df.columns:
    print('Column: {}  |  Type: {}'.format(col, type(df[col][0])))


# ## Region Column Analysis

# In[ ]:


df_region = df_buildingarea.copy()
df_region['Regionname'].unique()


# In[ ]:


df_region['Regionname'].isnull().sum()


# In[ ]:


for row in df_region['Regionname']:
    row = 'Unknown' if row is None else row


# In[ ]:


df_region['Regionname'].isnull().sum()


# In[ ]:


def remove_nan_str(value):
    if isinstance(value, str):
        return value
    else:
        return 'Unknown' if pd.np.isnan(value) else value


# In[ ]:


df_region['Regionname'] = df_region['Regionname'].apply(remove_nan_str)
df_region['Regionname'].isnull().sum()


# In[ ]:


df_region['Regionname'].unique()


# In[ ]:


df_region['Regionname'].value_counts()


# In[ ]:


regionname_dummies = pd.get_dummies(df_region['Regionname'])
regionname_dummies.head()


# In[ ]:


#Removes the Unknown column and avoid multicollinearity

if regionname_dummies.columns.contains('Unknown'):
    regionname_dummies = regionname_dummies.drop(['Unknown'], axis = 1)
regionname_dummies.head()


# ## Checkpoint

# In[ ]:


df_region = pd.concat([df_region, regionname_dummies], axis = 1)
df_region = df_region.drop(['Regionname'], axis = 1)
df_region.head()


# ## Type Column Analysis

# In[ ]:


df['Type'].isnull().sum()


# In[ ]:


df['Type'].unique()


# In[ ]:


type_dummies = pd.get_dummies(df['Type'])
type_dummies


# In[ ]:


type_dummies.sum()


# In[ ]:


# Since 'h' is the most seen value, we'll remove it and consider as the default value, again
# avoiding multicollinearity


# In[ ]:


type_dummies = type_dummies.drop(['h'], axis = 1)


# In[ ]:


df_type = pd.concat([df_region, type_dummies], axis = 1)
df_type = df_type.drop(['Type'], axis = 1)
df_type.head()


# ## 'YearBuilt' column Analisys

# In[ ]:


df_YearBuilt = df_type.copy()


# In[ ]:


df_YearBuilt['YearBuilt'].median()


# In[ ]:


df_YearBuilt.info()


# In[ ]:


# Checking the possible ages of the data

print(df_YearBuilt['YearBuilt'].value_counts())


# In[ ]:


# Age Groups in Decades
# Up to:
# 0, 1, 2, 3, 5, 100, 9999 (Unknown)


# In[ ]:


import datetime

age_groups = {0, 1, 2, 3, 5, 10, 20, 100}

def divide_data_by_age_groups(year_built):
    if pd.isnull(year_built):
        return 9999
    age = datetime.datetime.now().year - year_built
    if (age % 10) >= 5:
        age_decades = ((age // 10) + 1)
    else:
        age_decades = (age // 10)

    for group in age_groups:
        if age_decades <= group:
            age_decades = group
            break
        
    #print('AGE {} | DECADES {}'.format(age, age_decades))
    return age_decades


# In[ ]:


df_YearBuilt['AgeInDecades'] = df_YearBuilt['YearBuilt'].apply(divide_data_by_age_groups)


# In[ ]:


df_YearBuilt.head(5)


# In[ ]:


decades_dummies = pd.get_dummies(df_YearBuilt['AgeInDecades'])
decades_dummies.head()


# In[ ]:


column_names = ['0_Decades_Old', '1_Decades_Old', '2_Decades_Old', '3_Decades_Old', '100_Decades_Old', 'Unknown_Decades_Old']
decades_dummies.columns = column_names
decades_dummies.head()


# In[ ]:


decades_dummies = decades_dummies.drop(['Unknown_Decades_Old'], axis = 1)
decades_dummies.head()


# In[ ]:


df_YearBuilt = df_YearBuilt.drop(['AgeInDecades', 'YearBuilt'], axis = 1)
df_YearBuilt.head()


# In[ ]:


df_YearBuilt = pd.concat([df_YearBuilt, decades_dummies], axis = 1)
df_YearBuilt.head()


# # Car Column Analisys

# In[ ]:


df_Car = df_YearBuilt.copy()
df_Car['Car'].isnull().sum()


# In[ ]:


# The 62 null values must be filled, let's use the median
car_median = df_YearBuilt['Car'].median()

def remove_nan_car(value):
    if pd.isnull(value):
        return car_median
    else:
        return value


# In[ ]:


print(df_Car['Car'].isnull().sum())
df_Car['Car'] = df_Car['Car'].apply(remove_nan_car)
print(df_Car['Car'].isnull().sum())


# In[ ]:


df_Car.head()


# In[ ]:


df_Car.isnull().sum()


# ## Columns reorganization

# In[ ]:


df_Car.columns.values


# In[ ]:


column_names = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
       'BuildingArea', 'Lattitude', 'Longtitude',
       'Eastern Metropolitan', 'Eastern Victoria',
       'Northern Metropolitan', 'Northern Victoria',
       'South-Eastern Metropolitan', 'Southern Metropolitan',
       'Western Metropolitan', 'Western Victoria', 't', 'u',
       '0_Decades_Old', '1_Decades_Old', '2_Decades_Old', '3_Decades_Old',
       '100_Decades_Old', 'Price']

df_Car = df_Car[column_names]
df_Car.head()


# ## Pre Processed Checkpoint

# In[ ]:


df_preprocessed = df_Car.copy()
df_preprocessed.head()


# ## Initiating Machine Learning

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# ## Splitting the dataset

# In[ ]:


raw_data = df_preprocessed.copy()

# Since BuildingArea has been filtered by a factor of the ratio of Landsize vs BuildingArea
# a nice check is to run the Kernel without this column, and check it's accuracy without it
raw_data = raw_data.drop('BuildingArea', axis = 1)

raw_data.head()


# In[ ]:


y = raw_data['Price']
raw_data.drop(['Price'], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(df_preprocessed, y,train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[ ]:


test_X.head(100)


# In[ ]:


model = RandomForestRegressor(random_state=900)
model.fit(train_X, train_y)
predictions = model.predict(test_X)


# In[ ]:


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=900)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# In[ ]:


score_dataset(train_X,test_X, train_y, test_y)


# In[ ]:


def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()


# In[ ]:


get_mae(test_X, test_y)


# In[ ]:


## Plotting Predicted vs Real


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[ ]:


diff = test_y - predictions
plt.plot(diff, 'rd')
plt.ylabel('Absolute Price Error ($)')
plt.show()


# In[ ]:


print(diff.max())
print(diff.min())


# As the above plotting shows, the model has adhered well to the data, and the predictions seems to be good.
# 
# Maximum error seems to be of around 100K, but...

# In[ ]:


diff_percent = ((test_y - predictions)/test_y)*100
plt.plot(diff_percent, 'rd')
plt.ylabel('Relative Price Error (%)')
plt.show()


# In[ ]:


print("Minimum error: {}".format(diff_percent.min()))
print("Maximum error: {}".format(diff_percent.max()))
error_mean = diff_percent.mean()
print("Error Mean: {}".format(diff_percent.mean()))
accuracy = (1 - diff_percent.mean())*100
print("Accuracy: {}".format(accuracy))


# ... the biggest error is under 3%
# 
# Accuracy = 99.51%
# 
# I'm very glad and hoping it's right! =]
