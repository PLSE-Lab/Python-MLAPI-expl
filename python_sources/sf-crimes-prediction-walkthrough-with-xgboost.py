#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# 1. [Introduction](#introduction)
# 2. [Imports](#imports)
# 3. [Describe Dataset](#describe-dataset)
# 4. [Quick Preprocessing & Feature Engineering](#preprocessing)
# 5. [Feature Selection](#feature-selection)
# 6. [Training with XGBoost (Cross Validation)](#training)

# # Introduction <a name="introduction"></a>

# This is my solution ( *late submission* ) to the **San Francisco Crime Classification** competition.
# 
# I found this Dataset to be very interesting for learning to deal with *slightly big* datasets, it contains *Spatial Coordinates*,  a *Datetime* column and *cyclic features*.
# 
# In this Kernel, I want to share my approach to this problem. I will focus on **Feature Engineering** & **Prediction** using **XGBoost**
# 
# ( **I wont' be going through the visualizations** ( you can check [my Github Repo](https://github.com/hamzael1/kaggle-san-francisco-crime-classification) )

# # Imports <a name="imports"></a>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Describe Dataset <a name="describe-dataset"></a>

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Show Random rows:

# In[ ]:


# Show 5 random rows from dataset
train_df.sample(5)


# In[ ]:


test_df.sample(1)


# ## Show useful information (columns, types, number of rows)

# - Few important observations:
#     - We have **878049** Observations of **9** variables
#     - We have a **'Dates'** column which contains the date and time of the occurence of the crime, but it's a String.
#     - We have **spatial coordinates** ( Latitude and Longitude ) of the exact place of the crime.
#     - The Target column is **'Category'**, which is a Categorical Column ( 39 categories )
#     - The **'DayOfWeek'** column is also Categorical ( 7 days )
#     - The **'PdDistrict'** column is also Categorical ( 10 districts  )
#     - The **'Address'** column indicates whether the crime location was an intersection of two roads
#     - The **'Resolution'** column will be droped ( It won't help us with prediction )

# In[ ]:


print('Number of Categories: ', train_df.Category.nunique())
print('Number of PdDistricts: ', train_df.PdDistrict.nunique())
print('Number of DayOfWeeks: ', train_df.DayOfWeek.nunique())
print('_________________________________________________')
# Show some useful Information
train_df.info()


# # Quick Preprocessing & Feature Engineering <a name="preprocessing"></a>

#  ## Drop the Resolution Column:

# In[ ]:


train_df = train_df.drop('Resolution', axis=1)
train_df.sample(1)


# ## Parse the 'Dates' Column:

# ### The 'Dates' column type is String. It will be easier to work with by parsing it to Datetime.

# In[ ]:


train_df.Dates.dtype


# ### Check if there are any missing values or typos:

# In[ ]:


assert train_df.Dates.isnull().any() == False
assert test_df.Dates.isnull().any() == False


# In[ ]:


assert train_df.Dates.str.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d').all() == True
assert test_df.Dates.str.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d').all() == True


# ### Now we proceed to parsing using the function `pandas.to_datetime` :
# ( We will also change the column name to 'Date' singular ) 

# In[ ]:


train_df['Date'] = pd.to_datetime(train_df.Dates)
test_df['Date'] = pd.to_datetime(test_df.Dates)

train_df = train_df.drop('Dates', axis=1)
test_df = test_df.drop('Dates', axis=1)
train_df.sample(1)


# In[ ]:


# Confirm that it was parsed to Datetime
train_df.Date.dtype


# ## Engineer a feature to indicate whether the crime was commited by day or by night :

# In[ ]:


train_df['IsDay'] = 0
train_df.loc[ (train_df.Date.dt.hour > 6) & (train_df.Date.dt.hour < 20), 'IsDay' ] = 1
test_df['IsDay'] = 0
test_df.loc[ (test_df.Date.dt.hour > 6) & (test_df.Date.dt.hour < 20), 'IsDay' ] = 1

train_df.sample(3)


# ## Create 'Month', 'Year' and 'DayOfWeekInt' columns

# ### Encode 'DayOfWeek' to Integer:

# In[ ]:


days_to_int_dic = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7,
}
train_df['DayOfWeek'] = train_df['DayOfWeek'].map(days_to_int_dic)
test_df ['DayOfWeek'] = test_df ['DayOfWeek'].map(days_to_int_dic)

train_df.DayOfWeek.unique()


# ### Create Hour, Month and Year Columns: 

# In[ ]:


train_df['Hour'] = train_df.Date.dt.hour
train_df['Month'] = train_df.Date.dt.month
train_df['Year'] = train_df.Date.dt.year
train_df['Year'] = train_df['Year'] - 2000 # The Algorithm doesn't know the difference. It's just easier to work like that

test_df['Hour'] = test_df.Date.dt.hour
test_df['Month'] = test_df.Date.dt.month
test_df['Year'] = test_df.Date.dt.year
test_df['Year'] = test_df['Year'] - 2000 # The Algorithm doesn't know the difference. It's just easier to work like that

train_df.sample(1)


# ### Deal with the cyclic characteristic of Months and Days of Week:

# In[ ]:


train_df['HourCos'] = np.cos((train_df['Hour']*2*np.pi)/24 )
train_df['DayOfWeekCos'] = np.cos((train_df['DayOfWeek']*2*np.pi)/7 )
train_df['MonthCos'] = np.cos((train_df['Month']*2*np.pi)/12 )

test_df['HourCos'] = np.cos((test_df['Hour']*2*np.pi)/24 )
test_df['DayOfWeekCos'] = np.cos((test_df['DayOfWeek']*2*np.pi)/7 )
test_df['MonthCos'] = np.cos((test_df['Month']*2*np.pi)/12 )

train_df.sample(1)


# ## Dummy Encoding of 'PdDistrict':

# In[ ]:


train_df = pd.get_dummies(train_df, columns=['PdDistrict'])
test_df  = pd.get_dummies(test_df,  columns=['PdDistrict'])
train_df.sample(2)


# ## Label Encoding of 'Category':

# In[ ]:


from sklearn.preprocessing import LabelEncoder

cat_le = LabelEncoder()
train_df['CategoryInt'] = pd.Series(cat_le.fit_transform(train_df.Category))
train_df.sample(5)
#cat_le.classes_


# In[ ]:


train_df['InIntersection'] = 1
train_df.loc[train_df.Address.str.contains('Block'), 'InIntersection'] = 0

test_df['InIntersection'] = 1
test_df.loc[test_df.Address.str.contains('Block'), 'InIntersection'] = 0


# In[ ]:


train_df.sample(10)


# # Feature Selection <a name="feature-selection"></a>

# **Now let's get our dataset ready for training !**

# In[ ]:


train_df.columns


# In[ ]:


feature_cols = ['X', 'Y', 'IsDay', 'DayOfWeek', 'Month', 'Hour', 'Year', 'InIntersection',
                'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE',
                'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',
                'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN']
target_col = 'CategoryInt'

train_x = train_df[feature_cols]
train_y = train_df[target_col]

test_ids = test_df['Id']
test_x = test_df[feature_cols]


# In[ ]:


train_x.sample(1)


# In[ ]:


test_x.sample(1)


# # XGBOOST Training (Cross-Validation): <a name="training"></a>

# In[ ]:


type(train_x), type(train_y)


# ## Import XGBoost and create the DMatrices

# In[ ]:


import xgboost as xgb
train_xgb = xgb.DMatrix(train_x, label=train_y)
test_xgb  = xgb.DMatrix(test_x)


# ## Play with the parameters and do Cross-Validation

# In[ ]:


params = {
    'max_depth': 4,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 39,
}


# In[ ]:


CROSS_VAL = False
if CROSS_VAL:
    print('Doing Cross-validation ...')
    cv = xgb.cv(params, train_xgb, nfold=3, early_stopping_rounds=10, metrics='mlogloss', verbose_eval=True)
    cv


# ## Fit & Make the predictions

# In[ ]:


SUBMIT = not CROSS_VAL
if SUBMIT:
    print('Fitting Model ...')
    m = xgb.train(params, train_xgb, 10)
    res = m.predict(test_xgb)
    cols = ['Id'] + cat_le.classes_
    submission = pd.DataFrame(res, columns=cat_le.classes_)
    submission.insert(0, 'Id', test_ids)
    submission.to_csv('submission.csv', index=False)
    print('Done Outputing !')
    print(submission.sample(3))
else:
    print('NOT SUBMITING')

