#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Lots of people on Kaggle have worked with this heart disease dataset it seems. Here are the links.
# 
# https://www.kaggle.com/ronitf/heart-disease-uci
# http://archive.ics.uci.edu/ml/datasets/Heart+Disease

# In[ ]:


dataset = pd.read_csv("../input/cleveland-heart-disease-data-csv/Cleveland Heart Disease Data.csv")
dataset.head()


# So far so good. Here's the documentation on all the variables. Might be useful in trying to understand what's going on, over here.
# 
# Variables:
# * age: Age in years
# * sex: sex (1 = male; 0 = female)
# * cp: chest pain type
#     - Value 1: typical angina
#     - Value 2: atypical angina
#     - Value 3: non-anginal pain
#     - Value 4: asymptomatic
# * trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# * chol: serum cholestoral in mg/dl
# * fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# * restecg: resting electrocardiographic results
#     - Value 0: normal
#     - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#     - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# * thalach: maximum heart rate achieved
# * exang: exercise induced angina (1 = yes; 0 = no)
# * oldpeak = ST depression induced by exercise relative to rest
# * slope: the slope of the peak exercise ST segment
#     - Value 1: upsloping
#     - Value 2: flat
#     - Value 3: downsloping
# * ca: number of major vessels (0-3) colored by flourosopy
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# * num: diagnosis of heart disease (angiographic disease status) (the predicted attribute)
#     - Value 0: < 50% diameter narrowing
#     - Value 1: > 50% diameter narrowing

# In[ ]:


dataset.describe()


# The documentation on the website seems to make it pretty apparent which variables are numerical and which ones are categorical. I assume that because the file was already processed, there are no missing values and generally pretty good. Nonetheless, a cursory glance at the uniques will do well. 
# 
# So, first, I'm going to remove "num" from the dataset, since that's the y variable.

# In[ ]:


y = dataset['num']
X = dataset.drop('num', axis = 1)
X.shape


# In[ ]:


y.shape


# Next, it's time to split the dataset using the test train split. And then, turn the train data into valid data as well.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Checking if it worked
print(X_train_full.shape)
print(X_test.shape)
print(y_train_full.shape)
print(y_test.shape)


# In[ ]:


# Creating some valid sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# In[ ]:


X_train.dtypes


# It looks like pandas looks at most of the the variables as int64 or float64. So I'm going to have to manually tell it which features are categorical and which ones are numerical. As I understand it, my differentiation should be like thus:
# 
# Categorical:
# * Sex
# * CP
# * fbs
# * restecg
# * exang
# * slope
# * ca
# * thal
# 
# Numerical:
# * Age
# * trestbps
# * chol
# * thalach
# * oldpeak

# In[ ]:


numerical_columns = ['Age', 'trestbps','chol', 'thalach','oldpeak'] 
categorical_columns = ['Sex','CP','fbs','restecg','exang','slope','ca','thal']

cols_with_missing_train = [col for col in X_train.columns if X_train[col].isnull().any()]
print(cols_with_missing_train)
cols_with_missing_valid = [col for col in X_valid.columns if X_valid[col].isnull().any()]
print(cols_with_missing_valid)


# It looks like there are no missing values. But that's not exactly true.

# In[ ]:


for column in categorical_columns:
    print(X_train[column].value_counts())


# I'm going to replace the ?s with NaNs. And then use simple imputing to fill missing values.

# In[ ]:


X_train.ca.replace("?", np.NaN).value_counts()


# In[ ]:


X_train.thal.replace("?", np.NaN).value_counts()


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#pipelines!
#creating a transformers for categories
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])


# Training on the Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
        ])
    my_pipeline.fit(X_train, y_train)
    preds_valid = my_pipeline.predict(X_valid)
    scores = mean_absolute_error(y_valid, preds_valid)
    return scores

MAE = {}

for num_leaves in range(100,151):
    MAE[num_leaves] = get_score(num_leaves)
    
MAE


# It appears that the ideal number of leaves is 131, based on MAE.

# In[ ]:


my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(131, random_state=0))
        ])
my_pipeline.fit(X_train, y_train)

#Applying that to the test model. I actually have a y-value for test too.
preds_test = my_pipeline.predict(X_test)
score_test = mean_absolute_error(y_test, preds_test)
print("MAE:", score_test)

# Save test predictions to file
output = pd.DataFrame({'num': preds_test})
output.to_csv('submission.csv', index=False)

