#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Initial importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.feature_selection import f_regression, SelectKBest, f_classif
from sklearn.linear_model import Ridge, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier, XGBRegressor


# In[ ]:


# Enabling the entire df to be viewed when it goes beyond the normal 80 cols/rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


# In[ ]:


# Suppressing warning messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[ ]:


# Reading in the csv for the training features
df = pd.read_csv('../input/train_features.csv')
df.head().T


# In[ ]:


# Reading in the csv for the training labels
df_labels = pd.read_csv('../input/train_labels.csv')
df_labels.head()


# In[ ]:


# Reading in the csv for the test data
df_test = pd.read_csv('../input/test_features.csv')
df_test.head().T


# In[ ]:


# verifying shape of these dataframes
df.shape, df_labels.shape, df_test.shape


# In[ ]:


# Verifying the value counts of the target value, status_group
df_labels.status_group.value_counts(normalize=True)


# In[ ]:


# Verifying the data types of each column in our dataframe
df.dtypes


# In[ ]:


# Just selecting the numeric features from the features dataframe
df_num = df.select_dtypes(include=['number'])
print(df_num.shape)
df_num.head()


# In[ ]:


# Just selecting the numeric features from the test dataframe
df_num_test = df_test.select_dtypes(include=['number'])
print(df_num_test.shape)
df_num_test.head()


# In[ ]:


# Initializing our X and y variables for regression 
X = df_num
y = df_labels.status_group


# In[ ]:


# Using Logistic Regression model to make baseline prediction
model = LogisticRegression(solver='lbfgs', multi_class='auto')
model.fit(X, y)
y_pred = model.predict(X)
print(classification_report(y, y_pred))
print('accuracy', accuracy_score(y, y_pred))


# In[ ]:


# Fitting and scoring our Logistic Regression model
log_reg = LogisticRegression(solver='lbfgs').fit(X, y)
log_reg.score(X, y)


# In[ ]:


# Predicting outcomes on our test data
pred_test = log_reg.predict(df_num_test)


# In[ ]:


# creating a dataframe for the predicted outcomes
pred = pd.DataFrame(pred_test)
pred.head()


# In[ ]:


# Adding the 'id' column to the prediction dataframe
pred = pd.concat([df_num_test.id, pred], axis=1)
pred.head()


# In[ ]:


# Remaning the status_group column to the correct name
pred.columns.values[1]= 'status_group'
pred.head()


# In[ ]:


# Verifying the dataframe has the correct shape for submission
pred.shape


# In[ ]:


# ***PRINT TO CSV***
pred.to_csv('pred_test.csv', sep=',', encoding='utf-8', index=False)
# Or use this to print to csv
#pd.DataFrame(pred).to_csv("submission_pd.csv", index = False)


# First baseline test completed and submitted

# In[ ]:


# Find columns with unique categorical observations
# I did this to try to see how many categorical features 
# could/should be used for One Hot encoding
df_unique = df.select_dtypes(exclude=['number'])
df_unique.nunique()


# In[ ]:


# Check data types of this dataframe to ensure non-numerics
df_unique.dtypes


# In[ ]:


# Counting total number of unique cat observations
df_unique.nunique().sum()


# In[ ]:


# Checking to describe of unique dataframe
df_unique.describe()


# In[ ]:


# Dropping columns I dont find useful(too many unique values/odd observations)
df_unique_drop = df_unique.drop(columns=['date_recorded', 'funder', 'installer', 'wpt_name', 
                            'lga', 'ward', 'recorded_by', 'scheme_name', 
                            'subvillage', 'public_meeting', 'scheme_management', 'permit',
                            'funder', 'installer', 'scheme_name'])
  
# Head check for dropped unique
df_unique_drop.head()


# In[ ]:


# Checking unique dataframe after drop
df_unique_drop.nunique()


# In[ ]:


# Recount of unique observations after drop
df_unique_drop.nunique().sum()


# In[ ]:


# The unique dataframe was not as easy to work with
# so back to working with the original data
# verifying the numeric columns
df.describe()


# In[ ]:


# Dropping columns I dont find useful(too many unique values/odd observations)
df_drop = df.drop(columns=['id', 'longitude', 'latitude', 'num_private', 
                            'date_recorded', 'funder', 'installer', 'wpt_name', 
                            'lga', 'ward', 'recorded_by', 'scheme_name', 
                            'subvillage', 'public_meeting', 'scheme_management', 'permit',
                            'funder', 'installer', 'scheme_name'])
  
df_drop.head()


# In[ ]:


# Verifying nulls
df_drop.isnull().sum()


# In[ ]:


# Another null check
def no_nulls(df):
    return not any(df.isnull().sum())

no_nulls(df_drop)


# In[ ]:


# Verifying head of dataframe after dropping columns
df_drop.head()


# In[ ]:


# Verifying numeric features
df_drop.describe()


# In[ ]:


# Verifying which columns are numeric and which are strings/objects
df_drop.dtypes


# In[ ]:


# 1hot encode non-numeric columns
df_one_hot = pd.get_dummies(df_drop)
df_one_hot.head()


# In[ ]:


# Dropping same columns on the test dataframe
df_drop_test = df_test.drop(columns=['id', 'longitude', 'latitude', 'num_private', 
                            'date_recorded', 'funder', 'installer', 'wpt_name', 
                            'lga', 'ward', 'recorded_by', 'scheme_name', 
                            'subvillage', 'public_meeting', 'scheme_management', 'permit',
                            'funder', 'installer', 'scheme_name'])
                           
df_drop_test.head()


# In[ ]:


# Null verification
df_drop_test.isnull().sum()


# In[ ]:


# More null verification
def no_nulls(df):
    return not any(df.isnull().sum())

no_nulls(df_drop_test)


# In[ ]:


# checking numeric features
df_drop_test.describe()


# In[ ]:


# Verifying shape to ensure we have correct feature and observation counts
df_drop_test.shape


# In[ ]:


# 1hot encode non-numeric columns for test data
df_one_hot_test = pd.get_dummies(df_drop_test)
df_one_hot_test.head()


# In[ ]:


# Swapping '/' characters for '_' from data as it was causing errors, in test dataframe
df_one_hot_test.columns = [x.strip().replace('/', '_') for x in df_one_hot_test.columns]

df_one_hot_test.head().T


# In[ ]:


# Checking head after one hot encoding
df_one_hot.head().T


# In[ ]:


# Checking numeric cols after one hot
df_one_hot.describe()


# In[ ]:


# Fixing '/' to '_', this time on train dataframe
df_one_hot.columns = [x.strip().replace('/', '_') for x in df_one_hot.columns]
df_one_hot.head().T


# In[ ]:


# Had one column in train that was not in test so I had to drop it from train dataframe so 
# the number of features in test and train match
df_one_hot = df_one_hot.drop(columns=['extraction_type_other - mkulima_shinyanga'])
df_one_hot.head()


# In[ ]:


# Setting X and y variables
X = df_one_hot
y = df_labels.status_group


# In[ ]:


# Fit Logistic Regression model on X and y
model = LogisticRegression(solver='lbfgs')
model.fit(X, y)
y_pred = model.predict(X)
print(classification_report(y, y_pred))
print('accuracy', accuracy_score(y, y_pred))


# In[ ]:


# Fit and score Logistic Regression model
log_reg = LogisticRegression().fit(X, y)
log_reg.score(X, y)


# In[ ]:


# Scoring prediction with test data 
pred_test = log_reg.predict(df_one_hot_test)


# In[ ]:


# Sending prediction to a dataframe and verifying correct shape
pred = pd.DataFrame(pred_test)
pred.shape


# In[ ]:


# Adding 'id' column to dataframe
pred = pd.concat([df_test.id, pred], axis=1)
pred.head()


# In[ ]:


# Renaming status_group column
pred.columns.values[1]= 'status_group'
pred.head()


# In[ ]:


# Verifying data looks to be in acceptable ranges
pred.describe()


# In[ ]:


# ***PRINT TO CSV***
pred.to_csv('pred_test_again.csv', sep=',', encoding='utf-8', index=False)


# Submission number 2 was a success!

# In[ ]:


# Reread in data to have fresh dataframes
df_train = pd.read_csv('../input/train_features.csv')
df_labels = pd.read_csv('../input/train_labels.csv')
df_test = pd.read_csv('../input/test_features.csv')


# In[ ]:


# Working towards Random Forest and XGBoost here
# Verify head of training data
df_train.head()


# In[ ]:


# verify head of test data
df_test.head()


# In[ ]:


# verify head of feature labels
df_labels.head()


# In[ ]:


# Checking shape of each dataframe
df_train.shape, df_labels.shape, df_test.shape


# In[ ]:


# merge train and labels datasets
df_merged = pd.merge(df_train, df_labels, on='id')
df_merged.head().T


# In[ ]:


# set terget 
target=df_merged.pop('status_group')


# In[ ]:


# Verify head of target
target.head()


# In[ ]:


# Verify head of merged sataframe
df_merged.head().T


# In[ ]:


# Check shapes of each dataframe
df_train.shape, df_labels.shape, df_test.shape, df_merged.shape, target.shape


# In[ ]:


# Verifying merged dataframe feature info
df_merged.info()


# In[ ]:


# Adding 'train' column to training and test dataframes and adding value of 
# 1 if from training dataframe and 0 if from test dataframe
df_merged['train']=1
df_test['train']=0
df_merged.info()


# In[ ]:


# Verifying test dataframe feature info
df_test.info()


# In[ ]:


# Dataframe shape verification
df_merged.shape, df_test.shape


# In[ ]:


# concatenating the train and test dateframes for munging
combined = pd.concat([df_merged, df_test])
combined.head()


# In[ ]:


# verifying dataframe feature info
combined.info()


# In[ ]:


# Dropping features I do not feel fit the prediction model we will be using
combined.drop('construction_year',axis=1,inplace=True)
combined.drop('date_recorded',axis=1,inplace=True)
combined.drop('wpt_name',axis=1,inplace=True)
combined.drop('num_private',axis=1,inplace=True)
combined.drop('subvillage',axis=1,inplace=True)
combined.drop('region_code',axis=1,inplace=True)
combined.drop('ward',axis=1,inplace=True)
combined.drop('public_meeting',axis=1,inplace=True)
combined.drop('recorded_by',axis=1,inplace=True)
combined.drop('scheme_name',axis=1,inplace=True)
combined.drop('permit',axis=1,inplace=True)
combined.drop('extraction_type_group',axis=1,inplace=True)
combined.drop('extraction_type_class',axis=1,inplace=True)
combined.drop('management_group',axis=1,inplace=True)
combined.drop('payment',axis=1,inplace=True)
combined.drop('quality_group',axis=1,inplace=True)
combined.drop('quantity_group',axis=1,inplace=True)
combined.drop('source_type',axis=1,inplace=True)
combined.drop('source_class',axis=1,inplace=True)
combined.drop('waterpoint_type_group',axis=1,inplace=True)
combined.drop('installer',axis=1,inplace=True)
combined.info()


# In[ ]:


# Factorizing the remaining categorical features
# factorize swaps unique categorical observations into a unique numeric
combined['funder'] = pd.factorize(combined['funder'])[0]
combined['scheme_management'] = pd.factorize(combined['scheme_management'])[0]
combined['extraction_type'] = pd.factorize(combined['extraction_type'])[0]
combined['management'] = pd.factorize(combined['management'])[0]
combined['payment_type'] = pd.factorize(combined['payment_type'])[0]
combined['water_quality'] = pd.factorize(combined['water_quality'])[0]
combined['quantity'] = pd.factorize(combined['quantity'])[0]
combined['source'] = pd.factorize(combined['source'])[0]
combined['waterpoint_type'] = pd.factorize(combined['waterpoint_type'])[0]
combined['basin'] = pd.factorize(combined['basin'])[0]
combined['region'] = pd.factorize(combined['region'])[0]
combined['lga'] = pd.factorize(combined['lga'])[0]
combined['district_code'] = pd.factorize(combined['district_code'])[0]
combined.district_code.head(5)


# In[ ]:


# Splitting the combined dataframe back into test and train using the 'train' feature we added above
train_df = combined[combined["train"] == 1]
test_df = combined[combined["train"] == 0]


# In[ ]:


# Verifying the test dataframe 
test_df.head()


# In[ ]:


# checking to ensure the train dataframe has only training data 
train_df.train.value_counts()


# In[ ]:


# checking to ensure the test dataframe has only test data 
test_df.train.value_counts()


# In[ ]:


# verifying shapes are correct
train_df.shape, test_df.shape


# In[ ]:


# Dropping the 'train' column we added above
train_df.drop(["train"], axis=1, inplace=True)
train_df.head()


# In[ ]:


# verifying shape is correct
train_df.shape


# In[ ]:


# Dropping the 'train' column we added above
test_df.drop(["train"], axis=1, inplace=True)
test_df.head()


# In[ ]:


# verifying shape is correct
test_df.shape


# In[ ]:


#defining X as the entire train dataframe
X = train_df
# defining the target as 'y'
y = target


# In[ ]:


# Setting RendomForestClassifier estimators
model_rfc = RandomForestClassifier(n_estimators=1000)


# In[ ]:


# Setting cross validation score inputs
cross_val_score(model_rfc, X, y, cv=3)


# Large improvement with Random Forest model on prediction score over Logistic Regression

# In[ ]:


# Fitting the Random Forest model
model_rfc.fit(X,y)
X.info() # Just printing the features to aid in matching features to score below
importances = model_rfc.feature_importances_
importances # list of feature scores, can match using above print out


# In[ ]:


# Printing and sorting the most important features from Random Forest model
importances = model_rfc.feature_importances_
importances
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

for f in range(X.shape[1]):
    print(X.columns[indices[f]],end=', ')


# In[ ]:


# Random Forest model fitting
model_rfc.fit(X,y)


# In[ ]:


# Model set up for XGBoost classification model
model = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 
                      num_class = 4, maximize = False, eval_metric = 'merror', eta = .2,
                      max_depth = 14, colsample_bytree = .4)


# In[ ]:


# XGBoost fit and scoring
print(cross_val_score(model, X, y, cv=3))
model.fit(X,y)
importances = model.feature_importances_
importances
indices = np.argsort(importances)[::-1]


# In[ ]:


# Print the feature ranking without score
print("Feature ranking:")

for f in range(X.shape[1]):
    print(X.columns[indices[f]],end=', ')


# In[ ]:


# Printing feature ranking with rank and score
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[ ]:


# Setting X_test initial value
X_test=test_df


# In[ ]:


# Verify head of X_test looks good
X_test.head()


# In[ ]:


# Run test dataframe through model to receive score
a=X_test['id']
#X_test.drop(['id'],axis=1, inplace=True)
y_pred = model.predict(X_test)


# In[ ]:


# Place prediction into dataframe
y_pred=pd.DataFrame(y_pred)
y_pred.head()


# In[ ]:


# adding 'id feature to dataframe'
y_pred['id']=a
y_pred.head()


# In[ ]:


# Renaming columns in dataframe
y_pred.columns=['status_group','id']
y_pred.head()


# In[ ]:


# Swapping columns in dataframe to fit the Kaggle submission requirements
y_pred=y_pred[['id','status_group']]
y_pred.head()


# In[ ]:


# Verify shape of prediciton
y_pred.shape


# In[ ]:


# Output dataframe to CSV for Kaggle submission
pd.DataFrame(y_pred).to_csv("fifth_try.csv", index=False)


# The XGBoost model was the best score, barely beating out the Random Forest model.

# 

# 

# 
