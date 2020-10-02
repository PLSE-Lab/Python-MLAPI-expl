#!/usr/bin/env python
# coding: utf-8

# ## Fraud Detection Classification Problem using Quantative and Qualitative Features

# In[ ]:


# Set Directory
import os
os.getcwd()


# In[ ]:


import numpy as np
import pandas as pd

# 1. Preprocessing Libraries
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder    #Dummification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline    #NEW!
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold    #Hyperparameter tuning, StratifiedKFold
# another way to cross-validate
from sklearn.compose import ColumnTransformer


# 2. Algorithm Import
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier


# 3. Evaluation Library
from sklearn.metrics import confusion_matrix

# 4. Viz Lib
import matplotlib.pyplot as plt 
import seaborn as sns

# 5. Misc Lib
# !pip install imblearn
from imblearn.over_sampling import SMOTE    #Data/Class imbalance
import random
random.seed(123)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Train Data Reading
m_train_data = pd.read_csv(filepath_or_buffer = 'train_merchant_data-1561627820784.csv', sep = ",", header = 0)
o_train_data = pd.read_csv(filepath_or_buffer = 'train_order_data-1561627847149.csv', sep = ",", header = 0)
y_train_data = pd.read_csv(filepath_or_buffer = 'train-1561627878332.csv', sep = ",", header = 0)
print(m_train_data.shape)
print(o_train_data.shape)
print(y_train_data.shape)


#Test Data Reading   
m_test_data = pd.read_csv(filepath_or_buffer = 'test_merchant_data-1561627903902.csv', sep = ",", header = 0)
o_test_data = pd.read_csv(filepath_or_buffer = 'test_order_data-1561627931868.csv', sep = ",", header = 0)
y_test_data = pd.read_csv(filepath_or_buffer = 'test-1561627952093.csv', sep = ",", header = 0)
print(m_test_data.shape)
print(o_test_data.shape)
print(y_test_data.shape)   # 'y' Test Data is unlabelled


# **Understanding the Dataset**

# In[ ]:


print(m_train_data.columns, "\n")
print(o_train_data.columns, "\n")   #Join on 'Merchant_ID'

print(m_train_data.dtypes, "\n")   #Mixture of Integers and Objects
print(o_train_data.dtypes)


# In[ ]:


m_train_data.head()


# In[ ]:


m_train_data.describe()  #No NA's in numerical data


# In[ ]:


m_train_data.describe(include = 'object')


# In[ ]:


o_train_data.describe()   #No NA's in o_train_data (Quant)


# In[ ]:


o_train_data.describe(include = 'object')


# **Merging Dataframes on Merchant_ID**

# In[ ]:


train_data = pd.merge(left = m_train_data, right = o_train_data, how = 'outer', on = 'Merchant_ID')
train_data = pd.merge(left = train_data, right = y_train_data, how = 'outer', on = 'Merchant_ID')

test_data = pd.merge(left = m_test_data, right = o_test_data, how = 'outer', on = 'Merchant_ID')
test_data = pd.merge(left = test_data, right = y_test_data, how = 'outer', on = 'Merchant_ID')


# In[ ]:


print(train_data.shape, '\n')
print(train_data.dtypes)


# In[ ]:


print(test_data.shape, '\n')
print(test_data.dtypes)


# In[ ]:


train_data.describe(include = 'int64')
# from pandas.plotting import table # EDIT: see deprecation warnings below

# ax = plt.subplot(111, frame_on=False) # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis

# table(ax, train_data.describe())  # where df is your data frame

# plt.savefig('train_data_describe.png')


# ### Pre-Processing Data and Exploratory Data Analysis
# 
# Pre-processing steps of the Train Data will be Applied to Test Data respectively. 

# **Notes on Quantative Features**
# 
# 'Merchant_ID', 'Customer_ID' will be converted to dtype 'object'.
# 
# 'Ecommerce_Provider_ID' will be dropped due to No Information Gain.

# In[ ]:


train_data.drop(labels = 'Ecommerce_Provider_ID', axis = 1, inplace = True)
test_data.drop(labels = 'Ecommerce_Provider_ID', axis = 1, inplace = True)


# In[ ]:


train_data.shape


# In[ ]:


train_data


# In[ ]:


train_data[['Merchant_ID', 'Customer_ID']] = train_data[['Merchant_ID', 'Customer_ID']].astype('object') 
test_data[['Merchant_ID', 'Customer_ID']] = test_data[['Merchant_ID', 'Customer_ID']].astype('object')
print(train_data.dtypes, '\n')
print(test_data.dtypes)


# In[ ]:


train_data.describe(include = 'object')


# Percentage Distribution of Order Sources

# In[ ]:


train_data['Order_Source'].value_counts(normalize = True)


# In[ ]:


sns.countplot(x='Order_Source', data=train_data)
plt.show()
train_data.Order_Source.value_counts()

X_train.describe()
# Distribution of 'Age' is fairly symmetrical. Mean = 33.12, Median = 32

# In[ ]:


# Check distribution of age
# Density Curve + Histogram, shows distribution of Continuous Feature
get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(train_data["Age"]) #NT: y is percentage count 


# In[ ]:


train_data['Order_Payment_Method'].value_counts(normalize = True)


# In[ ]:


sns.countplot(x='Order_Payment_Method', data=train_data)
plt.show()
train_data.Order_Payment_Method.value_counts()


# ### Feature Engineering
#To get column index position
train_data.columns.get_loc('IP_Address')
# In[ ]:


f# !pip install maxminddb-geolite2

from geolite2 import geolite2
geo = geolite2.reader()

def get_country(ip):
    try :
        x = geo.get(ip)
    except ValueError:   #Faulty or improper IP value
        return np.nan
    try:
        return x['country']['names']['en'] if x is not None else np.nan
    except KeyError: 
        return np.nan
    
def get_city(ip):
    try:
        x = geo.get(ip)
    except ValueError:   #if Value passed is faulty
        return np.nan
    
    try:
        return x['city']['names']['en'] if x is not None  else np.nan
    except KeyError:  #if Dictionary indexing is faulty 
        return np.nan
    
def get_subdivisions(ip):
    try:
        x = geo.get(ip)
    except ValueError:   #if Value passed is faulty
        return np.nan
    
    try:
        return x['subdivisions'][0]['names']['en'] if x is not None  else np.nan
    except KeyError:  #if Dictionary indexing is faulty 
        return np.nan

def get_continent(ip):
    try:
        x = geo.get(ip)
    except ValueError:   #if Value pafssed is faulty
        return np.nan
    
    try:
        return x['continent']['names']['en'] if x is not None  else np.nan
    except KeyError:  #if Dictionary indexing is faulty 
        return np.nan


# In[ ]:


import time

s_time = time.time()
#apply(fn) applies fn. on all pd.series elements
train_data['country'] = train_data.loc[:,'IP_Address'].apply(get_country)  #Time:  41.6 s
test_data['country'] = test_data.loc[:,'IP_Address'].apply(get_country)   #Time: 7.71 s
train_data['Continent'] = train_data.loc[:,'IP_Address'].apply(get_continent)
test_data['Continent'] = test_data.loc[:,'IP_Address'].apply(get_continent)
train_data['Sub_Divisions'] = train_data.loc[:,'IP_Address'].apply(get_subdivisions)
test_data['Sub_Divisions'] = test_data.loc[:,'IP_Address'].apply(get_subdivisions)
train_data['City'] = train_data.loc[:,'IP_Address'].apply(get_city)
test_data['City'] = test_data.loc[:,'IP_Address'].apply(get_city)
# geolite2.close()
print('Time:',time.time()-s_time)


# In[ ]:


train_data.isna().sum()


# In[ ]:


print(test_data.shape)
test_data.isna().sum()


# In[ ]:


train_data.to_csv('train_data.csv', index = False, na_rep='NaN')
test_data.to_csv('test_data.csv', index = False, na_rep = 'NaN')


# # ---------------------------------------------------------------

# In[ ]:


#Time Stamping
train_data['Date_of_Order'] = pd.to_datetime(train_data['Date_of_Order'], infer_datetime_format=True)
test_data['Date_of_Order'] = pd.to_datetime(test_data['Date_of_Order'], infer_datetime_format=True)
train_data['Merchant_Registration_Date'] = pd.to_datetime(train_data['Merchant_Registration_Date'], infer_datetime_format=True)
test_data['Merchant_Registration_Date'] = pd.to_datetime(test_data['Merchant_Registration_Date'], infer_datetime_format=True)

Extracting DT info
train_data['Quarter'] = train_data['Date_of_Order'].dt.quarter
train_data['Month'] = train_data['Date_of_Order'].dt.month
train_data['Day'] = train_data['Date_of_Order'].dt.weekday_name
train_data['Time_Hour'] = train_data['Date_of_Order'].dt.hour
test_data['Quarter'] = test_data['Date_of_Order'].dt.quarter
test_data['Month'] = test_data['Date_of_Order'].dt.month
test_data['Day'] = test_data['Date_of_Order'].dt.weekday_name
test_data['Time_Hour'] = test_data['Date_of_Order'].dt.hour


# In[ ]:


import datetime as dt
from datetime import datetime


def get_month(x):
    return dt.datetime.strftime(x, format = '%b')

# train_data['Month'] = train_data.loc[:,'Date_of_Order'].apply(get_month)
test_data['Month'] = test_data.loc[:,'Date_of_Order'].apply(get_month)


# In[ ]:


#Drop Redundant Features, Convert approp dtypes
# train_data.drop(labels=['Merchant_ID', 'IP_Address', 'Registered_Device_ID', 'Customer_ID', 'Order_ID'], axis = 1, inplace=True)
# test_data.drop(labels=['Merchant_ID', 'IP_Address', 'Registered_Device_ID', 'Customer_ID', 'Order_ID'], axis = 1, inplace=True)


# **Notes on Qualitative Features**
# 
# 'Order_Source': Has 3 Levels - SEO, Ads, Direct
# 
# 'Order_Payment_Method': Has 5 Levels - Credit Card, Internet Banking, Debit Card, E-wallet, Cash On Delivery
# 
# 'Gender', 'Order_Source', 'Order_Payment_Method', 'country', 'Continent', 'City', 'Sub_Divisions', 'Quarter', 'Month', 'Day', 'Time_Hour': Can be converted to 'category' dtype

# In[ ]:


train_data[['Gender', 'Order_Source', 'Order_Payment_Method', 'country', 'Continent', 'City', 'Sub_Divisions', 'Quarter', 'Month', 'Day', 'Time_Hour']] = train_data[['Gender', 'Order_Source', 'Order_Payment_Method', 'country', 'Continent', 'City', 'Sub_Divisions', 'Quarter', 'Month', 'Day', 'Time_Hour']].astype('category')
test_data[['Gender', 'Order_Source', 'Order_Payment_Method', 'country', 'Continent', 'City', 'Sub_Divisions', 'Quarter', 'Month', 'Day', 'Time_Hour']] = test_data[['Gender', 'Order_Source', 'Order_Payment_Method', 'country', 'Continent', 'City', 'Sub_Divisions', 'Quarter', 'Month', 'Day', 'Time_Hour']].astype('category')


# In[ ]:


train_data['Account_Period'] = (train_data['Date_of_Order'].dt.date - train_data['Merchant_Registration_Date'].dt.date).dt.days
test_data['Account_Period'] = (test_data['Date_of_Order'].dt.date - test_data['Merchant_Registration_Date'].dt.date).dt.days


# In[ ]:


train_data = pd.read_csv('train_data.csv', header=0)
test_data = pd.read_csv('test_data.csv', header=0)


# In[ ]:


train_data.dtypes


# In[ ]:


unique_vals = train_data['Fraudster'].unique()
#list of sliced dataframes
targets = [train_data.loc[train_data['Fraudster']==val] for val in unique_vals]

dist_plot = sns.distplot(targets[0]['Account_Period'])
sns.distplot(targets[1]['Account_Period'])
dist_plot.set(xlabel='Account_Period', ylabel='freq')  #


# In[ ]:


print("Non-Fraud Records with +ve Account_Period:", (targets[0].loc[targets[0]['Account_Period']>=0].shape[0])/targets[0].shape[0])
print("Fraud Records with -ve Account_Period:", (targets[1].loc[targets[1]['Account_Period']<0].shape[0])/targets[1].shape[0])


# In[ ]:


def acct_period_status(x):
    if x  >= 0:
        return '+ve'
    else:
        return '-ve'


# In[ ]:


train_data['Account_Period_Status'] = train_data.loc[:,'Account_Period'].apply(acct_period_status)
test_data['Account_Period_Status'] = test_data.loc[:,'Account_Period'].apply(acct_period_status)


# In[ ]:


dist_plot = sns.distplot(targets[0]['Age'])
sns.distplot(targets[1]['Age'])
dist_plot.set(xlabel='Age', ylabel='freq')  #'Age' Uniform distb for both Fraudster =  {0,1}


# In[ ]:


train_data[['Fraudster', 'Age']].groupby('Fraudster').median()  #can be dropped


# In[ ]:


dist_plot = sns.distplot(targets[0]['Order_Value_USD'])
sns.distplot(targets[1]['Order_Value_USD'])
dist_plot.set(xlabel='Order_Value_USD', ylabel='freq')


# In[ ]:


train_data[['Fraudster', 'Order_Value_USD']].groupby('Fraudster').median() #Feature is dropped


# #### Remove Redundant Features

# In[ ]:


train_data.drop(labels=['Merchant_Registration_Date', 'Date_of_Order', 'Account_Period'], axis=1, inplace=True)
test_data.drop(labels=['Merchant_Registration_Date', 'Date_of_Order', 'Account_Period'], axis=1, inplace=True)


# #### Note: Numerical Features uniformaly distrib in both Target Classes can be removed !

# In[ ]:


train_data.drop(labels=['Age', 'Order_Value_USD'], axis=1, inplace=True)
test_data.drop(labels=['Age', 'Order_Value_USD'], axis=1, inplace=True)

train_data[['Fraudster', 'Account_Period_Status']] = train_data[['Fraudster', 'Account_Period_Status']].astype('category') 
test_data[['Account_Period_Status']] = test_data[['Account_Period_Status']].astype('category') 


# In[ ]:


train_data.isna().sum()


# In[ ]:


train_data.drop(labels=['City', 'Sub_Divisions'], axis = 1, inplace=True)


# In[ ]:


train_data.drop(['City', 'Sub_Divisions'], inplace=True, axis=1) #Dropped due to 50% of NaNs
train_data


# In[ ]:


si = SimpleImputer(strategy='constant', missing_values=np.nan, fill_value='other')
train_data[['country', 'Continent']] = si.fit_transform(X=train_data[['country', 'Continent']])
train_data[['country', 'Continent']] = train_data[['country', 'Continent']].astype('category')
train_data


# **-------------------------------------------------------------------**

# In[ ]:


train_data.iloc[:,:] = train_data.astype('category')
train_data.dtypes


# In[ ]:


X_data = train_data.copy()
X_data.drop(labels='Fraudster', axis=1, inplace=True)
y_data = train_data[['Fraudster']]


# In[ ]:


print(X_data.dtypes, '\n')
print(y_data.dtypes)


# #### Visualize No. of Target Labels:

# In[ ]:


sns.countplot(x='Fraudster', data=y_data)
plt.show()
y_data.Fraudster.value_counts(normalize=True)  #Class Imbalance, stratified sampling needed


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.33, random_state=123, stratify=y_data)
print(y_train['Fraudster'].value_counts())
print(y_val['Fraudster'].value_counts())


# **Model Building 1**: Logistic Regression

# In[ ]:


X_train.dtypes


# In[ ]:


# num_attr = list(X_train.select_dtypes('int64').columns)

cat_attr = list(X_train.select_dtypes('category').columns)
# cat_attr = list(set(cat_attr).difference(set(['country','Continent', 'City', 'Sub_Divisions'])))


# In[ ]:


# Numerical Pipeline: Pipeline has 2 operation - SimpleImputer() + StandardScaler()
# numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])    

# Categorical Pipeline: Pipeline has 2 operation - SimpleImputer() + OneHotEncoder()
categorical_transformer = Pipeline(steps =[('onehot', OneHotEncoder(handle_unknown='ignore'))])
#SimpleImputer strategy ='constant', then fill_values = 'Constant value string written here'

preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, cat_attr)])


# In[ ]:


clf_logreg = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])
# clf_logreg = Pipeline(steps=[('classifier', LogisticRegression())])


# In[ ]:


y_train.shape


# In[ ]:


X_train.isna().sum()


# In[ ]:


clf_logreg.fit(X = X_train, y = y_train)


# In[ ]:


test_pred = clf_logreg.predict(X_val)

print(clf_logreg.score(X_val, y_val))


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Conf Matrix : \n", confusion_matrix(y_val, test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_val, test_pred))
print("\nTrain DATA Precision", precision_score(y_val, test_pred))
print("\nTrain DATA Recall", recall_score(y_val, test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_val, test_pred,pos_label=1))
print("\nTrain data f1-score for class '0'",f1_score(y_val, test_pred,pos_label=0))


# **Model 2: Decision Tree Model**

# In[ ]:


clf_dt = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier())])

dt_param_grid = {'classifier__criterion': ['entropy', 'gini'], 'classifier__max_depth': [8,10,12], 
                 "classifier__min_samples_split": [2, 10, 20],"classifier__min_samples_leaf": [1, 5, 10]}


dt_grid = GridSearchCV(clf_dt, param_grid=dt_param_grid, cv=5)


# In[ ]:


dt_grid.fit(X_train,y_train)


# In[ ]:


dt_grid.best_params_


# In[ ]:


test_pred = dt_grid.predict(X_val)

print(dt_grid.score(X_val, y_val))


# In[ ]:


print("Conf Matrix : \n", confusion_matrix(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Precision", precision_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Recall", recall_score(y_true=y_val, y_pred = test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=1))
print("\nTraom data f1-score for class '0'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=0))


# #### Model - 3 Random Forest

# In[ ]:


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=143)

# param_grid = {"classifier__n_estimators" : [150, 250, 300],
#               "classifier__max_depth" : [5,8,10],
#               "classifier__max_features" : [3, 5, 7],
#               "classifier__min_samples_leaf" : [4, 6, 8, 10]}

param_grid = {"classifier__n_estimators" : [150, 300],
              "classifier__max_depth" : [5,10],
              "classifier__max_features" : [3, 7],
              "classifier__min_samples_leaf" : [4, 10]}

rf_grid = GridSearchCV(clf, param_grid=dt_param_grid, cv=kfold)


# In[ ]:


rf_grid.fit(X_train,y_train)


# In[ ]:


rf_grid.best_params_


# In[ ]:


test_pred = dt_grid.predict(X_val)

print(dt_grid.score(X_val, y_val))


# In[ ]:


print("Conf Matrix : \n", confusion_matrix(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Precision", precision_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Recall", recall_score(y_true=y_val, y_pred = test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=1))
print("\nTraom data f1-score for class '0'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=0))


# In[ ]:


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=143)

param_grid = {"classifier__n_estimators" : [150, 250, 300],
              "classifier__max_depth" : [5,8,10],
              "classifier__max_features" : [3, 5, 7],
              "classifier__min_samples_leaf" : [4, 6, 8, 10]}


rf_grid = GridSearchCV(clf, param_grid=dt_param_grid, cv=kfold)

rf_grid.fit(X_train, y_train)
test_pred = rf_grid.predict(X_val)


# In[ ]:


rf_grid.best_params_


# In[ ]:


print("Conf Matrix : \n", confusion_matrix(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Precision", precision_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Recall", recall_score(y_true=y_val, y_pred = test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=1))
print("\nTraom data f1-score for class '0'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=0))


# #### Model-4 Gradient Boosting Trees

# In[ ]:


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('GBM',GradientBoostingClassifier())])
gbm_param_grid = {'GBM__max_depth': [6, 8], 'GBM__subsample': [0.8], 'GBM__max_features':[0.3], 
              'GBM__n_estimators': [30, 40]}

gbm_grid = GridSearchCV(clf, param_grid=gbm_param_grid, cv=5)

gbm_grid.fit(X_train,y_train)


# In[ ]:


test_pred = gbm_grid.predict(X_val)


# In[ ]:


gbm_grid.best_params_


# In[ ]:


print("Conf Matrix : \n", confusion_matrix(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Precision", precision_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Recall", recall_score(y_true=y_val, y_pred = test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=1))
print("\nTraom data f1-score for class '0'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=0))


# #### Model - 5 Class Weights of Loss Function

# In[ ]:


clf_dt = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier())])

dt_param_grid = {'classifier__criterion': ['entropy', 'gini'], 'classifier__max_depth': [6,8,10,12], 
                 "classifier__min_samples_split": [2],"classifier__min_samples_leaf": [1],
                 "classifier__class_weight":['balanced']}

dt_grid_bal = GridSearchCV(clf_dt, param_grid=dt_param_grid, cv=10)
dt_grid_bal.fit(X_train,y_train)


# In[ ]:


test_pred = dt_grid_bal.predict(X_val)


# In[ ]:


dt_grid_bal.best_params_


# In[ ]:


print("Conf Matrix : \n", confusion_matrix(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Precision", precision_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Recall", recall_score(y_true=y_val, y_pred = test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=1))
print("\nTraom data f1-score for class '0'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=0))


# #### Model - 6 Using SMOTE

# In[ ]:


X_train.shape


# In[ ]:


X_train.dtypes

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['M', 'F'], ['M', 'F']]
XX = enc.fit_transform(X)
XX.toarray()
# In[ ]:


# Categorical Pipeline: Pipeline has 2 operation - SimpleImputer() + OneHotEncoder()
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy = 'constant', fill_value='other')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#Preprocessing Execution
clf = Pipeline(steps=[('P_1', p_1)])
x_1_pp = pd.DataFrame(clf.fit_transform(x_1), columns=x_1.columns)
# X_val_pp = pd.DataFrame(clf.transform(X_val))
x_1_pp = x_1_pp.astype('category', axis=1)
x_1_pp.dtypesX = [['India', 1], ['USA', 2], ['India', 3]]
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(X)
enc.get_feature_names()x_1_pp['country'] = x_1_pp['country'].astype('object')
cn_new = pd.get_dummies(x_1_pp,columns=['Gender','Continent'])
l_1 = list(cn_new.columns)
l_1
# In[ ]:


X_train.dtypes


# In[ ]:


clf = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_pp = pd.DataFrame(clf.transform(X_train).toarray())
X_val_pp = pd.DataFrame(clf.transform(X_val).toarray())
X_train_pp.shape


# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
os_data_X, os_data_y = smote.fit_sample(X=X_train_pp, y=y_train)

os_data_X = pd.DataFrame(data=os_data_X)
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of Clean Transactions in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of Fraud Transactions",len(os_data_y[os_data_y['y']==1]))
print("Proportion of Clean Transactions data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of Fraud Transactions in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


# In[ ]:


clf_dt = Pipeline(steps=[('classifier', DecisionTreeClassifier())])

dt_param_grid = {'classifier__criterion': ['entropy', 'gini'], 'classifier__max_depth': [6,8,10,12], 
                 "classifier__min_samples_split": [2, 10, 20],"classifier__min_samples_leaf": [1, 5, 10]}

dt_grid_bal = GridSearchCV(clf_dt, param_grid=dt_param_grid, cv=5)

dt_grid_bal.fit(os_data_X,os_data_y)


# In[ ]:


test_pred = dt_grid_bal.predict(X_val_pp)


# In[ ]:


dt_grid_bal.best_params_


# In[ ]:


print("Conf Matrix : \n", confusion_matrix(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA ACCURACY",accuracy_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Precision", precision_score(y_true=y_val, y_pred = test_pred))
print("\nTrain DATA Recall", recall_score(y_true=y_val, y_pred = test_pred))
print("\nTrain data f1-score for class '1'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=1))
print("\nTraom data f1-score for class '0'",f1_score(y_true=y_val, y_pred = test_pred,pos_label=0))

