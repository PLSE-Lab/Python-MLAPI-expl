#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from itertools import combinations

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/loan-prediction/train_loan.csv')


# In[ ]:


df.head()


# In[ ]:


#I see some 'object' type and need to deal with them
df.info()


# In[ ]:


df.describe()


# For this dataset I will comment my choices for each column in the related cell

# In[ ]:


#Loan_ID: unuseful fot the model, can be removed


# In[ ]:


# gender: There are some NaN but there is no way to infer this, so unfortunately I remove the related rows
df = df[df['Gender'].notna()]
df.Gender.unique()


# In[ ]:


#Married: I assume NaN for this feature means "Divorced"

missing_married = len(df[ df.Married.isna()])
print('Missing married before: %d' %(missing_married))

nan_married = df[df.Married.isna()].index
df.loc[nan_married,'Married'] = 'Divorced'

missing_married = len(df[df.Married.isna()])
print('Missing married after : %d' %(missing_married))
df.Married.unique()


# In[ ]:


# I assume that when Self_Employed is set to nan it means the guy lives with a revenue
def fix_self_employed(self_employed):
    ret = self_employed
    if str(self_employed) == 'nan':
        ret = 'Revenue'
    #print(ret)
    return ret

df['Self_Employed'] = df.apply(lambda row : fix_self_employed(row['Self_Employed']),axis=1)


# In[ ]:


# Dependent (number): I tried different setup here but nothing changed that much the results.
# I keep this where I set category '1' for more than 3 dependents, 0 for less than '3', -1 when the value is not applicable

def fix_dependents(num_dependents):
    
    map = {'3+' : 1, '3': 0, '2': 0, '1': 0, '0': 0, 'nan': -1 }
    return map[str(num_dependents)]
           
df['Dependents'] = df.apply(lambda row : fix_dependents(row['Dependents']),axis=1)


# In[ ]:


# Education: all values are there. I do not see any issue in this column
df['Education'].unique()


# In[ ]:


# ApplicantIncome: all values are there. I do not see any issue in this column
len(df[df['ApplicantIncome'].isna()]) == 0


# In[ ]:


# CoapplicantIncome: when there is no coapplicant this value is set to zero
coapplicant_income_set_to_zero = len(df[df['CoapplicantIncome'] == 0])
coapplicant_income_set_to_nan = len(df[df['CoapplicantIncome'].isna()])
print(coapplicant_income_set_to_zero,coapplicant_income_set_to_nan)

# create a new feature as the sum of ApplicantIncome and CoapplicantIncome
df['TotalApplicantIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']


# In[ ]:


# loanAmount : there are 22 rows without any value 
# (the dataset is already tiny so better replace them with mean rather than get rid of them)

print(len(df[df['LoanAmount'].isna()]))

loan_amount_mean = df['LoanAmount'].mean()

def replace_NaN_loan_Amount(loan_amount, replace_value):
    ret = loan_amount
    if str(loan_amount) == 'nan':
        #print(str(loan_amount))
        ret = replace_value
    return ret

df['LoanAmount'] = df['LoanAmount'].fillna(loan_amount_mean)

print(len(df[df['LoanAmount'].isna()]))


# In[ ]:


# Loan_Amount_Term : there are 14 rows without any value
# (the dataset is already tiny so better replace them with mean rather than get rid of them)

print(len(df[df['Loan_Amount_Term'].isna()]))
df.Loan_Amount_Term.unique()

loan_amount_term_mean = df['Loan_Amount_Term'].mean()

def replace_NaN_Loan_Amount_Term(loan_amount, replace_value):
    ret = loan_amount
    if str(loan_amount) == 'nan':
        #print(str(loan_amount))
        ret = replace_value
    return ret

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(loan_amount_term_mean)

print(len(df[df['Loan_Amount_Term'].isna()]))


# In[ ]:


#Credit_History : some people (50) don't have a credit history. I assign them a new category = 2. (unrated)
    
print(len(df[df['Credit_History'].isna()]))

df['Credit_History'] = df['Credit_History'].fillna(2.)

print(len(df[df['Credit_History'].isna()]))
df['Credit_History'].unique()


# In[ ]:


#Property_Area: this column seems fine: all values set and no NaN
len(df[df.Property_Area.isna()])
df.Property_Area.unique()


# In[ ]:


# Encode all values
for column in df.columns:
    if df[column].dtype == 'O':
        # object values are label-encoded
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].apply(str))
    # numerical values are scaled
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))


# At this point the dataset is consistent

# In[ ]:


# drop unuseful/meaningless features
features_to_be_dropped = ['Loan_ID','ApplicantIncome', 'CoapplicantIncome']
df.drop(features_to_be_dropped,inplace=True, axis=1)


# In[ ]:


# split in train and test
Y = df['Loan_Status'].values
X = df.drop(['Loan_Status'], axis=1).values

X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size=0.3, random_state=1)


# In[ ]:


# try a Dummy Classifier to see what is the reference accuracy for my model
dc = DummyClassifier()
dc.fit(X_train,Y_train)
Y_train_pred = dc.predict(X_train)
Y_test_pred = dc.predict(X_test)

print('Dummy ACCURACY train: %.4f, test: %.4f' %(accuracy_score(Y_train,Y_train_pred), accuracy_score(Y_test,Y_test_pred)))


# In[ ]:


# try with RandomForest
rfc = RandomForestClassifier(n_estimators=30)
rfc.fit(X_train,Y_train)

Y_train_pred = rfc.predict(X_train)
Y_test_pred = rfc.predict(X_test)

print('RandomForest ACCURACY train: %.4f, test: %.4f' %(accuracy_score(Y_train,Y_train_pred), accuracy_score(Y_test,Y_test_pred)))


# In[ ]:


# try with XGBoost
xg_reg = xgb.XGBClassifier(n_estimators=30)

xg_reg.fit(X_train,Y_train)

Y_train_pred = xg_reg.predict(X_train)
Y_test_pred = xg_reg.predict(X_test)

print('XGBoost ACCURACY train: %.4f, test: %.4f' %(accuracy_score(Y_train,Y_train_pred), accuracy_score(Y_test,Y_test_pred)))


# In[ ]:


# We have a lot of feature and I wonder whether a differnt combination of them can lead to better result,
#so here I try all the feature's combinations and see what happens

feature_list = []
for column in df.columns:
    if column == 'Loan_Status' or column == 'Loan_ID':
        continue
    feature_list.append(column)
print(feature_list)

for num_features_step in range(1,len(feature_list)+1):

    combs = combinations(feature_list, num_features_step) 
    features_for_model = []
    
    for comb in combs:
        features_for_model = [feature for feature in comb]
        print(features_for_model)
        # drop unuseful/meaningless features
        Y = df['Loan_Status'].values
        X = df[features_for_model].values
        X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size=0.3, random_state=1)
        rfc = RandomForestClassifier(n_estimators=30)
        rfc.fit(X_train,Y_train)
        Y_train_pred = rfc.predict(X_train)
        Y_test_pred = rfc.predict(X_test)
        print('---ITERATION---')
        print(features_for_model)
        print('RandomForestClassifier ACCURACY train: %.4f, test: %.4f' %(accuracy_score(Y_train,Y_train_pred), accuracy_score(Y_test,Y_test_pred)))
        xg_reg = xgb.XGBClassifier(n_estimators=30)
        xg_reg.fit(X_train,Y_train)
        Y_train_pred = xg_reg.predict(X_train)
        Y_test_pred = xg_reg.predict(X_test)
        print('XGBClassifier          ACCURACY train: %.4f, test: %.4f' %(accuracy_score(Y_train,Y_train_pred), accuracy_score(Y_test,Y_test_pred)))


# On my PC, with the features set as described above, I see that a relatively high score is reached by using only the 'Credit History' feature
# 
# **Features: ['Credit_History']**
# 
# **RandomForestClassifier ACCURACY train: 0.8190, test: 0.7845**
# 
# **XGBClassifier          ACCURACY train: 0.8190, test: 0.7845**
# 
# 
# With the following two features only I already reach the max accuracy seen, with quite balanced values between train and test set:
# 
# **Features: ['Self_Employed', 'Credit_History']**
# 
# **RandomForestClassifier ACCURACY train: 0.8214, test: 0.8011**
# 
# **XGBClassifier          ACCURACY train: 0.8214, test: 0.8011**
# 
# 
# Adding more features to the model does not improve my results :-|
# 
# 
# I ran out of ideas on this dataset: if you have any comments/suggestion/criticism/ideas please share them in the comments
