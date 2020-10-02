#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import XGBClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV


# In[ ]:


os.chdir('../input')


# In[ ]:


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')


# In[ ]:


# The columns are matching except the last one
len(train.columns[:-1] == test.columns)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.dtypes


# In[ ]:


# Since, 'MobileNo_Avl_Flag' has 0 variance.. We will be dropping this feature
print(train.MobileNo_Avl_Flag.var())
print(test.MobileNo_Avl_Flag.var())


# In[ ]:


UniqueID = [test.UniqueID]
train.drop(['UniqueID','MobileNo_Avl_Flag'], axis=1, inplace=True)
test.drop(['UniqueID','MobileNo_Avl_Flag'], axis=1, inplace=True)


# In[ ]:


# Setting up time marker

d_marker= '21-04-19'
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%d-%m-%y")
    d2 = datetime.strptime(d2, "%d-%m-%y")
    return abs((d2 - d1).days)


# In[ ]:



# age as on 1-1-2019 (in yrs)
train['Date.of.Birth'] = train['Date.of.Birth'].apply(lambda x:  days_between(x,d_marker)/365)
# Calculating time (in yrs) after disbursal
train['DisbursalDate']= train['DisbursalDate'].apply(lambda x:  days_between(x,d_marker)/365)
# age as on 1-1-2019 (in yrs)
test['Date.of.Birth'] = test['Date.of.Birth'].apply(lambda x:  days_between(x,d_marker)/365)
# Calculating time (in yrs) after disbursal
test['DisbursalDate']= test['DisbursalDate'].apply(lambda x:  days_between(x,d_marker)/365)


# In[ ]:


# Converting the given 'CREDIT.HISTORY.LENGTH' in months

train['CREDIT.HISTORY.LENGTH']= train['CREDIT.HISTORY.LENGTH'].apply(lambda x: (re.sub('[a-z]','',x)).split())
train['CREDIT.HISTORY.LENGTH']= train['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0])*12+int(x[1]))

# Converting the given 'AVERAGE.ACCT.AGE' in months
train['AVERAGE.ACCT.AGE']= train['AVERAGE.ACCT.AGE'].apply(lambda x: (re.sub('[a-z]','',x)).split())
train['AVERAGE.ACCT.AGE']= train['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0])*12+int(x[1]))

# Converting the given 'CREDIT.HISTORY.LENGTH' in months
test['CREDIT.HISTORY.LENGTH']= test['CREDIT.HISTORY.LENGTH'].apply(lambda x: (re.sub('[a-z]','',x)).split())
test['CREDIT.HISTORY.LENGTH']= test['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0])*12+int(x[1]))

# Converting the given 'AVERAGE.ACCT.AGE' in months
test['AVERAGE.ACCT.AGE']= test['AVERAGE.ACCT.AGE'].apply(lambda x: (re.sub('[a-z]','',x)).split())
test['AVERAGE.ACCT.AGE']= test['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0])*12+int(x[1]))


# In[ ]:


ct_col= ['branch_id','supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Employment.Type', 'State_ID', 'Employee_code_ID', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag','Driving_flag', 'Passport_flag','PERFORM_CNS.SCORE.DESCRIPTION']


con_col= ['disbursed_amount', 'asset_cost', 'ltv','Date.of.Birth','DisbursalDate', 'PERFORM_CNS.SCORE', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
       'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
       'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',
       'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
       'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES']


# In[ ]:


train[ct_col].head()


# #  Missing Values

# In[ ]:


train.isnull().sum()


# In[ ]:


train['Employment.Type'].value_counts()


# In[ ]:


print('Percentage of missing values is {0}% for the training set'.format(round(100*train['Employment.Type'].isnull().sum()/len(train),2)))
print('Percentage of missing values is {0}% for the test set'.format(round(100*test['Employment.Type'].isnull().sum()/len(train),2)))


# In[ ]:


# Substituting the null values by third category 'unknown'
train.fillna('unknown', inplace=True)
test.fillna('unknown', inplace=True)
train['Employment.Type'].value_counts()


# In[ ]:


pd.concat([train.isnull().sum(),test.isnull().sum()], sort= False, axis=1)


# In[ ]:


le = LabelEncoder()
train.iloc[:,8] = le.fit_transform(train.iloc[:,8])
train.iloc[:,18] = le.fit_transform(train.iloc[:,18])
test.iloc[:,8] = le.fit_transform(test.iloc[:,8])
test.iloc[:,18] = le.fit_transform(test.iloc[:,18])


# In[ ]:


for i in con_col:
    train[i] = (train[i]- min(train[i])) / (max(train[i])- min(train[i]))
    test[i] = (test[i]- min(test[i])) / (max(test[i])- min(test[i]))


# In[ ]:


X= train.drop('loan_default', axis=1)
y= train.loan_default


# In[ ]:


X= pd.concat([X[con_col], pd.get_dummies(X[ct_col], drop_first=True)], sort=False, axis=1)


# In[ ]:


# Handing Class im balance
smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(X, y)


# In[ ]:


# Not used in the model training
# Splitting the datasets
# X_tr,X_ts, y_tr, y_ts = train_test_split(X_smt,y_smt, test_size= .3)


# In[ ]:


cv =StratifiedKFold(n_splits=10,shuffle=True,random_state=45)
# xgb= XGBClassifier(n_estimators=120, learning_rate=1, n_jobs=-1,random_state=42)
#pre = cross_val_predict(xgb, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')
#pre= pre[:,1]


# In[ ]:


# print("auc score =\t" ,roc_auc_score(y_smt, pre))


# In[ ]:


# xgb= XGBClassifier(n_estimators=120, learning_rate=1, n_jobs=-1,random_state=42)
# scores = cross_val_score(xgb, cv=cv, X=X_smt,y=y_smt, verbose=1,scoring='roc_auc')
# print("auc\t=\t",scores.mean())


# In[ ]:


rf= RandomForestClassifier(n_estimators=300,verbose=1, n_jobs=-1,random_state=42)
rf.fit(X_smt,y_smt)


# In[ ]:


pre_out= rf.predict_proba(test)


# In[ ]:


u = pd.read_csv('test.csv')
u = u.iloc[:,0]
u['prob']= pre_out
u.to_csv('mycsvfile.csv',index=False)


# In[ ]:


rf= RandomForestClassifier(n_estimators=300,verbose=1, n_jobs=-1,random_state=42)
pre_rf = cross_val_predict(rf, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')


# In[ ]:


print("auc score =\t" ,roc_auc_score(y_smt, pre_rf[:,1]))


# In[ ]:


rf_f= RandomForestClassifier(n_estimators=300,verbose=1, n_jobs=-1,random_state=42)
rf.fit(X_smt,y_smt)
pre_out = rf.predict_proba(test)[:,1]


# In[ ]:


df_out = pre_out.to_csv('out.csv',index=False)


# In[ ]:


gbc= GradientBoostingClassifier()
gbc.fit(X_tr,y_tr)
pre= gbc.predict_proba(X_ts)
pre = pre[:,1]


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_ts,pre))

