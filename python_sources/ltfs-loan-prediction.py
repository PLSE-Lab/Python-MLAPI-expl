#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#%%capture
#!pip install fastai==0.7.0
#from fastai.imports import *
#from fastai.structured import *


# In[ ]:


#from fastai.imports import *
#from fastai.structured import *

from pandas_summary import DataFrameSummary
from IPython.display import display
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


# reading the data

train = pd.read_csv('../input/train (1).csv', parse_dates=["Date.of.Birth", "DisbursalDate"])
test = pd.read_csv('../input/test_bqCt9Pv.csv', parse_dates=["Date.of.Birth", "DisbursalDate"])
sub = pd.read_csv('../input/submission (1).csv')

# getting the shapes of the datasets
print("Shape of Train :", train.shape)
print("Shape of Test :", test.shape)
print("Shape of submission :", sub.shape)


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(train[5:100])


# In[209]:


unique_id_train = train['UniqueID']
unique_id_test = test['UniqueID']
data = pd.concat([train, test], axis = 0, ignore_index=True)


# In[151]:





# In[210]:


data['Disbursal_month'] = data['DisbursalDate'].dt.month


# In[211]:


display_all(data.isnull().sum().sort_index()/len(data))
data.isnull().sum()


# In[212]:


#display_all(data.isnull().sum().sort_index()/len(data))
#data.isnull().sum()
"""data['asset_cost_flag'] = 0 #eligible
data['asset_cost_flag'][data['asset_cost']>200000] = 1
data['asset_cost_flag'][data['asset_cost']<40000] = 1
data['asset_cost_flag'].value_counts()"""


# In[213]:


data['disbursed_amount_flag'] = 0 # eligible
data['disbursed_amount_flag'][data['disbursed_amount']<8000]=1 # not eligible
data['disbursed_amount_flag'][data['disbursed_amount']>200000] = 1


# In[214]:


data['Employment.Type'].fillna(1, inplace = True)# not eligible
data['Employment.Type'] = data['Employment.Type'].replace('Self employed', 0)
data['Employment.Type'] = data['Employment.Type'].replace('Salaried', 0)
#data['Employment.Type'] = data['Employment.Type'].replace('Unemployed', 1) 
data['Employment.Type'].value_counts()


# In[215]:


#Age of the customer at the time of loan disbursal
# extracting the year of birth of the customers
data['Year_of_birth'] = data['Date.of.Birth'].dt.year
data['Age']=2018-data['Year_of_birth']
#converting ALL negative ages to zero
data['Age'][data['Age'] < 0] = 0


# In[216]:


#calculating remaining time to retirement 
data['time.to.retire']= 65 - data['Age']
data['time.to.retire'][data['time.to.retire'] < 0] = 0
data['time.to.retire'][data['time.to.retire'] > 47] = 0
data['within.age.limit'] = 1 #eligible for loan
data['within.age.limit'][data['Age'] < 18] = 0
data['within.age.limit'][data['Age'] > 65] = 0
data['within.age.limit'].value_counts()


# In[217]:


# encodings for bureau score(perform cns score distribution)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('No Bureau History Available', 0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Sufficient History Not Available', 0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Not Enough Info available on the customer', 0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Activity seen on the customer (Inactive)',0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Updates available in last 36 months', 0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Only a Guarantor', 0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: More than 50 active Accounts found',0)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('M-Very High Risk', 5)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('L-Very High Risk', 5)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('K-High Risk', 4)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('J-High Risk', 4)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('I-Medium Risk', 3)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('H-Medium Risk', 3)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('G-Low Risk', 2)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('F-Low Risk', 2)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('E-Low Risk', 2)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('D-Very Low Risk', 1)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('C-Very Low Risk', 1)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('B-Very Low Risk', 1)
data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('A-Very Low Risk', 1)

# checing the values in bureau score
data['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[218]:


data['manufacturer_id'] = data['manufacturer_id'].replace(86, 11)
data['manufacturer_id'] = data['manufacturer_id'].replace(45, 10)
data['manufacturer_id'] = data['manufacturer_id'].replace(51, 9)
data['manufacturer_id'] = data['manufacturer_id'].replace(48, 8)
data['manufacturer_id'] = data['manufacturer_id'].replace(49, 7)
data['manufacturer_id'] = data['manufacturer_id'].replace(120, 6)
data['manufacturer_id'] = data['manufacturer_id'].replace(67, 5)
data['manufacturer_id'] = data['manufacturer_id'].replace(145, 4)
data['manufacturer_id'] = data['manufacturer_id'].replace(153, 3)
data['manufacturer_id'] = data['manufacturer_id'].replace(152, 2)
data['manufacturer_id'] = data['manufacturer_id'].replace(156, 1)
data['manufacturer_id'] = data['manufacturer_id'].replace(155, 0)
data['manufacturer_id'].value_counts()


# In[219]:


data['Postal.zone'] = round(data['Current_pincode_ID']/1000)
data['Postal.zone'] = data['Postal.zone'].astype(int)
data['major_city'] = 1 # not a major city
data['major_city'][data['Postal.zone']==1] = 0
data['major_city'][data['Postal.zone']==4] = 0
data['major_city'][data['Postal.zone']==6] = 0
data['major_city'][data['Postal.zone']==7] = 0
data['major_city'].value_counts()


# In[220]:


#Converting the AVERAGE.ACCT.AGE to months
data['AVERAGE.ACCT.AGE_year'] = data['AVERAGE.ACCT.AGE'].apply(lambda x: x.split('yrs')[0])
data['AVERAGE.ACCT.AGE_month'] = data['AVERAGE.ACCT.AGE'].apply(lambda x: x.split(' ')[1])
data['AVERAGE.ACCT.AGE_month'] = data['AVERAGE.ACCT.AGE_month'].apply(lambda x: x.split('mon')[0])
data['AVERAGE.ACCT.AGE_month']= data['AVERAGE.ACCT.AGE_month'].astype(int)
data['AVERAGE.ACCT.AGE_year'] = data['AVERAGE.ACCT.AGE_year'].astype(int)
data['AVERAGE.ACCT.AGE_period']= round((data['AVERAGE.ACCT.AGE_year']*12 + data['AVERAGE.ACCT.AGE_month'])/12,1)

#Converting the CREDIT.HISTORY.LENGTH to years
data['CREDIT.HISTORY.LENGTH_year'] = data['CREDIT.HISTORY.LENGTH'].apply(lambda x: x.split('yrs')[0])
data['CREDIT.HISTORY.LENGTH_month'] = data['CREDIT.HISTORY.LENGTH'].apply(lambda x: x.split(' ')[1])
data['CREDIT.HISTORY.LENGTH_month'] = data['CREDIT.HISTORY.LENGTH_month'].apply(lambda x: x.split('mon')[0])
data['CREDIT.HISTORY.LENGTH_month']= data['CREDIT.HISTORY.LENGTH_month'].astype(int)
data['CREDIT.HISTORY.LENGTH_year'] = data['CREDIT.HISTORY.LENGTH_year'].astype(int)
data['CREDIT.HISTORY.LENGTH_period']= round((data['CREDIT.HISTORY.LENGTH_year']*12 + data['CREDIT.HISTORY.LENGTH_month'])/12,1)
data['CREDIT.HISTORY.LENGTH_period'].value_counts()
data['AVERAGE.ACCT.AGE_period']


# In[221]:


data['loan_tenure_flag'] = 0
data['loan_tenure_flag'][data['AVERAGE.ACCT.AGE_period']<0.25]=1 #not valid
data['loan_tenure_flag'][data['AVERAGE.ACCT.AGE_period']>4]=1 # not valid
data['loan_tenure_flag'].value_counts()


# In[164]:


#data.drop(['VoterID_flag', 'PAN_flag', 'Driving_flag'], axis = 1, inplace = True)


# In[223]:


data['TOTAL.NO.OF.ACCTS'] = data['PRI.NO.OF.ACCTS'] + data['SEC.NO.OF.ACCTS']
data['TOTAL.ACTIVE.ACCTS'] = data['PRI.ACTIVE.ACCTS'] + data['SEC.ACTIVE.ACCTS']
data['TOTAL.OVERDUE.ACCTS'] = data['PRI.OVERDUE.ACCTS'] + data['SEC.OVERDUE.ACCTS']
data['TOTAL.CURRENT.BALANCE'] = data['PRI.CURRENT.BALANCE'] + data['SEC.CURRENT.BALANCE']
data['TOTAL.SANCTIONED.AMOUNT'] = data['PRI.SANCTIONED.AMOUNT'] +data['SEC.SANCTIONED.AMOUNT']
data['TOTAL.DISBURSED.AMOUNT'] = data['PRI.DISBURSED.AMOUNT'] + data['SEC.DISBURSED.AMOUNT']
data['TOTAL.INSTAL.AMT'] = data['PRIMARY.INSTAL.AMT'] + data['SEC.INSTAL.AMT']
data.drop(['PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS',  'PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT',
           'PRI.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.NO.OF.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
           'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT','SEC.ACTIVE.ACCTS', 'SEC.INSTAL.AMT'], axis = 1, inplace = True)


# In[235]:


data['TOTAL.CURRENT.BALANCE'][data['TOTAL.CURRENT.BALANCE']<0]=0
data['TOTAL.SANCTIONED.AMOUNT'][data['TOTAL.SANCTIONED.AMOUNT']<0]=0


# In[237]:


print(data.shape)
print(data.columns)


# In[238]:


data.drop(['Date.of.Birth','Employee_code_ID','UniqueID','supplier_id', 'Year_of_birth', 'Current_pincode_ID', 'MobileNo_Avl_Flag', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH','AVERAGE.ACCT.AGE_year', 'AVERAGE.ACCT.AGE_month','CREDIT.HISTORY.LENGTH_year',
           'CREDIT.HISTORY.LENGTH_month', 'DisbursalDate' ],axis = 1, inplace = True) 


# In[239]:


#data.drop(['Date.of.Birth','disbursed_amount','asset_cost', 'branch_id','Employee_code_ID','State_ID', 'UniqueID', 'Year_of_birth', 'Age', 'Current_pincode_ID', 'supplier_id', 'MobileNo_Avl_Flag', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH','AVERAGE.ACCT.AGE_year', 'AVERAGE.ACCT.AGE_month','CREDIT.HISTORY.LENGTH_year','CREDIT.HISTORY.LENGTH_month', 'DisbursalDate','PERFORM_CNS.SCORE' ]
                 #, axis = 1, inplace = True)
print(data.shape)
print(data.columns)


# In[240]:


data['branch_id'] = data['branch_id'].astype('category')#82 unique
data['State_ID'] = data['State_ID'].astype('category')#22 unique
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['branch_id'] = le.fit_transform(data['branch_id'])
data['State_ID'] = le.fit_transform(data['State_ID'])


# In[ ]:


#display_all(data)


# skewness measure

# In[241]:


data['TOTAL.ACTIVE.ACCTS'] = np.log1p(data['TOTAL.ACTIVE.ACCTS'])
data['TOTAL.CURRENT.BALANCE'] = np.log1p(data['TOTAL.CURRENT.BALANCE'])
data['TOTAL.DISBURSED.AMOUNT']= np.log1p(data['TOTAL.DISBURSED.AMOUNT'])
data['TOTAL.INSTAL.AMT']= np.log1p(data['TOTAL.INSTAL.AMT'])
data['TOTAL.NO.OF.ACCTS']= np.log1p(data['TOTAL.NO.OF.ACCTS'])
data['TOTAL.OVERDUE.ACCTS']= np.log1p(data['TOTAL.OVERDUE.ACCTS'])
data['TOTAL.SANCTIONED.AMOUNT']= np.log1p(data['TOTAL.SANCTIONED.AMOUNT'])


# In[242]:



"""ltv =data['ltv'].median()
cred  = data['CREDIT.HISTORY.LENGTH_period'].median()
accage = data['AVERAGE.ACCT.AGE_period'].median()
newacc = data['NEW.ACCTS.IN.LAST.SIX.MONTHS'].median()
asset=data['asset_cost'].median()
disbamt = data['disbursed_amount'].median()
print(ltv, cred, accage, newacc, asset, disbamt)"""


# In[243]:


"""data['CREDIT.HISTORY.LENGTH_period']= np.log1p(data['CREDIT.HISTORY.LENGTH_period'])
data['AVERAGE.ACCT.AGE_period']= np.log1p(data['AVERAGE.ACCT.AGE_period'])
data['NEW.ACCTS.IN.LAST.SIX.MONTHS']= np.log1p(data['NEW.ACCTS.IN.LAST.SIX.MONTHS'])
data['asset_cost'] = np.log1p(data['asset_cost'])
data['disbursed_amount']=np.log1p(data['disbursed_amount'])
data['ltv']=np.log1p(data['ltv'])"""


# In[246]:


#data['TOTAL.CURRENT.BALANCE'].fillna(0, inplace = True)
#data['TOTAL.SANCTIONED.AMOUNT'].fillna(0, inplace = True)
data.isnull().sum()


# In[248]:


#split into original test and train
data['loan_default'].fillna(2, inplace = True)
test_df = data[data['loan_default'] == 2]
train_df = data[data['loan_default'] != 2]
test_df.drop(['loan_default'], axis = 1, inplace = True)
print("train shape" + str(train_df.shape))
print("test shape"+str(test_df.shape))


# In[ ]:





# In[249]:


#train_df['loan_default'][train_df['within.age.limit']==0] = 1#not eligible for loan
#train_df['loan_default'][train_df['Employment.Type']==0] = 1
# 1 = loan defaulty and 0 non loan defaultys
y_train = train_df['loan_default']
train_df.drop(['loan_default'], axis = 1, inplace = True)
y_train.value_counts()


# In[180]:


get_ipython().system('pip install -U imbalanced-learn')


# In[181]:


from imblearn.over_sampling import SMOTE


# In[250]:


from imblearn.over_sampling import SMOTE

x_resample, y_resample = SMOTE().fit_sample(train_df, y_train.values.ravel()) 

# checking the shape of x_resample and y_resample
print("Shape of x:", x_resample.shape)
print("Shape of y:", y_resample.shape)
#print(x_resample.columns)


# In[ ]:





# In[ ]:


# train and valid sets from train
"""from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_df, y_train, test_size = 0.2, random_state = 0)

# checking the shapes
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)"""


# In[251]:


from sklearn import linear_model #for logistic regression
from sklearn.neural_network import MLPClassifier #for neural network
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict, validation_curve 
#GridSearchCV is used to optimize parameters of the models used
#the other modules and functions 
from sklearn.ensemble import VotingClassifier #for creating ensembles of classifiers


# In[ ]:


# standardization

"""from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
test_df = sc.transform(test_df)"""


# In[ ]:


# RANDOM FOREST CLASSIFIER
"""from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model_rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features=0.5, n_jobs=-1)
kfold = StratifiedKFold(n_splits=5, random_state=7)
cv_results = cross_val_score(model_rf, train_df, y_train, cv=kfold)
print (cv_results.mean()*100, "%")
model_rf.fit(x_train, y_train)

y_pred = model_rf.predict(x_valid)

print("Training Accuracy: ", model_rf.score(x_train, y_train))
print('Testing Accuarcy: ', model_rf.score(x_valid, y_valid))"""


# In[ ]:


"""from sklearn.model_selection import GridSearchCV
m = RandomForestClassifier()
parameters = [{'n_estimators': [100], 
               'min_samples_leaf': [3,5,7, 8, 9], 
               'max_features': ['sqrt', 'log2', 0.5]}
             ]
grid_search = GridSearchCV(estimator = m,
                           param_grid = parameters,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_)
best_accuracy = grid_search.best_score_#given by mean of 10 folds
best_parameters = grid_search.best_params_
print ("best accuracy is {}".format(best_accuracy))
print ("best parametrs are {}".format(best_parameters))"""


# In[ ]:


"""from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model_ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.001)
model_ada.fit(x_train, y_train)

y_pred = model_ada.predict(x_valid)

print("Training Accuracy: ", model_ada.score(x_train, y_train))
print('Testing Accuarcy: ', model_ada.score(x_valid, y_valid))"""


# In[252]:


x = pd.DataFrame(x_resample, columns=train_df.columns)
y_resample


# In[ ]:





# In[ ]:


x.shape


# In[253]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from sklearn.model_selection import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[254]:


from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model_xgb = XGBClassifier(n_estimators=120, learning_rate=1, n_jobs=-1,random_state=42)
#kfold = KFold(n_splits=5,random_state=7)
kfold = StratifiedKFold(n_splits=5, random_state=7)
cv_results = cross_val_score(model_xgb, x, y_resample, cv=kfold, scoring='roc_auc')
print (cv_results.mean()*100, "%")
"""model_xgb.fit(train_df, y_train)

y_pred = model_xgb.predict(x_valid)

print("Training Accuracy: ", model_xgb.score(x_train, y_train))
print('Testing Accuarcy: ', model_xgb.score(x_valid, y_valid))"""
# age score = 79.93900708025433 %
#age, Employmenttype, kfold score =80.%
#age, Employementtype, stratified fold = 80.62
#smote and droping min no. features(35) 88.65
# after log transformation of 7 features(Total group) 88.7837


# In[ ]:


model_xgb.fit( x, y_resample)


# In[ ]:


import pandas as pd
feature_importances = pd.DataFrame(model_xgb.feature_importances_,
                                   index = x.columns,
                                    columns=['importance']).sort_values('importance', 
                                                                        ascending=False)
feature_importances.head


# In[ ]:


"""from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model_lgb = LGBMClassifier()
kfold = KFold(n_splits=5,random_state=7)
cv_results = cross_val_score(model_lgb, x_norm, y_train, cv=kfold)
print (cv_results.mean()*100, "%")
#model_lgb.fit(train_df, y_train)

#y_pred = model_lgb.predict(x_valid)

#print("Training Accuracy: ", model_lgb.score(x_train, y_train))
#print('Testing Accuarcy: ', model_lgb.score(x_valid, y_valid))


# In[ ]:


"""x_mean = train_df.mean()
x_std = train_df.std()
x_norm = (train_df - x_mean)/x_std
print (x_norm.shape)"""


# In[ ]:


"""test_mean = test_df.mean()
test_std = test_df.std()
test_norm = (test_df - test_mean)/test_std
print (test_norm.shape)"""


# In[ ]:


"""logreg = linear_model.LogisticRegression()
kfold = KFold(n_splits=5,random_state=7)
cv_results = cross_val_score(logreg, x_norm, y_train, cv=kfold)
print (cv_results.mean()*100, "%")"""


# In[ ]:


"""logreg = linear_model.LogisticRegression()
param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
grid = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=kfold)
grid.fit(x_norm,y)
print (grid.best_estimator_)
print (grid.best_score_*100, "%")"""


# In[ ]:


"""clf = MLPClassifier(solver='sgd',learning_rate='adaptive', random_state=1, activation='relu', hidden_layer_sizes=(15,),
                n_iter_no_change = 20, early_stopping=True)
kfold = KFold(n_splits=5,random_state=7)
cv_results = cross_val_score(clf, x_norm, y_train, cv=kfold)
print (cv_results.mean()*100, "%")"""


# In[ ]:


#clf.fit(x_norm, y_train)


# In[ ]:


#y_pred_rf = model_rf.predict(x_test)
#y_pred_ada = model_ada.predict(x_test)
y_pred_xgb = model_xgb.predict(test_df)
#y_pred_lgb = model_lgb.predict(test_df)
#y_pred_mpl = clf.predict(test_norm)


# In[ ]:


sub['loan_default']= y_pred_xgb
sub.to_csv("submission_xgb6.csv",index=False)
sub['loan_default'].value_counts()


# In[ ]:


"""clf1 = linear_model.LogisticRegression()
clf2 = MLPClassifier(solver='lbfgs', alpha=1.0,hidden_layer_sizes=(15,), random_state=1, activation='logistic')
eclf = VotingClassifier(estimators=[('lr', clf1), ('nn', clf2)], voting='soft', weights=[1,2])
cv_results = cross_val_score(eclf, x_norm, y_train, cv=kfold)
print (cv_results.mean()*100, "%")"""

