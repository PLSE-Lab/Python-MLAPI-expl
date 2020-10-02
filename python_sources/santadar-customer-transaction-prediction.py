#!/usr/bin/env python
# coding: utf-8

# ## Santader Customer Prediction
# #### Chintan Chitroda

# In this, we identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve the problem.

# we have an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.
# 
# The task is to predict the value of target column in the test set.
# 
# ### Note:
# * The final output file named 'Predicted.csv' is generated using XGBoost algorithm.
# * The Rfe score has been recorded and is stored in list and for cross check you can run the markdown column above it.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


train.head(4)


# In[ ]:


test.head(5)


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


test.isnull().sum().sum()


# #### No null values

# In[ ]:


train.target.value_counts()


# In[ ]:


train.drop('ID_code',axis=1,inplace=True)


# In[ ]:


ids = test.ID_code
test.drop('ID_code',axis=1,inplace=True)


# ## ML Modelling

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[ ]:


X = train.drop('target',axis=1)      ## Passing out TFID out put to Train
y = train[['target']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7,random_state=101)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=50,random_state=101,
                             class_weight={1:1},verbose=1,n_jobs=-1)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))
print('Report:\n',classification_report(y_test, y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, y_pred))


# ###### As we see that the data is highly imbalanced so it cannot detect 1 

# ## Using RFE Recurisve Feature Elimination

# In[ ]:


train.target.value_counts()


# In[ ]:


from sklearn.feature_selection import RFE

## Takes 1 hr
rfe = RFE(rfc, 50)
rfe.fit(X_train,y_train)
# In[ ]:


#print(X_train.columns[rfe.support_])


# In[ ]:


### Rfe Generated Columns
rfe_result = ['var_0', 'var_1', 'var_2', 'var_6', 'var_9', 'var_12', 'var_13',
       'var_21', 'var_22', 'var_26', 'var_33', 'var_34', 'var_40', 'var_44',
       'var_53', 'var_75', 'var_76', 'var_78', 'var_80', 'var_81', 'var_91',
       'var_92', 'var_94', 'var_99', 'var_108', 'var_109', 'var_110',
       'var_115', 'var_121', 'var_122', 'var_123', 'var_133', 'var_139',
       'var_146', 'var_147', 'var_148', 'var_154', 'var_155', 'var_164',
       'var_165', 'var_166', 'var_169', 'var_170', 'var_174', 'var_177',
       'var_179', 'var_184', 'var_190', 'var_191', 'var_198']


# In[ ]:


len(rfe_result)


# In[ ]:


rfc.fit(X_train[rfe_result],y_train)
y_pred = rfc.predict(X_test[rfe_result])


# In[ ]:


print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))
print('Report:\n',classification_report(y_test, y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, y_pred))


# ### Under Smapling the imbalanced dataset

# In[ ]:


train[train.target ==1]


# In[ ]:


new_train = train[train.target ==0].sample(n=20100)


# In[ ]:


new_train = new_train.append(train[train.target ==1])


# In[ ]:


new_train


# ### Rfe Selected columns

# In[ ]:


X = new_train.drop('target',axis=1)      ## Passing out TFID out put to Train
y = new_train[['target']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=20,random_state=101,
                             class_weight={1:1},verbose=1,n_jobs=-1)


# In[ ]:


rfc.fit(X_train[rfe_result],y_train)


# In[ ]:


y_pred = rfc.predict(X_test[rfe_result])


# In[ ]:


print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))
print('Report:\n',classification_report(y_test, y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, y_pred))


# ### XGBoost

# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBClassifier(max_depth=7, 
                             random_state =101,
                             class_weight= 'balanced')


# In[ ]:


model_xgb.fit(X_train[rfe_result],y_train)


# In[ ]:


y_pred = model_xgb.predict(X_test[rfe_result])


# In[ ]:


print("F1 Score :",f1_score(y_pred,y_test))
print('Report:\n',classification_report(y_test, y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, y_pred))


# ###### We see the XGBoost works Fnatastic when trained with Rfe undersampled columns/data and provide best F1 score so we predict test file with same

# In[ ]:


sol = model_xgb.predict(test[rfe_result])


# In[ ]:


solution = pd.DataFrame()
solution['ids'] = ids
solution['target'] = sol
solution.to_csv('predicted_target.csv',index=False)


# ### Thank you

# In[ ]:


temp = train[rfe_result[1:10]]
temp['target'] = train['target']


# In[ ]:


temp.sample(2000,random_state=500).target.value_counts()
sns.pairplot(temp.sample(2000,random_state=500),hue='target')


# ### We see the Data is scammbled so we use K nearest neighbour for the Prediction

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3,leaf_size=15)


# In[ ]:


knn.fit(X_train[rfe_result],y_train)


# In[ ]:


y_pred = knn.predict(X_test[rfe_result])


# In[ ]:


print("F1 Score :",f1_score(y_pred,y_test))
print('Report:\n',classification_report(y_test, y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test, y_pred))


# ### Thank you
