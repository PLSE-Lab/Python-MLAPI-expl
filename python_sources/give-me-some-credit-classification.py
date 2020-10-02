#!/usr/bin/env python
# coding: utf-8

# **Problem statement**
#    -Predicting the probability that somebody will experience financial distress in the next two years.which can make banks a       guess at the probability of default, are use to determine whether or not a loan should be granted.

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#reading the data
sampleEntry = pd.read_csv('../input/credit/sampleEntry.csv')
train = pd.read_csv('../input/credit/cs-training.csv')
test = pd.read_csv('../input/credit/cs-test.csv')


# In[ ]:


#dimension of the data
print(train.shape)
print(test.shape)


# The dataset cantains **12** features along with **150000** observations.
# 
# The description for the 12 features is given below: <br>
# Variable Name	Description	Type
# - ``SeriousDlqin2yrs``	Person experienced 90 days past due delinquency or worse	Y/N
# - ``RevolvingUtilizationOfUnsecuredLines``	Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits	percentage
# - ``age``	Age of borrower in years	integer
# - ``NumberOfTime3059DaysPastDueNotWorse``	Number of times borrower has been 30-59 days past due but no worse in the last 2 years.	integer
# - ``DebtRatio``	Monthly debt payments, alimony,living costs divided by monthy gross income	percentage
# - ``MonthlyIncome``	Monthly income	real
# - ``NumberOfOpenCreditLinesAndLoans``	Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)	integer
# - ``NumberOfTimes90DaysLate``	Number of times borrower has been 90 days or more past due.	integer
# - ``NumberRealEstateLoansOrLines``	Number of mortgage and real estate loans including home equity lines of credit	integer
# - ``NumberOfTime60-89DaysPastDueNotWorse``	Number of times borrower has been 60-89 days past due but no worse in the last 2 years.	integer
# - ``NumberOfDependents``	Number of dependents in family excluding themselves (spouse, children etc.)	integer

# In[ ]:


#getting first five observations
train.head()


# In[ ]:


#getting first five observations 
test.head()


# In[ ]:


#describing train data
train.describe()


# **checking null values**

# In[ ]:


print(train.isnull().sum())


# In[ ]:


print(test.isnull().sum())


# **Checking unique items in each column**

# In[ ]:


train.nunique()


# **Imputing Null values**

# In[ ]:


train['MonthlyIncome'].fillna(train['MonthlyIncome'].mean(),inplace=True)


# In[ ]:


train['NumberOfDependents'].fillna(train['NumberOfDependents'].mode()[0], inplace=True)


# In[ ]:


test['MonthlyIncome'].fillna(test['MonthlyIncome'].mean(),inplace=True)


# In[ ]:


test['NumberOfDependents'].fillna(test['NumberOfDependents'].mode()[0], inplace=True)


# In[ ]:


print(train.isnull().sum())


# In[ ]:


print(test.isnull().sum())


# **classes in Target variable**

# In[ ]:


#plot two tyep classe "0" and "1"
sns.countplot(x='SeriousDlqin2yrs',data=train)
plt.show()


# **correlation**

# In[ ]:


cor=train.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,annot=True,ax=ax)


# In[ ]:


X = train.drop('SeriousDlqin2yrs',1)
y = train['SeriousDlqin2yrs']


# In[ ]:


train.columns


# **Splitting Data**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#splitting data into train and test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=568)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# **XGBOOST**

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


xgb = XGBClassifier(n_jobs=-1) 
 
# Use a grid over parameters of interest
param_grid = {
                  'n_estimators' :[100,150,200,250,300],
                  "learning_rate" : [0.001,0.01,0.0001,0.05, 0.10 ],
                  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3 ],
                  "colsample_bytree" : [0.5,0.7],
                  'max_depth': [3,4,6,8]
              }


# In[ ]:


xgb_randomgrid = RandomizedSearchCV(xgb, param_distributions=param_grid, cv=5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb_randomgrid.fit(X_train,y_train)')


# In[ ]:


best_est = xgb_randomgrid.best_estimator_


# In[ ]:


y_pred = best_est.predict_proba(X_train)
y_pred = y_pred[:,1]


# In[ ]:


from sklearn.metrics import auc,roc_curve
fpr,tpr,_ = roc_curve(y_train, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


X_test = test.drop('SeriousDlqin2yrs',1)
y_test=best_est.predict_proba(X_test)
y_test= y_test[:,1]


# In[ ]:


print(y_test)


# In[ ]:


sampleEntry["Probability"]=y_test
sampleEntry.head()


# In[ ]:


sampleEntry.to_csv("submission.csv",index=False)


# **Random Forest using SMOTE**

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


smote = SMOTE(random_state=0)

os_data_X,os_data_y=smote.fit_sample(X_train,y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt') 
 
# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [9, 18, 27, 36, 100, 150],
           "max_depth" : [2,3,5,7,9],
           "min_samples_leaf" : [2, 4]}


# In[ ]:


rfc_randomgrid = RandomizedSearchCV(rfc, param_distributions=param_grid, cv=5)


# In[ ]:


rfc_randomgrid.fit(os_data_X,os_data_y)


# In[ ]:


best_est1 = rfc_randomgrid.best_estimator_


# In[ ]:


y_pred1 = best_est1.predict_proba(X_train)
y_pred1 = y_pred1[:,1]


# In[ ]:


y_test1=best_est1.predict_proba(X_test)
y_test1= y_test1[:,1]


# In[ ]:


fpr,tpr,_ = roc_curve(y_train, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


sampleEntry["Probability"]=y_test1
sampleEntry.head()


# In[ ]:


sampleEntry.to_csv("submission1.csv",index=False)


# **KNN_model**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


y_pred2 = knn.predict_proba(X_train)
y_pred2 = y_pred2[:,1]


# In[ ]:


fpr,tpr,_ = roc_curve(y_train, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


md_KNN = KNeighborsClassifier()

neighbors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
param_grid = dict(n_neighbors=neighbors)


# In[ ]:


KNN_GridSearch = GridSearchCV(md_KNN,param_grid=param_grid,cv=10)


# In[ ]:


KNN_GridSearch.fit(X_train,y_train)


# In[ ]:


best_est2=KNN_GridSearch.best_estimator_


# In[ ]:


y_pred3 = best_est2.predict_proba(X_train)
y_pred3 = y_pred3[:,1]


# In[ ]:


fpr,tpr,_ = roc_curve(y_train, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


y_test2=best_est2.predict_proba(X_test)
y_test2= y_test2[:,1]


# In[ ]:


sampleEntry["Probability"]=y_test2
sampleEntry.head()


# In[ ]:


sampleEntry.to_csv("submission2.csv",index=False)


# **Conclusion**

# So, we have seen that AUC for  XGboost is around 89%  and also achieved score of 0.86 which is very close to 1 . Therefore, it is inferred that XGBoost is the suitable model for this dataset.SMOTE using with Random forest achieved a score of 0.85 which is also preferable.As KNN is performng well for train and slight deforming for test data getting a score of 0.5.

# In[ ]:




