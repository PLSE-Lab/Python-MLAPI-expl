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


os.listdir("../input/phd-dataset")


# In[ ]:


data=pd.read_excel("../input/phd-dataset/Train.xlsx")


# In[ ]:


test_data=pd.read_excel("../input/phd-test-data-actual/Test.xlsx")


# In[ ]:


data.head()


# In[ ]:


test_data.head()


# In[ ]:


print(data.shape)
print(test_data.shape)


# In[ ]:


target_column=data['Suspicious']


# In[ ]:


data=data.drop(axis=1,columns='Suspicious')


# In[ ]:


report_id_column=data['ReportID']


# In[ ]:


data=data.drop(axis=1,columns='ReportID')


# In[ ]:


test_Data_report_id_column=test_data['ReportID']


# In[ ]:


test_data=test_data.drop(axis=1,columns='ReportID')


# In[ ]:


print(data.shape)
print(test_data.shape)


# In[ ]:


from sklearn import preprocessing
le_SalesPersonID = preprocessing.LabelEncoder()


# In[ ]:


le_ProductID=preprocessing.LabelEncoder()


# In[ ]:


target_column=np.where(target_column=='indeterminate', 3, target_column) 


# In[ ]:


target_column=np.where(target_column=='Yes', 1, target_column) 
target_column=np.where(target_column=='No', 2, target_column) 


# In[ ]:


np.unique(target_column)


# In[ ]:


le_SalesPersonID.fit(data['SalesPersonID'])


# In[ ]:


le_ProductID.fit(data['ProductID'])


# In[ ]:


data['SalesPersonID']=le_SalesPersonID.transform(data['SalesPersonID'])


# In[ ]:





# In[ ]:


data['ProductID']=le_ProductID.transform(data['ProductID'])


# In[ ]:


data['PricePerUnit']=data['TotalSalesValue']/data['Quantity']


# In[ ]:


test_data['PricePerUnit']=test_data['TotalSalesValue']/test_data['Quantity']


# In[ ]:


data=data.drop(axis=1,columns=['SalesPersonID','ProductID'])


# In[ ]:


test_data=test_data.drop(axis=1,columns=['SalesPersonID','ProductID'])


# In[ ]:


#let us now standardize the columns
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:


scaler.fit(data)


# In[ ]:





# In[ ]:


data=pd.DataFrame(scaler.transform(data),columns=[data.columns])


# In[ ]:


test_data=pd.DataFrame(scaler.transform(test_data),columns=[test_data.columns])


# In[ ]:


#due to class imbalance problem let us do SMoting
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target_column, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[ ]:


X_test_copy=X_test


# In[ ]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '2': {} \n".format(sum(y_train==2)))
print("Before OverSampling, counts of label '3': {} \n".format(sum(y_train==3)))


# In[ ]:


y_train=y_train.astype('int')


# In[ ]:


y_test=y_test.astype('int')


# In[ ]:


sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train,y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))


# In[ ]:


print("after OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("after OverSampling, counts of label '2': {} \n".format(sum(y_train_res==2)))
print("after OverSampling, counts of label '3': {} \n".format(sum(y_train_res==3)))


# In[ ]:


print(X_train_res.shape)
print(y_train_res.shape)


# In[ ]:





# In[ ]:


#let us try using logistic regression on this model


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg=LogisticRegression()
logreg.fit(X_train_res,y_train_res)


# In[ ]:


train_predictions_logreg=logreg.predict(X_train)
test_predictions_logreg=logreg.predict(X_test)


# In[ ]:


print(classification_report(y_train, train_predictions_logreg))


# In[ ]:


print(classification_report(y_test, test_predictions_logreg))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train_res,y_train_res)


# In[ ]:


train_predictions_dt=logreg.predict(X_train)
test_predictions_dt=logreg.predict(X_test)


# In[ ]:


print(classification_report(y_train, train_predictions_dt))
print(classification_report(y_test,test_predictions_dt))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
ran_clf = RandomForestClassifier(n_jobs=2, random_state=0,class_weight='balanced')


# In[ ]:


ran_clf.fit(X_train_res, y_train_res)


# In[ ]:


train_predictions_ran_dt=ran_clf.predict(X_train)
test_predictions_ran_dt=ran_clf.predict(X_test)


# In[ ]:


print(classification_report(y_train,train_predictions_ran_dt))


# In[ ]:


print(classification_report(y_test, test_predictions_ran_dt))


# In[ ]:


from sklearn.ensemble import BaggingClassifier
bag_clf=BaggingClassifier(n_estimators=10)


# In[ ]:


bag_clf.fit(X_train_res,y_train_res)


# In[ ]:


train_predictions_bag=bag_clf.predict(X_train)
test_predictions_bag=bag_clf.predict(X_test)


# In[ ]:


print(classification_report(y_train, train_predictions_bag))
print(classification_report(y_test, test_predictions_bag))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train_res,y_train_res)
    print("Learning rate: ", learning_rate)
    train_predictions_gb_clas=gb.predict(X_train)
    print('train_classification_report')
    print(classification_report(y_train,train_predictions_gb_clas))
    test_predictions_gb_clas=gb.predict(X_test)
    print('test_classification_report')
    print(classification_report(y_test,test_predictions_gb_clas))


# In[ ]:


#let us submit this and see the result


# In[ ]:


test_data_predictions=gb.predict(test_data)


# In[ ]:


Submission1=pd.DataFrame(test_Data_report_id_column)


# In[ ]:


Submission1['Suspicious']=test_data_predictions


# In[ ]:


Submission1.to_csv('Submission1.csv')

