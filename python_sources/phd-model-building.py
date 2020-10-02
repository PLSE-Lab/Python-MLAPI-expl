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


data=pd.read_csv("../input/feature_engineering.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data=data.drop(axis=1,columns=['ProductID','SalesPersonID'])


# In[ ]:


data.columns


# In[ ]:


feature_cols = ['Quantity', 'TotalSalesValue',
       'price_Per_Unit', 'Average_qty_guy_prdID',
       'Average_TotalSales_guy_prdID', 'Average_TotalSales_guy',
       'Average_Quantity_guy', 'Average_Quantity_Product',
       'Average_TotalSales_ProductID']


# In[ ]:


X=data[feature_cols]


# In[ ]:


y=data['Suspicious']


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


train_predictions=logreg.predict(X_train)
test_predictions=logreg.predict(X_test)


# In[ ]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, test_predictions)
cnf_matrix


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_train, train_predictions))
print("Precision:",metrics.precision_score(y_train, train_predictions,average='macro'))
print("Recall:",metrics.recall_score(y_train, train_predictions,average='macro'))
print("F1 score:",metrics.f1_score(y_train,train_predictions,average='macro'))


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, test_predictions))


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, test_predictions))
print("Precision:",metrics.precision_score(y_test, test_predictions,average='macro'))
print("Recall:",metrics.recall_score(y_test, test_predictions,average='macro'))
print("F1 score:",metrics.f1_score(y_test,test_predictions,average='macro'))


# Decision tree 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
clf = DecisionTreeClassifier()


# In[ ]:


y_train


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


train_predictions_dt=clf.predict(X_train)
test_predictions_dt=clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_train, train_predictions_dt))
print("Precision:",metrics.precision_score(y_train, train_predictions_dt,average='macro'))
print("Recall:",metrics.recall_score(y_train, train_predictions_dt,average='macro'))
print("F1 score:",metrics.f1_score(y_train,train_predictions_dt,average='macro'))


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, test_predictions_dt))
print("Precision:",metrics.precision_score(y_test, test_predictions_dt,average='macro'))
print("Recall:",metrics.recall_score(y_test, test_predictions_dt,average='macro'))
print("F1 score:",metrics.f1_score(y_test,test_predictions_dt,average='macro'))


# In[ ]:


print(classification_report(y_test, test_predictions_dt))


# let's build a Randomn forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


ran_clf = RandomForestClassifier(n_jobs=2, random_state=0)


# In[ ]:


ran_clf.fit(X_train, y_train)


# In[ ]:


train_predictions_ran_dt=ran_clf.predict(X_train)
test_predictions_ran_dt=ran_clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_train, train_predictions_ran_dt))
print("Precision:",metrics.precision_score(y_train, train_predictions_ran_dt,average='macro'))
print("Recall:",metrics.recall_score(y_train, train_predictions_ran_dt,average='macro'))
print("F1 score:",metrics.f1_score(y_train,train_predictions_ran_dt,average='macro'))


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, test_predictions_ran_dt))
print("Precision:",metrics.precision_score(y_test, test_predictions_ran_dt,average='macro'))
print("Recall:",metrics.recall_score(y_test, test_predictions_ran_dt,average='macro'))
print("F1 score:",metrics.f1_score(y_test,test_predictions_ran_dt,average='macro'))


# In[ ]:


print(classification_report(y_test, test_predictions_ran_dt))


# let us try Svm 

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_clf = SVC(gamma='auto')


# In[ ]:


svc_clf.fit(X_train, y_train) 


# In[ ]:


train_predictions_svc=svc_clf.predict(X_train)
test_predictions_svc=svc_clf.predict(X_test)


# In[ ]:


classification_report(y_train, train_predictions_svc)


# In[ ]:


print(classification_report(y_test, test_predictions_svc))


# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


bag_clf=BaggingClassifier(n_estimators=10)


# In[ ]:


bag_clf.fit(X_train,y_train)


# In[ ]:


train_predictions_bag=bag_clf.predict(X_train)
test_predictions_bag=bag_clf.predict(X_test)


# In[ ]:


classification_report(y_train, train_predictions_bag)


# In[ ]:


print(classification_report(y_test, test_predictions_bag))


# In[ ]:


bag_clf=BaggingClassifier(n_estimators=10,base_estimator=ran_clf)


# In[ ]:


bag_clf.fit(X_train,y_train)


# In[ ]:


train_predictions_bagged_rnd=bag_clf.predict(X_train)
test_predictions_bagged_rnf=bag_clf.predict(X_test)


# In[ ]:


classification_report(y_train, train_predictions_bagged_rnd)


# In[ ]:


print(classification_report(y_test, test_predictions_bagged_rnf))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


ada_clas=AdaBoostClassifier()


# In[ ]:


ada_clas.fit(X_train,y_train)


# In[ ]:


train_predictions_ada_clas=ada_clas.predict(X_train)
test_predictions_ada_clas=ada_clas.predict(X_test)


# In[ ]:


classification_report(y_train, train_predictions_ada_clas)


# In[ ]:


print(classification_report(y_test, test_predictions_ada_clas))


# In[ ]:


from sklearn import neighbors


# In[ ]:


kn1= neighbors.KNeighborsRegressor(n_neighbors=10,weights='uniform')
kn2= neighbors.KNeighborsRegressor(n_neighbors=10,weights='distance')


# In[ ]:


bag_clf=BaggingClassifier(n_estimators=10,base_estimator=kn1)


# In[ ]:


bag_clf.fit(X_train,y_train)


# In[ ]:


train_predictions_bagged_kn1=bag_clf.predict(X_train)
test_predictions_bagged_kn1=bag_clf.predict(X_test)

