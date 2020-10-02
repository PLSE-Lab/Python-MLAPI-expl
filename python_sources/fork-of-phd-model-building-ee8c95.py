#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[20]:


from sklearn.metrics import classification_report


# In[2]:


data=pd.read_csv("../input/feature_engineering (1).csv")


# In[3]:


test_data=pd.read_csv("../input/test_feture_engineering.csv")


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


test_data.head()


# In[7]:


test_data.isnull().sum()


# In[8]:


data.dtypes


# In[9]:


data.columns


# In[10]:


feature_cols=['Quantity', 'TotalSalesValue', 'PricePerUnit', 'diff_Avg_Qty_ProdID',
       'diff_Avg_Tsale_ProdID', 'diff_Avg_PPerUnit_ProdID', 'diff_Avg_Qty',
       'diff_Avg_TSale', 'diff_Avg_PPerUnit', 'diff_Avg_TsaleVal_ProdID',
       'diff_Avg_Qty_pProdID']


# In[11]:


len(test_data.columns)


# In[12]:


len(data.columns)


# In[13]:


X=data[feature_cols]


# In[15]:


X.columns


# In[16]:


test_data.columns


# In[17]:


y=data['Suspicious']


# In[ ]:





# In[18]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[19]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Decision tree 

# In[21]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
clf = DecisionTreeClassifier()


# In[23]:


y_train.head()


# In[24]:


clf.fit(X_train,y_train)


# In[25]:


train_predictions_dt=clf.predict(X_train)
test_predictions_dt=clf.predict(X_test)


# In[27]:


print(classification_report(y_train, train_predictions_dt))


# In[26]:


print(classification_report(y_test, test_predictions_dt))


# let's build a Randomn forest classifier

# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[57]:


ran_clf = RandomForestClassifier(n_jobs=2, random_state=0,class_weight='balanced')


# In[58]:


ran_clf.fit(X_train, y_train)


# In[59]:


train_predictions_ran_dt=ran_clf.predict(X_train)
test_predictions_ran_dt=ran_clf.predict(X_test)


# In[60]:


print(classification_report(y_train,train_predictions_ran_dt))


# In[61]:


print(classification_report(y_test, test_predictions_ran_dt))


# let us try Svm 

# In[34]:


#from sklearn.svm import SVC


# In[35]:


#svc_clf = SVC(gamma='auto')


# In[36]:


#svc_clf.fit(X_train, y_train) 


# In[37]:


#train_predictions_svc=svc_clf.predict(X_train)
#test_predictions_svc=svc_clf.predict(X_test)


# In[39]:


#print(classification_report(y_train, train_predictions_svc))


# In[40]:


#print(classification_report(y_test, test_predictions_svc))


# In[41]:


from sklearn.ensemble import BaggingClassifier


# In[42]:


bag_clf=BaggingClassifier(n_estimators=10)


# In[43]:


bag_clf.fit(X_train,y_train)


# In[44]:


train_predictions_bag=bag_clf.predict(X_train)
test_predictions_bag=bag_clf.predict(X_test)


# In[46]:


print(classification_report(y_train, train_predictions_bag))


# In[47]:


print(classification_report(y_test, test_predictions_bag))


# In[48]:


from sklearn.ensemble import AdaBoostClassifier


# In[49]:


ada_clas=AdaBoostClassifier()


# In[50]:


ada_clas.fit(X_train,y_train)


# In[51]:


train_predictions_ada_clas=ada_clas.predict(X_train)
test_predictions_ada_clas=ada_clas.predict(X_test)


# In[52]:


classification_report(y_train, train_predictions_ada_clas)


# In[53]:


print(classification_report(y_test, test_predictions_ada_clas))


# In[62]:


from sklearn.ensemble import GradientBoostingClassifier


# In[65]:


learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train,y_train)
    print("Learning rate: ", learning_rate)
    train_predictions_gb_clas=gb.predict(X_train)
    print(classification_report(y_train,train_predictions_gb_clas))
    test_predictions_gb_clas=ada_clas.predict(X_test)
    print(classification_report(y_test,test_predictions_gb_clas))

