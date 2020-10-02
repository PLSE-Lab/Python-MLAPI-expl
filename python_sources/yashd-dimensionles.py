#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv("C:/Users/Yash/Desktop/SM/CRM_Train/CRM_Train.csv")
train.head()


# In[ ]:


train = train.drop(['Loan ID','Customer ID','Months since last delinquent'],1)
train.head()


# In[ ]:


train.shape


# In[ ]:


train = train.dropna()
train.shape


# In[ ]:


train.dtypes


# In[ ]:


from sklearn.preprocessing import LabelBinarizer, LabelEncoder


# In[ ]:


train['Home Ownership'].value_counts()


# In[ ]:


lb = LabelBinarizer()
le = LabelEncoder()
train['Loan Status'] = lb.fit_transform(train['Loan Status'])
train['Term'] = lb.fit_transform(train['Term'])
#train[' Years in current job'] = le.fit_transform(train[' Years in current job'])
#train['Home Ownership'] = le.fit_transform(train['Home Ownership'])
train['Purpose'] = le.fit_transform(train['Purpose'])
train['Number of Open Accounts'] = le.fit_transform(train['Number of Open Accounts'])
train['Number of Credit Problems'] = le.fit_transform(train['Number of Credit Problems'])
train['Bankruptcies'] = le.fit_transform(train['Bankruptcies'])
train['Tax Liens'] = le.fit_transform(train['Tax Liens'])


# In[ ]:


train = pd.get_dummies(train,columns = ['Home Ownership',' Years in current job'])
train.head()


# In[ ]:


train.columns


# In[ ]:


train['Loan Status'].astype('category')
train['Term'].astype('category')
train['Purpose'].astype('category')
train['Number of Open Accounts'].astype('category')
train['Number of Credit Problems'].astype('category')
train['Bankruptcies'].astype('category')
train['Tax Liens'].astype('category')


# In[ ]:


train.dtypes


# In[ ]:


train.columns


# In[ ]:


y_train = train['Loan Status']
X_train = train


# In[ ]:


X_train.shape


# In[ ]:


X_train.drop(['Loan Status'],1,inplace = True)
X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


test = pd.read_csv("C:/Users/Yash/Desktop/SM/CRM_Test/CRM_TestData.csv")
test.head()


# In[ ]:


test = test.drop(['Customer ID','Months since last delinquent','Unnamed: 2'],1)
test.head()


# In[ ]:





# In[ ]:


test.shape


# In[ ]:


test = test.dropna()
test.shape


# In[ ]:


test.dtypes


# In[ ]:


test['Term'] = lb.fit_transform(test['Term'])
#test['Years in current job'] = le.fit_transform(test['Years in current job'])
#test['Home Ownership'] = le.fit_transform(test['Home Ownership'])
test['Purpose'] = le.fit_transform(test['Purpose'])
test['Number of Open Accounts'] = le.fit_transform(test['Number of Open Accounts'])
test['Number of Credit Problems'] = le.fit_transform(test['Number of Credit Problems'])
test['Bankruptcies'] = le.fit_transform(test['Bankruptcies'])
test['Tax Liens'] = le.fit_transform(test['Tax Liens'])


# In[ ]:


test = pd.get_dummies(test, columns = ['Years in current job','Home Ownership'])
test.head()


# In[ ]:


test.columns


# In[ ]:


test['Term'].astype('category')
test['Purpose'].astype('category')
test['Number of Open Accounts'].astype('category')
test['Number of Credit Problems'].astype('category')
test['Bankruptcies'].astype('category')
test['Tax Liens'].astype('category')


# In[ ]:


X_test = test.drop(['Loan ID'],1)
X_test.shape


# In[ ]:


X_test.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfclass = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10,random_state=42)


# In[ ]:


r11 = rfclass.fit(X_train,y_train)


# In[ ]:


y_pred1 = r11.predict(X_test)


# In[ ]:


r11.feature_importances_


# In[ ]:


y_pred1 = list(y_pred1)


# In[ ]:


d = pd.DataFrame(y_pred1, columns = ['Loan Status'])
d.head()


# In[ ]:


df = pd.DataFrame(test['Loan ID'], columns = ['Loan ID'])


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


d1 = pd.concat([df,d], axis = 1)
d1.head()


# In[ ]:


d1.to_csv('C:/Users/Yash/Desktop/SM/CRM_Test/Submission_YashD.csv')


# In[ ]:




