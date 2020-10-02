#!/usr/bin/env python
# coding: utf-8

# # Import Libraries 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read Data

# In[ ]:


train_data = pd.read_csv('../input/train_AV3.csv')
train_data.head()


# # Data Info

# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# #  Visualization & Feature Engineering

# In[ ]:


sns.countplot(x='Gender',data=train_data)


# In[ ]:


train_data['Gender'][train_data['Gender'].isnull()]='Male'


# In[ ]:


sns.countplot(x='Married',data=train_data)


# In[ ]:


train_data['Married'][train_data['Married'].isnull()]='Yes'


# In[ ]:


train_data['LoanAmount'][train_data['LoanAmount'].isnull()]= train_data['LoanAmount'].mean()


# In[ ]:


sns.countplot(x='Loan_Amount_Term',data=train_data)


# In[ ]:


train_data['Loan_Amount_Term'][train_data['Loan_Amount_Term'].isnull()]='360'


# In[ ]:


sns.countplot(x='Self_Employed',data=train_data)


# In[ ]:


train_data['Self_Employed'][train_data['Self_Employed'].isnull()]='No'


# In[ ]:


sns.countplot(x='Credit_History',data=train_data)


# In[ ]:


train_data['Credit_History'][train_data['Credit_History'].isnull()]='1'


# In[ ]:


train_data.info()


# In[ ]:


sns.countplot(x='Dependents',data=train_data)


# In[ ]:


train_data['Dependents'][train_data['Dependents'].isnull()]='0'


# In[ ]:


train_data.loc[train_data.Dependents=='3+','Dependents']= 4


# In[ ]:


train_data.tail()


# In[ ]:


train_data.loc[train_data.Loan_Status=='N','Loan_Status']= 0
train_data.loc[train_data.Loan_Status=='Y','Loan_Status']=1


# In[ ]:


train_data.loc[train_data.Gender=='Male','Gender']= 0
train_data.loc[train_data.Gender=='Female','Gender']=1


# In[ ]:


train_data.loc[train_data.Married=='No','Married']= 0
train_data.loc[train_data.Married=='Yes','Married']=1


# In[ ]:


train_data.loc[train_data.Education=='Graduate','Education']= 0
train_data.loc[train_data.Education=='Not Graduate','Education']=1


# In[ ]:


train_data.loc[train_data.Self_Employed=='No','Self_Employed']= 0
train_data.loc[train_data.Self_Employed=='Yes','Self_Employed']=1


# In[ ]:


property_area= pd.get_dummies(train_data['Property_Area'],drop_first=True)


# In[ ]:


train_data= pd.concat([train_data,property_area],axis=1)


# In[ ]:


train_data.head()


# # Data Modeling

# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X= train_data.drop(['Loan_ID','Property_Area','Loan_Status'],axis=1)
y = train_data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# # Predictions 

# In[ ]:


prediction= logmodel.predict(X_test)


# # Accuracy of Model

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,prediction))


# # Visualization & Feature Engineering of Test Data

# In[ ]:


data_test= pd.read_csv('../input/test_AV3.csv')


# In[ ]:


data_test.head()


# In[ ]:


data_test.info()


# In[ ]:


sns.countplot('Gender',data=data_test)


# In[ ]:


data_test['Gender'][data_test['Gender'].isnull()]='Male'


# In[ ]:


sns.countplot('Dependents',data=data_test)


# In[ ]:


data_test['Dependents'][data_test['Dependents'].isnull()]=0


# In[ ]:


data_test.loc[data_test.Dependents=='3+','Dependents']= 4


# In[ ]:


sns.countplot('Self_Employed',data=data_test)


# In[ ]:


data_test['Self_Employed'][data_test['Self_Employed'].isnull()]='No'


# In[ ]:


sns.countplot('Loan_Amount_Term',data=data_test)


# In[ ]:


data_test['Loan_Amount_Term'][data_test['Loan_Amount_Term'].isnull()]=360


# In[ ]:


sns.countplot('Credit_History',data=data_test)


# In[ ]:


data_test['Credit_History'][data_test['Credit_History'].isnull()]=1


# In[ ]:


sns.countplot('Property_Area',data=data_test)


# In[ ]:


data_test['Property_Area'][data_test['Property_Area'].isnull()]='Urban'


# In[ ]:


data_test.head()


# In[ ]:


data_test['LoanAmount'][data_test['LoanAmount'].isnull()]= data_test['LoanAmount'].mean()


# In[ ]:


data_test.loc[data_test.Gender=='Male','Gender']= 0
data_test.loc[data_test.Gender=='Female','Gender']=1


# In[ ]:


data_test.loc[data_test.Married=='No','Married']= 0
data_test.loc[data_test.Married=='Yes','Married']=1


# In[ ]:


data_test.loc[data_test.Education=='Graduate','Education']= 0
data_test.loc[data_test.Education=='Not Graduate','Education']=1


# In[ ]:


data_test.loc[data_test.Self_Employed=='No','Self_Employed']= 0
data_test.loc[data_test.Self_Employed=='Yes','Self_Employed']=1


# In[ ]:


property_area= pd.get_dummies(data_test['Property_Area'],drop_first=True)


# In[ ]:


data_test = pd.concat([data_test,property_area],axis=1)


# In[ ]:


X_data_test= data_test.drop(['Loan_ID','Property_Area'],axis=1)


# In[ ]:


X_data_test.head()


# # Predictions of Test Data

# In[ ]:


data_test['Loan_Status']= logmodel.predict(X_data_test)


# In[ ]:


data_frame=data_test[['Loan_ID','Loan_Status']]


# In[ ]:


data_frame.loc[data_frame.Loan_Status==0,'Loan_Status']='N'
data_frame.loc[data_frame.Loan_Status==1,'Loan_Status']='Y'


# In[ ]:


data_frame.head()


# In[ ]:


data_frame.to_csv('Loan Predictions Submission.csv',index=0)


# In[ ]:




