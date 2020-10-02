#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=['object', 'bool'])


# In[ ]:


df.Contract.value_counts()


# In[ ]:


df.Churn.value_counts()


# In[ ]:


df.Dependents.value_counts()


# In[ ]:


df.Churn.value_counts(normalize=True)


# In[ ]:


df.dtypes


# In[ ]:


#I am changing Churn to 1 and 0 for analysis purpose
df['Churn']=df['Churn'].replace('No',0)
df['Churn']=df['Churn'].replace('Yes',1)


# In[ ]:


#I am changing Seniorcitizon as yes or no
df['SeniorCitizen']=df['SeniorCitizen'].replace(1,'Yes')
df['SeniorCitizen']=df['SeniorCitizen'].replace(0,'No')


# In[ ]:


# Here Totalcharges actually is integer but it showing object so i converted into integer
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


# In[ ]:


df.head()
pd.set_option('display.max_columns', None)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:



df.isnull().sum()


# In[ ]:


#Defining categorical variables
categorical_features=df.select_dtypes(include=[object])


# In[ ]:


categorical_features.columns


# In[ ]:


# Removing unwanted coloumns
df.drop(['customerID'],axis=1,inplace=True)


# In[ ]:


df.TotalCharges.isnull().sum()


# In[ ]:


df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())


# In[ ]:


df.TotalCharges.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


## below are manually done encoding


# In[ ]:


df['gender']=df['gender'].replace('Male',1)
df['gender']=df['gender'].replace('Female',0)


# In[ ]:


df['Partner']=df['Partner'].replace('No',0)
df['Partner']=df['Partner'].replace('Yes',1)


# In[ ]:


df['Dependents']=df['Churn'].replace('No',0)
df['Dependents']=df['Dependents'].replace('Yes',1)


# In[ ]:


df['SeniorCitizen']=df['SeniorCitizen'].replace('No',0)
df['SeniorCitizen']=df['SeniorCitizen'].replace('Yes',1)


# In[ ]:


df['PhoneService']=df['PhoneService'].replace('No',0)
df['PhoneService']=df['PhoneService'].replace('Yes',1)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


### One Hot Encoding by ceating dummies


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


oe=OneHotEncoder()


# In[ ]:


final_df=pd.get_dummies(columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'],data=df)


# In[ ]:


final_df.head()


# In[ ]:


final_df.shape


# In[ ]:


y=final_df['Churn']


# In[ ]:


X=final_df.drop(['Churn'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:


y_train.shape


# In[ ]:


X.shape


# In[ ]:


final_df.dtypes


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr=LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_train,y_train)


# In[ ]:


lr_predict=lr.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn import metrics


# In[ ]:


cm=confusion_matrix(y_test,lr_predict)


# In[ ]:


cm


# In[ ]:


metrics.accuracy_score(y_test,lr_predict)


# In[ ]:


metrics.roc_auc_score(y_test,lr_predict)


# In[ ]:


metrics.classification_report(y_test,lr_predict)


# In[ ]:


metrics.f1_score(y_test,lr_predict)


# In[ ]:


print('Actual:', y_test.values[0:25])
print('Predicted:', lr_predict[0:25])


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier(criterion='entropy',n_estimators=1000,max_depth=100,oob_score=True,random_state=42)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


rf.score(X_train,y_train)


# In[ ]:


rf_predict=rf.predict(X_test)


# In[ ]:


metrics.confusion_matrix(y_test,rf_predict)


# In[ ]:


metrics.accuracy_score(y_test,rf_predict)


# In[ ]:


metrics.f1_score(y_test,rf_predict)


# In[ ]:


import pandas as pd
feature_importances1 = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_importances1


# In[ ]:




