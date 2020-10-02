#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


data=pd.read_csv("../input/loan_data_set.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


data.nunique()


# In[ ]:


# Take numerical values
data_numerical = data[['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]


# In[ ]:


# Take categorical values
data_categorical = data.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1)


# In[ ]:


data_categorical.head(5)


# In[ ]:


data_numerical.head(5)


# In[ ]:


## 
data_categorical.groupby('Gender')['Loan_ID'].count()  # .idmax() gives Male 


# In[ ]:


'''
1- using fillna function for nan values
2- I used max count function on with group by (Loan Id).
** for example Gender column has male and female values.Male count bigger than female.
idxmax() gives us male . it is used to be max function.This script shows above code line.You want to use min or other maths function.
i have tested min function but it is decreased accuracy score
'''
for x in data_categorical.columns:
    data_categorical[x]=data_categorical[x].fillna(data_categorical.groupby(x)['Loan_ID'].count().idxmax())


# In[ ]:


# Male counts increases because we fill nan values with male
data_categorical.groupby('Gender')['Loan_ID'].count()  


# In[ ]:


#Now, nan values of numeric columns fill with mean of columns
for x in data_numerical.iloc[:,1:].columns:
    data_numerical[x]=data_numerical[x].fillna(data_numerical[x].mean())


# In[ ]:


data_numerical.info()
data_categorical.info()
# it shows that dataset doesn't exist nan values


# In[ ]:


# Using Dummy variable method for encode Categorical data set
data_cat_encode = pd.get_dummies(data_categorical['Gender'],drop_first=True) 
for x in data_categorical.iloc[:,2:].columns:
    dummy=pd.get_dummies(data_categorical[x],drop_first=True)
    data_cat_encode=pd.concat([data_cat_encode,dummy],axis=1)


# In[ ]:


# Scaling for Numerical data set
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data_numerical.iloc[:,1:] =sc.fit_transform(data_numerical.iloc[:,1:])   


# In[ ]:


data_finally =pd.concat([data_numerical.iloc[:,1:],data_cat_encode],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(data_finally.iloc[:,:-1],data_finally.iloc[:,-1:],test_size=0.33,random_state=0)


# In[ ]:


from sklearn.svm import SVC
svc_rbf=SVC(kernel='rbf')
svc_rbf.fit(X_train,y_train)
y_pred=svc_rbf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

