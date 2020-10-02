#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/Loan.train.csv')
df1 = pd.read_csv('../input/Loan.test.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df1.head()


# In[ ]:


df1.shape


# In[ ]:


df.columns=df.columns.str.lower()


# In[ ]:


df.head()


# In[ ]:


df1.columns=df1.columns.str.lower()


# In[ ]:


df1.head()


# In[ ]:


##made the function to perform value counts of cateogrical variable


# In[ ]:


def valuecounts(value):
    return value.value_counts()


# In[ ]:


valuecounts(df['gender'])


# In[ ]:


valuecounts(df['married'])


# In[ ]:


valuecounts(df['dependents'])


# In[ ]:


valuecounts(df['education'])


# In[ ]:


valuecounts(df['self_employed'])


# In[ ]:


valuecounts(df['property_area'])


# In[ ]:


valuecounts(df['loan_status'])


# In[ ]:


##value counts for test data


# In[ ]:


valuecounts(df1['gender'])


# In[ ]:


valuecounts(df1['married'])


# In[ ]:


valuecounts(df1['dependents'])


# In[ ]:


valuecounts(df1['education'])


# In[ ]:


valuecounts(df1['education'])


# In[ ]:


valuecounts(df1['self_employed'])


# In[ ]:


valuecounts(df1['property_area'])


# In[ ]:


df.dtypes


# In[ ]:


df.describe().T


# In[ ]:


## making histogram function to visualise Numerical variable


# In[ ]:


def histogram(value):
    return value.plot.hist()


# In[ ]:


histogram(df['applicantincome'])


# In[ ]:


histogram(df['coapplicantincome'])


# In[ ]:


histogram(df['loanamount'])


# In[ ]:


histogram(df['loan_amount_term'])


# In[ ]:


histogram(df['credit_history'])


# In[ ]:


##Histogram for test datatype


# In[ ]:


histogram(df1['applicantincome'])


# In[ ]:


histogram(df1['coapplicantincome'])


# In[ ]:


histogram(df1['loanamount'])


# In[ ]:


histogram(df1['loan_amount_term'])


# In[ ]:


histogram(df1['credit_history'])


# In[ ]:


##Treatment of outliers


# In[ ]:


def boxplot(value):
    return value.plot.box()


# In[ ]:


boxplot(df['applicantincome'])


# In[ ]:


df['applicantincome'].describe()


# In[ ]:


df['applicantincome']=df['applicantincome'].clip(2000,6000)


# In[ ]:


boxplot(df['applicantincome'])


# In[ ]:


boxplot(df['coapplicantincome'])


# In[ ]:


df['coapplicantincome'].describe()


# In[ ]:


df['coapplicantincome']=df['coapplicantincome'].clip(1000,3000)


# In[ ]:


boxplot(df['coapplicantincome'])


# In[ ]:


boxplot(df['loanamount'])


# In[ ]:


df['loanamount'].describe()


# In[ ]:


df['loanamount']=df['loanamount'].clip(100,200)


# In[ ]:


boxplot(df['loanamount'])


# In[ ]:


boxplot(df['loan_amount_term'])


# In[ ]:


df['loan_amount_term'].describe()


# In[ ]:


df['loan_amount_term']=df['loan_amount_term'].clip(360,480)


# In[ ]:


boxplot(df['loan_amount_term'])


# In[ ]:


boxplot(df['credit_history'])


# In[ ]:


df['credit_history'].describe()


# In[ ]:


df.dtypes


# In[ ]:


df.corr()


# In[ ]:


##Bivariate analysis


# In[ ]:


married = pd.crosstab(df['loan_status'],df['married']).plot(kind="bar",stacked=True,figsize=(5,5))


# In[ ]:


self_employed = pd.crosstab(df['loan_status'],df['self_employed']).plot(kind="bar",stacked=True,figsize=(5,5))


# In[ ]:


property_area = pd.crosstab(df['loan_status'],df['property_area']).plot(kind="bar",stacked=True,figsize=(5,5))


# In[ ]:


df.isna().sum()


# In[ ]:


df1.isna().sum()


# In[ ]:


df1['gender'].value_counts()


# In[ ]:


##Impute Na value from each column 


# In[ ]:


df['gender'].value_counts()


# In[ ]:


df['gender']=df['gender'].fillna('Male')


# In[ ]:


df1['gender']=df1['gender'].fillna('Male')


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum(),df1.isna().sum()


# In[ ]:


df['married'].value_counts()


# In[ ]:


df['married']=df['married'].fillna('Yes')


# In[ ]:


c = {'3+':3,'2':2,'1':1,'0':0}


# In[ ]:


df['dependents']=df['dependents'].map(c)


# In[ ]:


df1['dependents']=df1['dependents'].map(c)


# In[ ]:


df1['dependents'].value_counts()


# In[ ]:


df['dependents']=df['dependents'].fillna(0)


# In[ ]:


df1['dependents'] = df1['dependents'].fillna(0)


# In[ ]:


df.isna().sum()


# In[ ]:


df['self_employed'].value_counts()


# In[ ]:


df1['self_employed'].value_counts()


# In[ ]:


df['self_employed']=df['self_employed'].fillna('No')


# In[ ]:


df1['self_employed']=df['self_employed'].fillna('No')


# In[ ]:


df1.isna().sum()


# In[ ]:


df['loanamount']=df['loanamount'].fillna(df['loanamount'].mean())


# In[ ]:


df1['loanamount']=df1['loanamount'].fillna(df['loanamount'].mean())


# In[ ]:


df1.isna().sum()


# In[ ]:


df['loan_amount_term']=df['loan_amount_term'].fillna(360)


# In[ ]:


df1['loan_amount_term']=df1['loan_amount_term'].fillna(360)


# In[ ]:


df['credit_history'].value_counts()


# In[ ]:


df['credit_history']=df['credit_history'].fillna(1)


# In[ ]:


df1['credit_history']=df1['credit_history'].fillna(360)


# In[ ]:


##Add the two column (applicant income and coapplicant income)


# In[ ]:


df.columns


# In[ ]:


df['sumofincome']=df['applicantincome']+df['coapplicantincome']


# In[ ]:


df1['sumofincome']=df1['applicantincome']+df1['coapplicantincome']


# In[ ]:


delcolumns =['applicantincome','coapplicantincome']


# In[ ]:


df.drop(delcolumns,axis=1,inplace=True)


# In[ ]:


df.columns


# In[ ]:


df1.drop(delcolumns,axis=1,inplace=True)


# In[ ]:


df1.columns


# In[ ]:


##Label Encoding


# In[ ]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['gender','married','dependents','education','self_employed','property_area','loan_status']
le = LabelEncoder()
for i in var_mod:
   df[i] = le.fit_transform(df[i])
df.dtypes


# In[ ]:


df['dependents'].value_counts()


# In[ ]:


var_mod = ['gender','married','dependents','education','self_employed','property_area']
le = LabelEncoder()
for i in var_mod:
    df1[i] = le.fit_transform(df1[i])
df1.dtypes


# In[ ]:


df1


# In[ ]:


##Remove the loan_id columns


# In[ ]:


df.drop(columns='loan_id',axis=1,inplace=True)


# In[ ]:


df1.drop(columns='loan_id',axis=1,inplace=True)


# In[ ]:


df.dtypes


# In[ ]:



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[ ]:


##Splitting the data


# In[ ]:


train,test = train_test_split(df,test_size=0.10,random_state=0)


# In[ ]:



print('shape of training data : ',train.shape)
print('shape of testing data',test.shape)


# In[ ]:


##Separate the target and independent variable


# In[ ]:


train_x = train.drop(columns=['loan_status'],axis=1)
train_y = train['loan_status']
test_x = test.drop(columns=['loan_status'],axis=1)
test_y = test['loan_status']


# In[ ]:


##Build the model and train the model
model = LogisticRegression()


# In[ ]:


model.fit(train_x,train_y)


# In[ ]:


predict=model.predict(test_x)


# In[ ]:


print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))


# In[ ]:


predict1=model.predict(df1)


# In[ ]:


predict1.shape

