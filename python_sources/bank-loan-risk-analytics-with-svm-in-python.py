#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  A support vector machine (SVM) is a type of supervised machine learning classification algorithm.
# i am using the risk_analytics data from kaggle site.
# the main objective is that wethear bank loan will give or not. 
# there are two dataset first is train dataset and second is test data.
# intersting thing is that we dont have y variable (response variable).
# i will provide the y variable on basis of predicted the  X_test dataset.


# In[ ]:


# import the library
import numpy as np
import pandas as pd


# In[ ]:


#import the train dataset and testdata set.
train_data = pd.read_csv('../input/risk_analytics_train.csv')
test_data = pd.read_csv('../input/risk_analytics_test.csv')


# In[ ]:


#Preprosessing the training dataset
train_data.shape


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe(include='all')


# In[ ]:


#finding missing values
print(train_data.isnull().sum())
train_data.head()


# In[ ]:


#gender,married,self-emp,dependent,loan_amt_term we will replace using mode value here
#imputing categorical missing data with mode value.

colname=["Gender","Married","Dependents","Self_Employed","Loan_Amount_Term"]
for x in colname[:]:
        train_data[x].fillna(train_data[x].mode()[0],inplace = True)
train_data.isnull().sum()


# In[ ]:


# imputting the missing value in numerical variable replace by mean value
train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(),inplace = True)

train_data.isnull().sum()


# In[ ]:


# imputting the missing value replace by 0 value 
train_data['Credit_History'].fillna(value = 0, inplace = True)

train_data.isnull().sum()

# finally we removed all missing values in categorical variales and numerical variables.


# In[ ]:


# transforming categorical data to numerical
from sklearn import preprocessing
colname1=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'] 

le={}

for x in colname1:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    train_data[x]=le[x].fit_transform(train_data[x])
#converted load status as Y---->1 and N----->0     

train_data.head()


# In[ ]:


# now we  lets the test data preprocessing the data

test_data.head()


# In[ ]:


# findout the missing value in test dataset
test_data.isnull().sum()


# In[ ]:


#imputing categorical missing data with mode value

colname2=["Gender","Dependents","Self_Employed","Loan_Amount_Term"]

for x in colname2[:]:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)
    
print(test_data.isnull().sum())      


# In[ ]:


#imputing categorical missing data with mean value
test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace=True)
print(test_data.isnull().sum())


# In[ ]:


#imputing value for credit_history col differently

test_data["Credit_History"].fillna(value=0,inplace=True)

test_data.isnull().sum()


# In[ ]:


# transforming categorical data to numerical
from sklearn import preprocessing
colname3=['Gender','Married','Education','Self_Employed','Property_Area'] 

le={}

for x in colname3:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname3:
    test_data[x]=le[x].fit_transform(test_data[x])
#converted load status as Y---->1 and N----->0     
test_data.head()


# In[ ]:


#we will now create set of dependant and independant variables
#creating/splitting training and testing datasets and running model
#here in X_train we are taking data from loan id to property_area i.e all independant vars
X_train=train_data.values[:,1:-1]
#in y_test we will take only last col i.e loan_stat
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)


# In[ ]:


# in training we wont have y_test as test dataset doesnt have depended variable so we only create X_test here
#test_data.head()
X_test=test_data.values[:,1:]


# In[ ]:


#we use fit on train ds as its scale might be different from test data according to data present 
#as we have to test the data on training dataset hence we fit only train data and transform train and test.
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_train)
X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)


# In[ ]:


# now we can predict the model
from  sklearn import svm
svc_model = svm.SVC(kernel = 'rbf', C = 1.0, gamma = 0.1)

svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)


# In[ ]:


print(list(Y_pred))


# In[ ]:


# finally we predit the y value other word we can say the depended value from X_test 

