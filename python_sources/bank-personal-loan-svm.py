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


data = pd.read_excel('/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# In[ ]:


data.head()


# In[ ]:


#Missing Or Null data points
data.isnull().sum()
data.isna().sum()


# In[ ]:


data.shape


# In[ ]:


data.describe().transpose()


# Information on the features or attributes
# 
# 
# The attributes can be divided accordingly :
# 
# The variable ID does not add any interesting information. There is no association between a person's customer ID andloan, also it does not provide any general conclusion for future potential loan customers. We can neglect this information for our model prediction.
# 
# 
# The binary category have five variables as below:
# Personal Loan - Did this customer accept the personal loan offered in the last campaign? This is our target variable
# Securities Account - Does the customer have a securities account with the bank?
# CD Account - Does the customer have a certificate of deposit (CD) account with the bank?
# Online - Does the customer use internet banking facilities?
# Credit Card - Does the customer use a credit card issued by UniversalBank?
# 
# 
# Interval variables are as below:
# Age - Age of the customer
# Experience - Years of experience
# Income - Annual income in dollars
# CCAvg - Average credit card spending
# Mortage - Value of House Mortgage
# 
# 
# Ordinal Categorical Variables are:
# Family - Family size of the customer
# Education - education level of the customer
# 
# 
# The nominal variable is :
# ID
# Zip Code

# In[ ]:


data.nunique()


# In[ ]:


# there are 52 records with negative experience. Before proceeding any further we need to clean the same
data[data['Experience'] < 0]['Experience'].count()


# In[ ]:


#clean the negative variable
dfExp = data.loc[data['Experience'] >0]
negExp = data.Experience < 0
column_name = 'Experience'
mylist = data.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience


# In[ ]:


# there are 52 records with negative experience
negExp.value_counts()


# > The following code does the below steps:
# 
# For the record with the ID, get the value of Age column
# For the record with the ID, get the value of Education column
# Filter the records matching the above criteria from the data frame which has records with positive experience and take the median
# Apply the median back to the location which had negative experience

# In[ ]:


for id in mylist:
    age = data.loc[np.where(data['ID']==id)]["Age"].tolist()[0]
    education = data.loc[np.where(data['ID']==id)]["Education"].tolist()[0]
    df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]
    exp = df_filtered['Experience'].median()
    data.loc[data.loc[np.where(data['ID']==id)].index, 'Experience'] = exp


# In[ ]:


# checking if there are records with negative experience
data[data['Experience'] < 0]['Experience'].count()


# In[ ]:


data.describe().transpose()


# In[ ]:


x=data.iloc[:,[1,3,4,5,6,7,8,10,11,12,13]].values
y=data.iloc[:,9].values


# In[ ]:


# Divide the data as train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=4)
x_train.shape


# In[ ]:


#modelling SVM
from sklearn import svm
classifier=svm.SVC(kernel='rbf',gamma='auto',C=1)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_predict)
print(c)
Accuracy=sum(np.diag(c))/(np.sum(c))
Accuracy


# Accuracy came out to be 91.2%

# Lets implement kernal SVM

# In[ ]:


from sklearn import svm
classifier=svm.SVC(kernel='linear',gamma='auto',C=1)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_predict)
print(c)
Accuracy=sum(np.diag(c))/(np.sum(c))
Accuracy


# Accuracy slightly increased to 92.08%
