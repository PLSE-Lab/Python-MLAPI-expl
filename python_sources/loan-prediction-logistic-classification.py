#!/usr/bin/env python
# coding: utf-8

# # Import libraries & load dataset

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sms
from sklearn.preprocessing import StandardScaler,LabelEncoder


# In[ ]:


loan = pd.read_csv('../input/analytics-vidhya-loan-prediction/train.csv')


# In[ ]:


loan.head()


# In[ ]:


loan.info()


# In[ ]:


pd.DataFrame([loan.isnull().sum(),loan.isnull().sum()/loan.isnull().count() * 100]).T


# In[ ]:


loan.describe(include=np.object)


# In[ ]:


loantest= pd.read_csv('../input/analytics-vidhya-loan-prediction/test.csv')
loantest.head()


# In[ ]:


loantest.info()


# # Data cleaning

# In[ ]:


objcols = loan.columns[loan.dtypes == np.object]
objcols


# In[ ]:


for col in objcols:
    if (loan[col].isnull().sum() > 0) :
        loan[col].fillna(loan[col].mode()[0],inplace=True)


# In[ ]:


intcols = loan.columns[loan.dtypes != np.object]
intcols


# In[ ]:


for col in intcols:
    if (loan[col].isnull().sum() > 0) :
        loan[col].fillna(loan[col].median(),inplace=True)


# In[ ]:


objcols = loantest.columns[loantest.dtypes == np.object]
for col in objcols:
    if (loantest[col].isnull().sum() > 0) :
        loantest[col].fillna(loantest[col].mode()[0],inplace=True)


# In[ ]:


intcols = loantest.columns[loantest.dtypes != np.object]
for col in intcols:
    if (loantest[col].isnull().sum() > 0) :
        loantest[col].fillna(loantest[col].median(),inplace=True)


# In[ ]:


loan['Loan_Status'] = loan['Loan_Status'].map({'Y':1,'N':0})


# # Check skewness

# In[ ]:


sns.distplot(np.log1p(loan.CoapplicantIncome))


# # Scaling & Encoding

# In[ ]:


dummyTrain = pd.get_dummies(loan.drop(['Loan_Status','Loan_ID','Gender','ApplicantIncome','Loan_Amount_Term'],axis=1))
dummyTrain.head()


# In[ ]:


sc = StandardScaler()


# In[ ]:


scaledTrain = pd.DataFrame(sc.fit_transform(dummyTrain),columns=dummyTrain.columns)
scaledTrain['Loan_Status'] = loan['Loan_Status']
scaledTrain.head()


# In[ ]:


dummyTest = pd.get_dummies(loantest.drop(['Loan_ID','Gender','ApplicantIncome','Loan_Amount_Term'],axis=1))
scaledTest = pd.DataFrame(sc.fit_transform(dummyTest),columns=dummyTest.columns)
scaledTest.head()


# # Apply Model

# In[ ]:


x = scaledTrain.drop('Loan_Status',axis=1)
y = scaledTrain['Loan_Status']


# In[ ]:


lor = LogisticRegression()


# In[ ]:


lor.fit(x,y)


# In[ ]:


ypred = lor.predict(scaledTest)
ypred


# In[ ]:


submission = pd.DataFrame({'Loan_ID':loantest.Loan_ID,'Loan_Status':ypred})
submission['Loan_Status'] = submission['Loan_Status'].map({1:'Y',0:'N'})
submission.head()


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(scaledTrain.corr(),annot=True)


# In[ ]:


# result = sms.Logit(y,x).fit()
# result.summary()

