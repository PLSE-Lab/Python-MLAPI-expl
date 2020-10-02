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
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


bnk_prt_df = pd.read_csv("../input/PS_20174392719_1491204439457_log.csv")


# In[ ]:


bnk_prt_df.head(10)


# <h3>Analysis of dataset for null values</h3>

# In[ ]:


## Heatmap of isnull indicates that this dataset does not have any null values
sns.heatmap(bnk_prt_df.isnull())


# In[ ]:


bnk_prt_df.describe()


# <h3>Analysis of isFraud True or False</h3>

# In[ ]:


## Analyzing the data, seems like positive fraud data is very minimal in this dataset. Find below the stats for the same
sns.countplot(x='isFraud',data=bnk_prt_df)
print(bnk_prt_df['isFraud'].value_counts())


# <h3>Analysis of type vs isFraud in the dataset </h3>

# In[ ]:


## Analyzing the data, seems like positive fraud data is very minimal in this dataset. 
## Analyzing only the positive fraud dataset, it seems like they belong only to the datatype - Transfer, Cash_out
## Other types doesn't seem to contribute to positive fraud data
## Find below the stats for the same

sns.countplot(x='type',data=bnk_prt_df[bnk_prt_df['isFraud']==1],hue='isFraud')
print(bnk_prt_df['type'].value_counts())
print(bnk_prt_df[bnk_prt_df['isFraud']==1]['type'].unique())


# <h3>Analysis of isFlaggedFraud Vs isFraud in the dataset</h3>

# In[ ]:


## 16 data which are identified as true isFlaggedFraud are identified as Fraud
## 8197 data out of 63.62 lakhs data which are identified with true isFlaggedFraud are identified as Fraud
## Among 8197 data, there are equal splits for Transfer and Cashout types

sns.countplot(x='isFlaggedFraud',data=bnk_prt_df,hue='isFraud')


# In[ ]:


bnk_prt_df[bnk_prt_df['isFlaggedFraud']==1]['isFraud'].value_counts()


# In[ ]:


bnk_prt_df[bnk_prt_df['isFlaggedFraud']==0]['isFraud'].value_counts()


# In[ ]:


bnk_prt_df[np.logical_and(bnk_prt_df['isFlaggedFraud']==0, bnk_prt_df['isFraud']==1)]['type'].value_counts()


# <h3>Analysis of step vs isFraud in the dataset</h3>

# In[ ]:


sns.countplot(x='step',data=bnk_prt_df)


# In[ ]:


bnk_prt_df['step'].value_counts()


# * <h3>Analysis of amount vs isFraud</h3>

# In[ ]:


## Fraudulent transactions have happened when the amount is less. It is evident from the graph below
sns.jointplot('amount','isFraud',data=bnk_prt_df)


# * <h3>Analysis of oldbalanceOrg vs isFraud</h3>

# In[ ]:


## fraudulent transactions are more when oldBalanceOrg of the origin from where the account transfer happened is higher than
## when it is less
sns.jointplot(x='oldbalanceOrg',y='isFraud',data=bnk_prt_df)


# <h3>Analysis of newBalanceOrig vs isFraud in the dataset</h3>

# In[ ]:


## fraudulent transactions are more when newBalanceOrg of the origin from where the account transfer happened is higher than
## when it is less
sns.jointplot(x='newbalanceOrig',y='isFraud',data=bnk_prt_df)


# <h3>Analysis of oldbalanceDest Vs isFraud in the dataset</h3>

# In[ ]:


## fraudulent transactions are more when oldBalanceDest of the destination to where the account transfer happened is 
## higher than when it is less
sns.jointplot(x='oldbalanceDest',y='isFraud',data=bnk_prt_df)


# In[ ]:


## fraudulent transactions are more when newbalanceDest of the destination to where the account transfer happened is 
## higher than when it is less
sns.jointplot(x='newbalanceDest',y='isFraud',data=bnk_prt_df)


# In[ ]:


##One hot encoding for type column so that the values are captured

bnk_prt_df['CASH_IN'] = pd.get_dummies(bnk_prt_df['type'])['CASH_IN']
bnk_prt_df['CASH_OUT'] = pd.get_dummies(bnk_prt_df['type'])['CASH_OUT']
bnk_prt_df['DEBIT'] = pd.get_dummies(bnk_prt_df['type'])['DEBIT']
bnk_prt_df['PAYMENT'] = pd.get_dummies(bnk_prt_df['type'])['PAYMENT']
bnk_prt_df['TRANSFER'] = pd.get_dummies(bnk_prt_df['type'])['TRANSFER']
bnk_prt_df.drop('CASH_IN',axis=True,inplace=True)
bnk_prt_df.info()


# In[ ]:


## multivariate analysis
## As seen from correlation matrix values, doesn't seem that isFraud is related to any variables here

bnk_prt_df.corr()


# In[ ]:


X = bnk_prt_df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud','CASH_OUT','DEBIT','PAYMENT','TRANSFER']]
Y = bnk_prt_df['isFraud']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


lr = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC()
nb = GaussianNB()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
bc = BaggingClassifier()
abc = AdaBoostClassifier()
gbc = GradientBoostingClassifier()


# In[ ]:


def trainFitTest(model):
    ##models = [lr,knn,svc,nb,dtc,rfc,bc,abc,gbc]
    score = []
    ##for model in models:
    model.fit(X_train,Y_train)
    print(' score - ',model.score(X_test,Y_test))
    score.append(model.score(X_test,Y_test))
    Y_rfc_pred = model.predict(X_test)
    print(classification_report(Y_test,Y_rfc_pred))
    print(confusion_matrix(Y_test,Y_rfc_pred))


# In[ ]:


trainFitTest(lr)


# In[ ]:


##trainFitTest(knn)


# In[ ]:


trainFitTest(dtc)


# In[ ]:


trainFitTest(nb)


# In[ ]:


##trainFitTest(svc)


# In[ ]:


trainFitTest(rfc)


# In[ ]:


trainFitTest(bc)


# In[ ]:


trainFitTest(abc)


# In[ ]:


trainFitTest(gbc)


# In[ ]:




