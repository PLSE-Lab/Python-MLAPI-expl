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


df=pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['LoanAmount'].hist(bins=50)


# In[ ]:





# In[ ]:


df.boxplot(column='ApplicantIncome')


# In[ ]:





# In[ ]:


df['LoanAmount'].hist(bins=50)


# In[ ]:


df.boxplot(column='LoanAmount')


# In[ ]:


df['Property_Area'].value_counts()


# In[ ]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)
print ('\nProbility of getting loan for each Credit History class:')
print (temp2)




# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# In[ ]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)



# In[ ]:


df.isnull().sum(0)


# In[ ]:


df['Gender'].value_counts()


# In[ ]:


df['Gender'].fillna('Male', inplace=True)


# In[ ]:


df['Credit_History'].value_counts()


# In[ ]:


df['Credit_History'].fillna(1,inplace=True)


# In[ ]:


df['Self_Employed'].value_counts()


# In[ ]:


df['Self_Employed'].fillna('No',inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Dependents'].value_counts()


# In[ ]:


df['Dependents'].fillna(0,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Married'].value_counts()


# In[ ]:


df['Married'].fillna('Yes',inplace=True)


# In[ ]:


X=df[['Credit_History','Gender','Married','Education']]
X=pd.get_dummies(X)
y=df['Loan_Status']


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X,y)
pdt=model.predict(X)


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(pdt,y))


# In[ ]:


df.head()


# In[ ]:


import sklearn.model_selection as ms
import sklearn.tree as tree
clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
mod=ms.GridSearchCV(clf,param_grid={'max_depth':[4]})
mod.fit(X,y)


# In[ ]:


mod.best_estimator_


# In[ ]:


mod.best_score_

