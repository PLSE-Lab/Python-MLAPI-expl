#!/usr/bin/env python
# coding: utf-8

# In[93]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[94]:


train = pd.read_csv("../input/train.csv")


# In[95]:


test = pd.read_csv("../input/test.csv")


# In[96]:


train.head()


# In[97]:


test.head()


# In[98]:


train.info()


# In[99]:


test.isnull().sum()


# In[100]:


train.drop('Education', axis=1, inplace=True)
test.drop('Education', axis=1, inplace=True)


# In[101]:


train.head(30)


# In[102]:


train.drop('Self_Employed', axis=1, inplace=True)
test.drop('Self_Employed', axis=1, inplace=True)


# In[103]:


train.head(30)


# In[104]:


train.Gender=pd.get_dummies(train.Gender)
train.Married=pd.get_dummies(train.Married)


# In[105]:


train.head(30)


# In[106]:


train=pd.get_dummies(train,columns=['Property_Area'])


# In[107]:


train.head(30)


# In[108]:


test.Gender=pd.get_dummies(test.Gender)
test.Married=pd.get_dummies(test.Married)
test=pd.get_dummies(test,columns=['Property_Area'])


# In[109]:


test.head()


# In[110]:


train.isnull().sum()


# In[111]:


test.isnull().sum()


# In[112]:


train.head()


# In[113]:


train.drop('Property_Area_Rural', axis=1, inplace=True)
test.drop('Property_Area_Rural', axis=1, inplace=True)


# In[114]:


train["LoanAmount"].fillna(train["LoanAmount"].mean(), inplace=True)


# In[115]:


train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mean(), inplace=True)
train["Credit_History"].fillna(train["Credit_History"].mean(), inplace=True)


# In[116]:


train.head()


# In[117]:


train.isnull().sum()


# In[118]:


train['Dependents'].value_counts()


# In[120]:


train.Dependents=train.Dependents.replace(["3+"],["4"])


# In[121]:


train['Dependents'].value_counts()


# In[122]:


train.isnull().sum()


# In[123]:


train["Dependents"].fillna(train["Dependents"].median(), inplace=True)


# In[124]:


train.isnull().sum()


# In[125]:


test.head(30)


# In[126]:


test["LoanAmount"].fillna(test["LoanAmount"].mean(), inplace=True)
test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].mean(), inplace=True)
test["Credit_History"].fillna(test["Credit_History"].mean(), inplace=True)
test['Dependents'].value_counts()
test.Dependents=test.Dependents.replace(["3+"],["4"])
test["Dependents"].fillna(test["Dependents"].median(), inplace=True)


# In[127]:


test.isnull().sum()


# In[128]:


test.head()


# In[129]:


train.head()


# In[130]:


train.drop('Loan_ID', axis=1, inplace=True)
test.drop('Loan_ID', axis=1, inplace=True)
    


# In[131]:


target = train['Loan_Status']
train.drop('Loan_Status', axis=1, inplace=True)

train.shape, target.shape


# In[132]:


train.head()


# In[133]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[134]:


clf = KNeighborsClassifier(n_neighbors = 10)
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[135]:


import numpy as np


# In[136]:


round(np.mean(score)*100, 2)


# In[137]:


from sklearn.tree import DecisionTreeClassifier


# In[138]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[140]:


round(np.mean(score)*100, 2)


# In[141]:


from sklearn.ensemble import RandomForestClassifier


# In[142]:


clf = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[143]:


round(np.mean(score)*100, 2)


# In[ ]:


#so basically i am partially correct using 

