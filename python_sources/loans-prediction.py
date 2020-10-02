#!/usr/bin/env python
# coding: utf-8

# This project is done by Hoummada Outman and Mohamed El-eliem, we are engineering students at the National Institut of Statistic and Applied Economy in Rabat Morocco, this project is still on work, comment below for any suggestion, Thanks.... 

# In[ ]:


import numpy as np
import pandas as pd
train_df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
train_df.head()


# In[ ]:


train_df=train_df.drop("Loan_ID", axis=1)


# In[ ]:


train_df['Loan_Status']=pd.get_dummies(train_df['Loan_Status'])


# In[ ]:


train_df.info()


# In[ ]:


train_df["Credit_History"] = train_df["Credit_History"].astype(object)


# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df['LoanAmount']=train_df['LoanAmount'].fillna(train_df['LoanAmount'].mean())
train_df['Loan_Amount_Term']=train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mean())


# In[ ]:


train_df.shape


# In[ ]:


train_df=train_df.dropna()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=train_df,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# In[ ]:


cat_df = train_df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History','Loan_Status']]
num_df = train_df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Loan_Status']]


# In[ ]:


cat_df.head()


# In[ ]:


num_df.head()


# In[ ]:


num_df.describe()


# In[ ]:


num_df.corr(method = 'spearman')


# In[ ]:


dummy_variable_1 = pd.get_dummies(cat_df['Dependents'])
dummy_variable_1.head()


# In[ ]:


import seaborn as sns

corr = train_df.corr(method = 'spearman')

sns.heatmap(corr, annot = True)

plt.show()


# In[ ]:


from sklearn import tree
X=train_df[[]]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

