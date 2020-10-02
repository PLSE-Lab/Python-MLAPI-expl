#!/usr/bin/env python
# coding: utf-8

# 
# ___
# # LendingClub Project Using Descison Tree and Random Forest
# 
# Lending Club website Estimation of customers whether they will pay th loan or not
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Importing Libraries
# 

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Getting Data

# In[6]:


loans = pd.read_csv('../input/loan_data.csv')


# In[7]:


loans.info()


# In[ ]:


loans.head()


# In[ ]:





# # Exploratory Data Analysis
# 

# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Paid =1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Not Paod=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(11,7))
sns.countplot(x=loans['purpose'] , hue=loans['not.fully.paid'], palette='Set1')


# In[ ]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[ ]:


sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy' , col='not.fully.paid')


# In[ ]:


loans.info()


# ## Converting Catagorical Features into Dummies for Sklearn

# In[ ]:


cat_feats = ['purpose']


# 

# In[ ]:


final_data = pd.get_dummies(loans,columns=cat_feats , drop_first=True)


# In[ ]:


final_data.head()


# ## Train Test Split
# 

# In[ ]:


X = final_data.drop('not.fully.paid' , axis=1)
y = final_data['not.fully.paid']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.3 , random_state=101)


# ## Training Our Data into Descision Tree Model
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# 

# In[ ]:


y_pre = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pre))


# In[ ]:


print(confusion_matrix(y_test,y_pre))


# ## Training Our Data into Random Forest model
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=900)


# In[ ]:


rfc.fit(X_train,y_train)


# ## Predictions and Evaluation
# 

# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))

