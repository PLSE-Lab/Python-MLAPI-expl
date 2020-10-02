#!/usr/bin/env python
# coding: utf-8

# > This project explores publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). This project aims to create a model to show profiles of borrowers with a high probability of paying back this loan. 
# > The columns in the dataset represent the following:
# > * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# > * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# > * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# > * installment: The monthly installments owed by the borrower if the loan is funded.
# > * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# > * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# > * fico: The FICO credit score of the borrower.
# > * days.with.cr.line: The number of days the borrower has had a credit line.
# > * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# > * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# > * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# > * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# > * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


loans = pd.read_csv("../input/loan_data.csv")
loans.head()


# In[ ]:


loans.info()


# In[ ]:


loans.describe()


# In[ ]:


sns.set_style('darkgrid')
plt.hist(loans['fico'].loc[loans['credit.policy']==1], bins=30, label='Credit.Policy=1')
plt.hist(loans['fico'].loc[loans['credit.policy']==0], bins=30, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, color='green', label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:



#creating a countplot to see the counts of purpose of loans by not.fully.paid
plt.figure(figsize=(12,6))
sns.countplot(data=loans, x='purpose', hue='not.fully.paid')


# In[ ]:


#checking the trend between FICO and the interest rate
plt.figure(figsize=(10,6))
sns.jointplot(x='fico', y='int.rate', data=loans)


# In[ ]:


#understanding the relationship between credit.policy and not.fully.paid
sns.lmplot(data=loans, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set2')


# In[ ]:


loans.head()


# In[ ]:


#handling categorical variable purpose
purpose_c = pd.get_dummies(loans['purpose'], drop_first=True)
loans_f = pd.concat([loans, purpose_c], axis=1).drop('purpose', axis=1)
loans_f.head()


# In[ ]:


#checking for null values
sns.heatmap(loans.isnull())


# In[ ]:


#Splitting the dataset into test and train set
from sklearn.model_selection import train_test_split
y = loans_f['not.fully.paid'] 
X = loans_f.drop('not.fully.paid', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# In[ ]:


#using decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
prediction = dtree.predict(X_test)

#checking performance of the model
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))


# In[ ]:


#using random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train, y_train)
predictionRF = rfc.predict(X_test)

#checking performance of the model
print(confusion_matrix(y_test, predictionRF))
print(classification_report(y_test, predictionRF))


# Conclusion:
# If precision (TP/(TP+FP)) or the total number of correct classifications is considered to be the determining factor, then decision tree algorithm did as well as random forests for class 0.
# If recall (TP/(TP+FN)) or the total number of true positives is considered to be the determining factor, then random forests did better than decision trees for class 0.
# For class 1, random forests gave recall of only 0.01, whereas for decision tree it was 0.24. Neither of the models did too well for class 1. However, decision trees did slightly better for class 1.
