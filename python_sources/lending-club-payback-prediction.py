#!/usr/bin/env python
# coding: utf-8

# # Lending Club Payback Prediction

# ## Problem Overview
# 
# Lending Club is a lending company that connects people who need money (borrowers) with people who have money (investors). As an investor, anyone would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# The problem at hand is to predict the people who have high probablity of paying back the money as comapred to those who dont. Since it is a classification problem, a good way to approach it is by Decision Trees and Random Forest algorithm

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset Details
# 
# The dataset consists of the following columns:
# 
# - credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise
# - purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other")
# - int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates
# - installment: The monthly installments owed by the borrower if the loan is funded
# - log.annual.inc: The natural log of the self-reported annual income of the borrower
# - dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income)
# - fico: The FICO credit score of the borrower
# - days.with.cr.line: The number of days the borrower has had a credit line
# - revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle)
# - revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available)
# - inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months
# - delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years
# - pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments)
# - no.fully.paid: This is the column that we are trying to predict. We are trying to predict whether or not the loan was fully paid back or not

# In[ ]:


loans = pd.read_csv('../input/lending-club-data/loan_data.csv')


# In[ ]:


loans.head()


# In[ ]:


loans.head().info()


# In[ ]:


loans.head().shape


# In[ ]:


loans.head().describe()


# ## Exploratory Data Analysis

# #### Creating a histogram to analyse the credit policy outcome for each borrower on the basis of the FICO score

# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit Policy=0')
plt.legend()
plt.xlabel('FICO')


# The above graph shows that people with a good FICO score matched more to the lending club credit policy as compared to the people with lesser credit scores

# #### Creating a histogram to analyse the not fully paid column for each borrower on the basis of the FICO score

# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Not Fully Paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Not Fully Paid=0')
plt.legend()
plt.xlabel('FICO')


# The above graph shows that ratio of people having paid back the money is better with people having a good FICO score as compared to the peopl having not a good FICO score

# #### Creating a countplot to visualize the counts of loan purpose on the basis of full payments made or not (not.fully.paid)

# In[ ]:


plt.figure(figsize=(11,7))
sns.countplot('purpose',hue='not.fully.paid',data=loans,palette='Set1')


# The above graph shows that the ratio of the people who paid back the money fully as to those who did not is almost the same in every loan purpose category

# #### Creating a jointplot to visualize the trend between FICO score and interest rate

# In[ ]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# The above graph shows that the as the FICO score goes on increasing, the rate of interest for the borrowers go on decreasing because FICO score gives the investors a sense of confidence that the loan will be repaid by the borrowers

# #### Creating a lmplot to visualize the trend between FICO score and interest rate on the basis of credit policy

# In[ ]:


plt.figure(figsize=(11,7))
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')


# ## Coverting Categorical Features
# 
# The purpose column in the loans dataset has categorical values as it tells about the purpose for the loan. So the column needs to be converted to dummy variable using pandas so that purpose can be used as an input in our machine learning model

# In[ ]:


loans.info()


# In[ ]:


loan_purpose=['purpose']


# In[ ]:


final_data=pd.get_dummies(loans,columns=loan_purpose,drop_first=True)


# In[ ]:


# In the above code, drop_first is done to avoid multi-colinearity


# In[ ]:


final_data.info()


# In[ ]:


final_data.head()


# ## Dividing the Data into Features and Labels
# 
# We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


X = final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']


# ## Train Test Split
# 
# - Once the features and the labels are decided, the data is to be divided into training data and testing data
# - The model will be trained on the training set and then the test set will be used to evaluate the model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Decision Tree

# ### Creating and Training a Decision Tree Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Instantiating Decision Tree model (basically creating a decision tree object)


# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


# Training or fitting the model on training data


# In[ ]:


dtree.fit(X_train,y_train)


# ### Predictions

# In[ ]:


dtree_predictions = dtree.predict(X_test)


# ### Decision Tree Model Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(y_test,dtree_predictions))


# In[ ]:


print(confusion_matrix(y_test,dtree_predictions))


# ## Random Forest

# ### Creating and Training a Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Instantiating Random Forest model (basically creating a random forest object)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=300)


# In[ ]:


# Training or fitting the model on training data


# In[ ]:


rfc.fit(X_train,y_train)


# ### Predictions

# In[ ]:


rfc_predictions = rfc.predict(X_test)


# ## Random Forest Model Evaluation

# In[ ]:


print(classification_report(y_test,rfc_predictions))


# In[ ]:


print(confusion_matrix(y_test,rfc_predictions))


# ## Conclusion
# 
# The Random Forest Model performed slightly better than the Decision Tree Model
