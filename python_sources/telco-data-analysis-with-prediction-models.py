#!/usr/bin/env python
# coding: utf-8

# Conclusions of this dataset:
# - Most contracts are monthly
# - Most contracts are paid via electronic check
# - Just as the highest average monthly payments are paid via electronic checks
# - However, the highest total amount paid per customer has the highest average bank transfer
# - The sexes of clients are divided equally
# - Most clients don't have children and are men
# - 36% of contracts churned, of wich:
#     - 88% of these monthly
#     - 8.9% per year
#     - 2,6% multiannual
# - Of churn, the majority of customers:
#     - Have Phone service 
#     - Have optic fiber internet 
#     - Have No TV Streaming
#     - Pays via electronic check
#     - Are not senior.
#     - Don't dependents
#     - Pay invoices of up to 1500
# 
# Trying to predict churns was quite efective, although I've reached only 79% of precision. For this type of analysis I think it's a good precision rate.
# 
# The used algorithm with the best precision was Naive Bayes, the precision percentages goes below:
# 
# - Random Forest: 73%.
# - Naive Bayes: 79%.
# - Logistic regression: 76%.
# - KNN: 74%.
# - SVC: 76%.
# 
# Code below:

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['Contract'].value_counts().plot(kind='bar')
df['Contract'].value_counts()


# In[ ]:


df['PaymentMethod'].value_counts().plot(kind='bar')
df['PaymentMethod'].value_counts()


# In[ ]:


df.groupby(by='PaymentMethod')['MonthlyCharges'].mean().plot(kind='bar')


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df['TotalCharges'] = df['TotalCharges'].convert_objects(convert_numeric=True)


# In[ ]:


df.groupby(by='PaymentMethod')['TotalCharges'].mean().plot(kind='bar')
df.groupby(by='PaymentMethod')['TotalCharges'].mean()


# In[ ]:


pd.set_option('display.max_columns', 100)


# In[ ]:


plt.figure(figsize=(12,16))
plt.subplot(4, 2, 1)
sns.countplot(x='gender', data=df, hue='Partner')
plt.subplot(4, 2, 2)
sns.countplot(x='gender', data=df, hue='Dependents')
plt.subplot(4, 2, 3)
sns.countplot(x='gender', data=df, hue='SeniorCitizen')
plt.subplot(4, 2, 4)
sns.countplot(x='gender', data=df)


# In[ ]:


df.groupby(by='Contract')['Churn'].value_counts().to_frame()


# Percentage of churns with month-to-month contract.

# In[ ]:


df[(df['Churn']=='Yes')&(df['Contract']=='Month-to-month')]['Churn'].count()/df[df['Churn']=='Yes']['Churn'].count()*100


# Percentage of churns with one year contract.

# In[ ]:


df[(df['Churn']=='Yes')&(df['Contract']=='One year')]['Churn'].count()/df[df['Churn']=='Yes']['Churn'].count()*100


# Percentage of churns with two year contract.

# In[ ]:


df[(df['Churn']=='Yes')&(df['Contract']=='Two year')]['Churn'].count()/df[df['Churn']=='Yes']['Churn'].count()*100


# In[ ]:


plt.figure(figsize=(22,34))
plt.subplot(6, 2, 1)
sns.countplot(x='Churn', data=df, hue='PhoneService')
plt.subplot(6, 2, 2)
sns.countplot(x='Churn', data=df, hue='InternetService')
plt.subplot(6, 2, 3)
sns.countplot(x='Churn', data=df, hue='StreamingTV')
plt.subplot(6, 2, 4)
sns.countplot(x='Churn', data=df, hue='PaymentMethod')
plt.rcParams['legend.fontsize'] = 10
plt.subplot(6, 2, 5)
sns.countplot(x='Churn', data=df, hue='SeniorCitizen')
plt.subplot(6, 2, 6)
sns.countplot(x='Churn', data=df, hue='Dependents')


# In[ ]:


sns.barplot(x='TotalCharges', y='Churn', data=df)


# In[ ]:


df['MonthlyCharges'].describe()


# Creating a DF with data I'll use to try to predict churn.

# In[ ]:


df_cat = df.drop(columns=['customerID','MonthlyCharges','TotalCharges', 'tenure'])


# In[ ]:


df_cat.columns.values.tolist()


# In[ ]:


df_cat.head()


# Calling LabelEnconder librarie to encode categorical data.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


df_cat = df_cat.apply(LabelEncoder().fit_transform)


# In[ ]:


df_cat.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_cat.drop(columns=['Churn'])
y = df_cat['Churn']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=101, test_size=0.30)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


predrf = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, predrf))


# In[ ]:


rf.score(X_test,y_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


prob = gnb.fit(X_train, y_train)


# In[ ]:


prednb = gnb.predict(X_test)


# In[ ]:


print(classification_report(y_test, prednb))


# In[ ]:


gnb.score(X_test,y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


predlr = lr.predict(X_test)


# In[ ]:


print(classification_report(y_test, predlr))


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


kn = KNeighborsClassifier()


# In[ ]:


kn.fit(X_train,y_train)


# In[ ]:


predkn = kn.predict(X_test)


# In[ ]:


kn.score(X_test,y_test)


# In[ ]:


print(classification_report(y_test, predkn))


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


sv = SVC()


# In[ ]:


sv.fit(X_train, y_train)


# In[ ]:


predsv = sv.predict(X_test)


# In[ ]:


print(classification_report(y_test, predsv))


# In[ ]:


sv.score(X_test,y_test)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(14,16))
plt.subplot(4,2,1)
sns.distplot(df['tenure'], kde=False, bins=70)
plt.subplot(4,2,2)
sns.distplot(df['MonthlyCharges'], kde=False, bins=70)
plt.subplot(4,2,3)
sns.distplot(df['TotalCharges'], kde=False, bins=70)


# In[ ]:


df['TotalCharges'].max()-df['TotalCharges'].min()


# In[ ]:




