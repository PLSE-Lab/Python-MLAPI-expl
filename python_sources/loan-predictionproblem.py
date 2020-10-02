#!/usr/bin/env python
# coding: utf-8

# -----------------------------------------------CLASSIFICATION PROBLEM---------------------------------------------

# This is a binary classification problem where we need to predict whether or not to approve a loan based on the past information of the person.

# Importing Important Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Loading dataset

# In[ ]:


data=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')


# In[ ]:


df=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')


# In[ ]:


data.head()


# In[ ]:


#info about the Dataset
data.info()


# In[ ]:


#Shape of the dataset
data.shape


# There are 614 rows and 13 columns in the Dataset.

# In[ ]:


#Description about the Dataset
data.describe()


# Here we can observe that the maximum value of ApplicantIncome and CoapplicantIncome is much higher than the 75%of the data of ApplicantIncome and CoapplicantIncome respectively.

# In[ ]:


#Datatypes of the features
data.dtypes


# Continous variables are:  CoapplicantIncome, ApplicantIncome,  LoanAmount,  Loan_Amount_Term,  Credit_History.
# Categorical Variables are:   Loan_ID, Gender,  Married,  Dependents,  Eduaction,  Self_Employed,  Property_area,  Loan_Status.

# In[ ]:


#Checking for missing values
data.isnull().sum()


# There are missing values in the following features:-             'Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'.

# There are 22 missing values in the LoanAmount variable so we fill the missing values by the mean of all values in that variable.

# In[ ]:


data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean())


# There are 50 missing values in the credit History. we fill those missing values by the median of Credit History.

# In[ ]:


data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].median())


# In[ ]:


data.isnull().sum()


# We drop all other missing values in the variables.

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.shape


# Now,the new shape of the dataset is (542,13).

# **UNIVARIATE ANALYSIS**

# In[ ]:


df['Gender'].value_counts()


# Out of 542, there are 444 males and 98 females applying for loan.

# In[ ]:


df['Married'].value_counts()


# We observe there are more number of married people applying for loan than unmarried.

# In[ ]:


df['Education'].value_counts()


# More than half of the total applying for loan are graduates.

# In[ ]:


df['Self_Employed'].value_counts()


# People who are self employed tend not to apply for loan.

# In[ ]:


df['Property_Area'].value_counts()


# There is no much difference among the locality of people applying for loan. Almost people from all areas apply for a loan.

# In[ ]:


plt.boxplot(df['ApplicantIncome'])


# In[ ]:


plt.boxplot(df['CoapplicantIncome'])


# In[ ]:


plt.boxplot(df['LoanAmount'])


# In[ ]:


plt.boxplot(df['Loan_Amount_Term'])


# In[ ]:


plt.boxplot(df['Credit_History'])


# **BIVARIATE ANALYSIS**

# In[ ]:


sns.scatterplot(x='Property_Area',y='Loan_Status',data=df)


# In[ ]:


sns.scatterplot(x='Self_Employed',y='Loan_Status',data=df)


# In[ ]:


print(pd.crosstab(df['Property_Area'],df['Loan_Status']))


# In[ ]:


sns.countplot(df['Property_Area'],hue=df['Loan_Status'])


# Out of all semiurban people who apply for a loan more than half of them got a loan.
# So this seems to a useful feature.

# In[ ]:


print(pd.crosstab(df['Gender'],df['Loan_Status']))


# In[ ]:


sns.countplot(df['Gender'],hue=df['Loan_Status'])


# Males have high chances of getting a loan.

# In[ ]:


print(pd.crosstab(df['Married'],df['Loan_Status']))


# In[ ]:


sns.countplot(df['Married'],hue=df['Loan_Status'])


# Out of all married people who applied for a loan,maximum of them get their loan approved.

# In[ ]:


print(pd.crosstab(df['Self_Employed'],df['Loan_Status']))


# In[ ]:


sns.countplot(df['Self_Employed'],hue=df['Loan_Status'])


# In[ ]:


print(pd.crosstab(df['Education'],df['Loan_Status']))


# In[ ]:


sns.countplot(df['Education'],hue=df['Loan_Status'])


# Half of the graduates who applied for loan got their loan approved.

# Converting dependent categorical variable to continous variable.

# In[ ]:


df['Loan_Status'].replace('N',0,inplace=True)
df['Loan_Status'].replace('Y',1,inplace=True)


# In[ ]:


plt.title('Correlation Matrix')
sns.heatmap(df.corr(),annot=True)


# As we can observe from the above correlation matrix that the dependent variable loan_status is dependent only on Credit_History.
# So we keep this independent variable and discard all other variables which are not related with the loan_status.

# In[ ]:


df2=df.drop(labels=['ApplicantIncome'],axis=1)


# In[ ]:


df2=df2.drop(labels=['CoapplicantIncome'],axis=1)


# In[ ]:


df2=df2.drop(labels=['LoanAmount'],axis=1)


# In[ ]:


df2=df2.drop(labels=['Loan_Amount_Term'],axis=1)


# In[ ]:


df2=df2.drop(labels=['Loan_ID'],axis=1)


# In[ ]:


df2.head()


# Changing categorical variables to continous variables.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()


# In[ ]:


df2['Property_Area']=le.fit_transform(df2['Property_Area'])


# In[ ]:


df2['Dependents']=le.fit_transform(df2['Dependents'])


# In[ ]:


df2=pd.get_dummies(df2)


# In[ ]:


df2.dtypes


# Now we can see each and every variable is changed to continous variable.

# In[ ]:


df2=df2.drop(labels=['Gender_Female'],axis=1)


# In[ ]:


df2=df2.drop(labels=['Married_No'],axis=1)


# In[ ]:


df2=df2.drop(labels=['Education_Not Graduate'],axis=1)


# In[ ]:


df2=df2.drop(labels=['Self_Employed_No'],axis=1)


# In[ ]:


df2.head()


# In[ ]:


plt.title('Correlation Matrix')
sns.heatmap(df2.corr(),annot=True)


# Loan status is least correlated with self_employed_yes,education_graduate,dependents.

# In[ ]:


df2=df2.drop('Self_Employed_Yes',1)


# In[ ]:


df2=df2.drop('Dependents',1)


# In[ ]:


df2=df2.drop('Education_Graduate',1)


# In[ ]:


X=df2.drop('Loan_Status',1)


# In[ ]:


Y=df2['Loan_Status']


# Splitting train and test dataset.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=6)


# In[ ]:


print('Shape of X_train is: ',x_train.shape)
print('Shape of X_test is: ',x_test.shape)
print('Shape of Y_train is: ',y_train.shape)
print('Shape of y_test is: ',y_test.shape)


# ---**LOGISTIC REGRESSION**---

# In[ ]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()


# In[ ]:


log.fit(x_train,y_train)


# In[ ]:


log.score(x_train,y_train)


# In[ ]:


#Predicting trest dataset
pred=log.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,pred)


# This seems to be a good accuracy score.

# In[ ]:


from sklearn import metrics


# Confusion matrix

# In[ ]:


metrics.confusion_matrix(y_test,pred)


# From confusion matrix we can observe that
# 
# * True positive value= 23
# * True negative value=1
# * False negative=16
# * False positive=69

# In[ ]:


metrics.recall_score(y_test,pred)


# In[ ]:


metrics.precision_score(y_test,pred)


# In[ ]:


metrics.f1_score(y_test,pred)


# In[ ]:


data={'y_test':y_test,'pred':pred}
pd.DataFrame(data=data)


# -----**DECISION TREE**------

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()


# In[ ]:


clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_train,y_train)


# In[ ]:


pred1=clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,pred1)


# In[ ]:


metrics.confusion_matrix(y_test,pred1)


# From confusion matrix we can observe that
# * True positive value= 23
# * True negative value=1
# * False negative=16
# * False positive=69

# In[ ]:


metrics.f1_score(y_test,pred1)


# In[ ]:


metrics.recall_score(y_test,pred1)


# In[ ]:


metrics.precision_score(y_test,pred1)

