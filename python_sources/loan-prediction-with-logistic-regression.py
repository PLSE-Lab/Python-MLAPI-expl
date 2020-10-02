#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# ### Reading the csv file

# In[46]:


data = pd.read_csv('../input/train_loan.csv')


# ### First 5 samples of the dataset

# In[47]:


data.head()


# ### Checking the shape of the dataset

# In[48]:


data.shape


# We have 614 rows and 13 columns in our training dataset. 

# ### Summary of the dataset

# In[49]:


data.info()


# ### Checking for null values

# In[50]:


data.isnull().sum()


# ### Filling Missing values (categorical variables with mode value and numerical variables with the mean/meian value)

# In[51]:


data['LoanAmount'].fillna(data.LoanAmount.median(), inplace = True)
data['Loan_Amount_Term'].fillna(data.Loan_Amount_Term.mode()[0], inplace = True)
data['Gender'].fillna(data.Gender.mode()[0], inplace = True)
data['Married'].fillna(data.Married.mode()[0], inplace = True)
data['Dependents'].fillna(data.Dependents.mode()[0], inplace = True)
data['Self_Employed'].fillna(data.Self_Employed.mode()[0], inplace = True)
data['Credit_History'].fillna(data.Credit_History.mode()[0], inplace = True)


# ### Checking the datatypes in the dataset

# In[52]:


data.dtypes


# There are three different datatypes in our dataset.
# 
# object - consists of categorical features/variables. Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status in the dataset belong to this category.
# 
# int64 - variable/feature with integer value. ApplicantIncome belongs to this category.
# 
# float64 - variable/integer with a decimal value. In the dataset variables/features belonging to this category are: CoApplicantIncome, LoanAmount, Loan_Amount_term, and Credit_History.

# ### Converting the Applicant Income datatype to a float

# In[53]:


data['ApplicantIncome'] = data['ApplicantIncome'].astype('float64')
data.dtypes


# ### Visualizing categorical features

# 1. Target variable (Loan_Status)

# In[54]:


data.Loan_Status.value_counts()


# In[55]:


data.Loan_Status.value_counts(normalize = True).plot(kind = 'bar').grid(True, axis = 'y')


# We can see that out of 614 applicants, only 422(about 69%) applicants get their loan approved while the loan request of the remaining people (192) get rejected.

# 2. Other categorical features

# In[56]:


plt.figure(figsize = (15,15))

plt.subplot(3,2,1)
data.Gender.value_counts(normalize = True).plot.bar(title = 'Gender').grid(True, axis = 'y')
plt.xticks(rotation = 45)

plt.subplot(3,2,2)
data.Married.value_counts(normalize = True).plot.bar(title = 'Married').grid(True, axis = 'y')
plt.xticks(rotation = 45)

plt.subplot(3,2,3)
data.Education.value_counts(normalize = True).plot.bar(title = 'Education').grid(True, axis = 'y')
plt.xticks(rotation = 30)

plt.subplot(3,2,4)
data.Dependents.value_counts(normalize = True).plot.bar(title= 'Dependants').grid(True, axis = 'y')
plt.xticks(rotation = 45)

plt.subplot(3,2,5)
data.Self_Employed.value_counts(normalize = True).plot.bar(title = 'Self_Employed').grid(True, axis = 'y')
plt.xticks(rotation = 45)

plt.subplot(3,2,6)
data.Property_Area.value_counts(normalize = True).plot.bar(title = 'Property-Area').grid(True, axis = 'y')
plt.xticks(rotation = 45)


# From the above graphs we can say that-
# 1. Over 80% of the applicants were male.
# 2. About 65% of the total applicants were married.
# 3. Around 78% of the applicants were graduate.
# 4. About 85% of the applicants were not self-employed.
# 5. About 35% of the applicants were from semi-urban area.
# 6. Most of the applicants have no dependants.

# ### Visualizing numerical features

# In[57]:


plt.figure(figsize = (15,10))
plt.subplot(231)
sns.boxplot(y= data.ApplicantIncome)

plt.subplot(232)
sns.boxplot(y= data.CoapplicantIncome)

plt.subplot(233)
sns.boxplot(y= data.LoanAmount)

plt.subplot(234)
sns.distplot(data.ApplicantIncome)

plt.subplot(235)
sns.distplot(data.CoapplicantIncome)

plt.subplot(236)
sns.distplot(data.LoanAmount)


# We can see that there are many outliers present for the applicant income, coapplicant income and loan amount

# In[58]:


data.Credit_History.value_counts(normalize = True).plot.bar(title = 'Credit History', figsize = (7,5)).grid(True, axis = 'y')


# About 85% of the applicants have their previous debts repayed.

# In[59]:


pd.crosstab(data.Gender, data.Loan_Status)


# In[60]:


pd.crosstab(data.Gender, data.Loan_Status).plot.bar(figsize = (5,5))


# In[61]:


pd.crosstab(data.Married, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))

pd.crosstab(data.Dependents, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))

pd.crosstab(data.Education, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))

pd.crosstab(data.Self_Employed, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))


# Following types of applicants have a higher chance of getting their loan approved-
# 1. Married applicants
# 2. Applicants with zero dependants or 3+
# 3. Graduate applicants
# 

# ### Numerical variables and target variable

# In[62]:


sns.boxplot(y= 'ApplicantIncome', x= 'Loan_Status', data = data)


# In[63]:


bins = [0,2500,4000,6000,81000]
group = ['low', 'average', 'high', 'very high']
data['ApplicantIncome new'] = pd.cut(data['ApplicantIncome'], bins, labels = group)
pd.crosstab(data['ApplicantIncome new'], data['Loan_Status'], normalize = True).plot.bar(figsize = (5,5), stacked = True)


# In[64]:


sns.boxplot(y= 'CoapplicantIncome', x= 'Loan_Status', data = data)


# In[65]:


bins = [0,1000,2000,4000,42000]
group = ['low', 'average', 'high', 'very high']
data['CoapplicantIncome new'] = pd.cut(data['CoapplicantIncome'], bins, labels = group)
pd.crosstab(data['CoapplicantIncome new'], data['Loan_Status'], normalize = True).plot.bar(figsize = (5,5), stacked = True)


# In[66]:


data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
sns.boxplot(y= 'Total_Income', x= 'Loan_Status', data = data)


# In[67]:


bins = [0,2500,5000,10000,81000]
groups = ['low', 'average', 'high', 'very high']
data['Total_Income_new'] = pd.cut(data['Total_Income'], bins, labels = group)
pd.crosstab(data.Total_Income_new, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))


# We can say that applicants with average to high total income has better chances to get their loan approved as compare to low income applicants.

# In[68]:


pd.crosstab(data.Credit_History, data.Loan_Status).plot.bar(stacked = True, figsize = (5,5))


# Applicants who are not able to pay their previous debts does not get their loan approved.

# In[69]:


bin = [0,100,200,700]
group = ['low', 'average', 'high']
data['loanamount'] = pd.cut(data['LoanAmount'], bin, labels = group)
pd.crosstab(data.loanamount, data.Loan_Status, normalize = True).plot.bar(stacked = True, figsize = (5,5))


# In[70]:


data['Loan_Status'] = data['Loan_Status'].map({'N': 0, 'Y': 1})
data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})


# In[71]:


train = data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']]
sns.heatmap(train.corr(), annot = True, cmap="BuPu")


# In[72]:


X = data.drop(['Loan_ID', 'Loan_Status', 'ApplicantIncome new', 'CoapplicantIncome new',
               'Total_Income_new', 'loanamount', 'ApplicantIncome', 'CoapplicantIncome'], axis = 1)
y = data.Loan_Status
X = pd.get_dummies(X)


# In[73]:


X.columns


# In[74]:


skf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
model = LogisticRegression(random_state = 1)
model.fit(X_train, y_train)
model.predict(X_test)
print(accuracy_score(y_test, model.predict(X_test)))


# ### Preparing the testing dataset

# In[75]:


test = pd.read_csv('../input/test_loan.csv')
test_original = test.copy()
test.head()


# In[76]:


test.shape


# In[77]:


test['LoanAmount'].fillna(test.LoanAmount.median(), inplace = True)
test['Loan_Amount_Term'].fillna(test.Loan_Amount_Term.mode()[0], inplace = True)
test['Gender'].fillna(test.Gender.mode()[0], inplace = True)
test['Married'].fillna(test.Married.mode()[0], inplace = True)
test['Dependents'].fillna(test.Dependents.mode()[0], inplace = True)
test['Self_Employed'].fillna(test.Self_Employed.mode()[0], inplace = True)
test['Credit_History'].fillna(test.Credit_History.mode()[0], inplace = True)


# In[78]:


test['ApplicantIncome'] = test['ApplicantIncome'].astype('float64')
test['Dependents'] = test['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
test['Total_Income'] =test['ApplicantIncome'] + test['CoapplicantIncome']
test = test.drop(['Loan_ID','ApplicantIncome', 'CoapplicantIncome'], axis = 1)


# In[79]:


test = pd.get_dummies(test)


# In[80]:


test.columns


# In[81]:


test.isna().sum()


# ### Prediction on the testing dataset

# In[82]:


prediction = model.predict(test)
submission = pd.DataFrame({'Loan_ID': test_original['Loan_ID'], 'Loan_Status': prediction})
submission['Loan_Status'].replace(0, 'N', inplace = True)
submission['Loan_Status'].replace(1, 'Y', inplace = True)


# In[83]:


submission.head()


# In[84]:


submission['Loan_Status'].value_counts(normalize = True).plot(kind = 'bar')

