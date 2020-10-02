#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Importing & Loading the dataset

# In[ ]:


df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.head()


# ### Dataset Info:

# In[ ]:


df.info()


# ### Dataset Shape:

# In[ ]:


df.shape


# ### Dataset Description:

# In[ ]:


df.describe()


# ### Checking the Missing Values

# In[ ]:


df.isnull().sum()


# #### First we will fill the Missing Values in "LoanAmount" & "Credit_History" by the 'Mean' & 'Median' of the respective variables.

# In[ ]:


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())


# In[ ]:


df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())


# ### Let's confirm if there are any missing values in 'LoanAmount' & 'Credit_History'

# In[ ]:


df.isnull().sum()


# ### Now, Let's drop all the missing values remaining.

# In[ ]:


df.dropna(inplace=True)


# ### Let's check the Missing values for the final time!

# In[ ]:


df.isnull().sum()


# Here, we have dropped all the missing values to avoid disturbances in the model. The Loan Prediction requires all the details to work efficiently and thus the missing values are dropped.

# ### Now, Let's check the final Dataset Shape

# In[ ]:


df.shape


# ### Exploratory Data Analyis

# #### Comparison between Genders in getting the Loan:

# In[ ]:


sns.countplot(df['Gender'],hue=df['Loan_Status'])
print(pd.crosstab(df['Gender'],df['Loan_Status']))


# Here, we can see that the **Males** have more chances to get the Loan.

# #### Comparison between Married Status in getting the Loan:

# In[ ]:


sns.countplot(df['Married'],hue=df['Loan_Status'])
print(pd.crosstab(df['Married'],df['Loan_Status']))


# Here, we can see that the **Married Person** has more chance of getting the Loan.

# #### Comparison between Education Status of an Individual in getting the Loan:

# In[ ]:


sns.countplot(df['Education'],hue=df['Loan_Status'])
print(pd.crosstab(df['Education'],df['Loan_Status']))


# Here, we can see that a **Graduate Individual** has more chance of getting the Loan.

# #### Comparison between Self-Employed or Not in getting the Loan:

# In[ ]:


sns.countplot(df['Self_Employed'],hue=df['Loan_Status'])
print(pd.crosstab(df['Self_Employed'],df['Loan_Status']))


# Here, we can see that **Not Self-Employed** has more chance of getting the Loan.

# #### Comparison between Property Area for getting the Loan:

# In[ ]:


sns.countplot(df['Property_Area'],hue=df['Loan_Status'])
print(pd.crosstab(df['Property_Area'],df['Loan_Status']))


# Here, we can see that People living in **Semi-Urban** Area have more chance to get the Loan.

# ### Let's replace the Variable values to Numerical form & display the Value Counts
# 
# The data in Numerical form avoids disturbances in building the model. 

# In[ ]:


df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)


# In[ ]:


df['Loan_Status'].value_counts()


# In[ ]:


df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()


# In[ ]:


df.Married=df.Married.map({'Yes':1,'No':0})
df['Married'].value_counts()


# In[ ]:


df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependents'].value_counts()


# In[ ]:


df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()


# In[ ]:


df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()


# In[ ]:


df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df['Property_Area'].value_counts()


# In[ ]:


df['LoanAmount'].value_counts()


# In[ ]:


df['Loan_Amount_Term'].value_counts()


# In[ ]:


df['Credit_History'].value_counts()


# ### Display the Correlation Matrix

# In[ ]:


plt.figure(figsize=(16,5))
sns.heatmap(df.corr(),annot=True)
plt.title('Correlation Matrix (for Loan Status)')


# From the above figure, we can see that **Credit_History** (Independent Variable) has the maximum correlation with **Loan_Status** (Dependent Variable). Which denotes that the Loan_Status is heavily dependent on the Credit_History.

# ### Final DataFrame

# In[ ]:


df.head()


# ### Importing Packages for Classification algorithms

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# ### Splitting the data into Train and Test set

# In[ ]:


X = df.iloc[1:542,1:12].values
y = df.iloc[1:542,12].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


# In[ ]:


print(X_train)


# In[ ]:


print(X_test)


# In[ ]:


print(y_train)


# ### Logistic Regression (LR)

# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))


# ### Support Vector Machine (SVM)

# In[ ]:


model = svm.SVC()
model.fit(X_train,y_train)

svc_prediction = model.predict(X_test)
print('SVM accuracy = ', metrics.accuracy_score(svc_prediction,y_test))


# ### Decision Tree

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)

dt_prediction = model.predict(X_test)
print('Decision Tree accuracy = ', metrics.accuracy_score(dt_prediction,y_test))


# ### K-Nearest Neighbors (KNN)

# In[ ]:


model = KNeighborsClassifier()
model.fit(X_train,y_train)

knn_prediction = model.predict(X_test)
print('KNN accuracy = ', metrics.accuracy_score(knn_prediction,y_test))


# **CONCLUSION:**
# 
# 1. The Loan Status is heavily dependent on the Credit History for Predictions.
# 2. The Logistic Regression algorithm gives us the maximum Accuracy (79% approx) compared to the other 3 Machine Learning Classification Algorithms.

# In[ ]:




