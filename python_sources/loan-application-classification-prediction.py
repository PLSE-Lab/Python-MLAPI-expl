#!/usr/bin/env python
# coding: utf-8

# HI Everyone,
# 
# In this tutorial, we are going to do a complete data science life cycle. We have a loan application dataset. It has many features which helps banking team to decide whether the user is eligible for loan or not.
# It is the case of classification problem with 2 classes ( Approved or Not Appoved). 
# 
# What we will learn in here:
# 1. Desctriptive Analysis
# 2. Basic Statistics
# 3. Missing Value Handling
# 4. Encoding Categorical Features
# 5. Visualization & Story Telling
# 6. Train Test Split
# 7. Random Forest Algorithm
# 8. Model Tuning using GridSearch and k-fold
# 9. Performance Metics.
#  

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


# # Load & Explore Data

# In[ ]:


df = pd.read_csv("/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv")


# In[ ]:


df.head() # top 5 reacords


# In[ ]:


df.isnull().sum() # Find total number of missing 


# # Visualization

# Before fixing all the missing values lets just explore our data
# So data visualization is very effective method to get info from any kind of data.
# but before doing anything we have to think one thing..... What are we going to find????
# Our findings will be:
# 1. How many males or females apply for the loans?
# 2. How many graduate or non-graduated male or female apply for the loan?
# 3. Is self employed people are most likely to get loan?
# 4. What kind of male or females apply for loan, married or unmarried?
# 
# So i think these 4 finding will be enough for you to explain a real case study.

# ## Q1. How many males or females apply for the loans?

# In[ ]:


# to visualize our data we will use matplotlib
import matplotlib.pyplot as plt


# In[ ]:


# To answer this data we will use value_counts to find uot gender counts.
df.Gender.value_counts() # So we can see mostly Males are applying for the loan


# In[ ]:


# Lets plot it
df.Gender.value_counts().plot(kind="bar")


# In[ ]:


df.Gender.value_counts(normalize=True)


# In[ ]:


# we can also plot the percentage
df.Gender.value_counts(normalize=True).plot(kind="bar"); # so we can see almost 80 loan applicants are Males.
# So here we find out our first answer.


# Q1. How many males or females apply for the loans?
# Ans. 81.3% Males, 18.6% Females

# ## Q2. How many graduate or non-graduated male or female apply for the loan?

# In[ ]:


# So in this case we have to group our data according to gender and education
 df.groupby(["Gender", "Education"])["Loan_Status"].count()
    # So we can see there are 92 graduated females & 20 Non-Graduated females


# In[ ]:


df.groupby(["Gender", "Education"])["Loan_ID"].count().plot(kind="bar");
# So we can answer here our second question.


# ## Q3.Is self employed people are most likely to get loan?

# In[ ]:


# Lets check whether selfemployed people are more likely to get loan?
df.groupby(["Self_Employed", "Loan_Status"])["Loan_ID"].count()


# In[ ]:


df.groupby(["Self_Employed", "Loan_Status"])["Loan_ID"].count().plot(kind='bar')


# In[ ]:


# So answer 3rd is according to our data 
# So we can see most of the approvals are in account of the persons who is not self employed.


# ## Q4. What kind of male or females apply for loan, married or unmarried?

# In[ ]:


df.groupby(["Gender", "Married"])["Loan_ID"].count().plot(kind="bar");
# So we can see mostly married males apply for loan.


# In[ ]:


# So till now we answered our 4 questions. Thoough in real life you will find even more complicate questions. But still
# its a good start.


# # Handling Missing Value

# In[ ]:


# So finlly we reached to this step. It is very important and interesting step.
# Lets check again how many null values we have
df.isnull().sum()


# In[ ]:


#df_gen_dum = pd.get_dummies(df.Gender)


# In[ ]:


df.LoanAmount


# In[ ]:


# Lets replace gender with most frequent or mode value as it is a categorical feature
mode_gen = df.Gender.value_counts().idxmax()# so we can see Male is the most common value so lets replace with it
df.Gender.fillna(mode_gen , inplace=True)


# In[ ]:


df.isnull().sum() # See now gender column doesnt have any mssing value


# In[ ]:


# Lets do same for the married column
df[ "Married"].fillna(df.Married.value_counts().idxmax(), inplace=True) # here all missing values are replace with "YES" as
# It is the most frequent value


# In[ ]:


# Lets do it for dependent column.
df.Dependents.value_counts().idxmax() # We can see mostly people has no dependent. So we wil use that value


# In[ ]:


df.Dependents.fillna(df.Dependents.value_counts().idxmax(), inplace=True)


# In[ ]:


df.Self_Employed.value_counts().idxmax()


# In[ ]:


df.Self_Employed.fillna(df.Self_Employed.value_counts().idxmax(), inplace=True)


# In[ ]:


# now lets see for loan amount
df.LoanAmount.plot("hist") # so we can see loan amount is float value. Let plot it first


# In[ ]:


df.LoanAmount.mean()


# In[ ]:


# So from our histograme we can see mostly loan amount lies between 100-150 and our mean value is 146. Which means 
# we can replace our missing values with mean in this case.
df.LoanAmount.fillna(df.LoanAmount.mean(), inplace=True)


# In[ ]:


# Loan Term column is the number of days for the loan. So here we can not use mean value as mean value probably gives us 
# decimal values and days are obvously not in decimals. So here also lets see what is the most common value.
# We can see 85% cases term value is 360. So we will replace it with this only
df.Loan_Amount_Term.value_counts(normalize=True)


# In[ ]:


df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.value_counts().idxmax(), inplace=True)


# In[ ]:


df.Credit_History.value_counts() # again we can see in case of credit history also we can only have two values
# and mostl we have 1 value. So lets do i.


# In[ ]:


df.Credit_History.fillna(df.Credit_History.value_counts().idxmax(), inplace=True)


# In[ ]:


df.isnull().sum() # so now finally we have replaced all the missing values.


# In[ ]:


df.head()


# In[ ]:


# We can see there are few categrical featues ike Gender, Married, Education, Self_Employed, Property_Area, Load_Status
# As machine learning algorithm only understands numerical values. So we have to change them into numeric values.
# to do that we add dummy variables for each categorical value


# In[ ]:


# before everythning lets delete Loan_ID column as we do not need it.
df.drop(["Loan_ID"],axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


# Lets do it
dummy_gen = pd.get_dummies(df.Gender) # So we can see get_dummies() change our categorical values into dummy variable


# In[ ]:


dummy_gen.head()


# In[ ]:


# lets add these values to our main df
df = pd.concat([dummy_gen, df],  axis=1)


# In[ ]:


df.head()  # So we can see two new columns added to our dataset. So we dont need Gender column anymore.


# Also here there is one more problem. As we can see MAle & Female column are anti. Mean If Female =1 Then Male =0 vice versa.
# So we have to do delete one otherwise we will face dummy variable trap. 
# So lets delete Female & Gender column

# In[ ]:


df.drop(["Female", "Gender"], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


# now notice if we convert married column & Self_Employed  Column to dummy variable, 
# We will get same column name so it will be a problem & confusing.
df.Married.replace("No", "Unmarried", inplace=True)
df.Married.replace("Yes", "Married", inplace=True)

df.Self_Employed.replace("No", "Job", inplace=True)
df.Self_Employed.replace("Yes", "Business", inplace=True)


# In[ ]:


df.head() # so now our problem solved lets continuee..


# In[ ]:


# Lets do same for other columns. Dummy variable idea is like if u have n dummy variable take only n-1. In or case
# 2 columns were there Male, Female. We took only one.
# Lets apply for Married.
dummy_mar = pd.get_dummies(df.Married)
df.drop(["Married"], axis=1, inplace=True) # delete the original married column
df = pd.concat([dummy_mar, df], axis=1)
df.drop(["Unmarried"], axis=1, inplace=True)


# In[ ]:


# For Sel_Employed Column
dummy_emp = pd.get_dummies(df.Self_Employed)
df = pd.concat([dummy_emp, df], axis=1)
df.drop(["Business", "Self_Employed"], axis=1, inplace=True)


# In[ ]:


# For `education Column
dummy_edu = pd.get_dummies(df.Education)
df = pd.concat([dummy_edu, df], axis=1)
df.drop(["Not Graduate", "Education"], axis=1, inplace=True)


# In[ ]:


# For `Property_Area Column
dummy_prop = pd.get_dummies(df.Property_Area)
df = pd.concat([dummy_prop, df], axis=1)
df.drop(["Rural", "Property_Area"], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


# one more problem i just found in dependent column
df.Dependents.value_counts() # you can see there is 3+ vlaue which is string. So lets change it with 3


# In[ ]:


df.Dependents.replace("3+", 3, inplace=True) 
df.Dependents =df.Dependents.astype("int")  # change the column type to integer


# In[ ]:


df.Dependents.value_counts()


# In[ ]:


# well now we have replaced all the cateorical features to numerical values.


# # Feature & Target Set

# In[ ]:


X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values


# In[ ]:


X


# In[ ]:


y


# In[ ]:


# now we can see our y is still in categorical form. Let replace it also but with the help of labelencode class.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[ ]:


y # So you can see now our y is also encoded


# # Scaling

# In[ ]:


# now scaling means every data is in same range. But if you see continoues variables in our dataset. ApplicantIncome has a huge
# range if you compare to CoapplicantIncome & LoanAmount. So it is better to scale them to the same range. 
# We can do it wi


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler_ApplicantIncome = StandardScaler()
X = scaler_ApplicantIncome.fit_transform(X)


# In[ ]:


X # so we can see our data is now scaled.


# # Train & Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Step 1 we will calculate vairability of all the columns so that we can identify the principal components.
from sklearn.decomposition import PCA


# # Random Forest - Model

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(random_state=42)


# # Model Tuning

# In[ ]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[ ]:


print(CV_rfc.best_params_) # So these are our best parameters
print(CV_rfc.best_score_)


# In[ ]:


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=5, criterion='entropy')


# In[ ]:


rfc1.fit(X_train, y_train)


# In[ ]:


y_predict = rfc1.predict(X_test)


# In[ ]:


cf = confusion_matrix(y_test, y_predict) # confusion Matrix


# In[ ]:


cf


# In[ ]:


rfc1.score(X_test, y_test) # 81% accuracy


# Thanks You...

# In[ ]:




