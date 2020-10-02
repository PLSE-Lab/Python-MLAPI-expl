#!/usr/bin/env python
# coding: utf-8

# # Predicting Customer Churn
# 
# ## Introduction
# 
# In this analysis, I used an already available data from an imaginery bank in order to solve a very common problem. - Churn- The problem statement is straight forward, "What type of customers' are ending their contracts with the bank?". If the bank can predict the customers' that are leaving, organization can further use targeted marketing techniques in order to keep these individuals as customers. This is a well known practice in marketing, as it is much easier and cost effective to keep existing customers rather than acquiring new ones. 

# ## About the Data
# 
# The data is not from an existing bank and it is imaginery. This is a very common dataset used in institutations for educational purposes such as Standford or CUNY SPS. The data set provides below variables for each customer (observation).
# 
# *RowNumber : Row Numbers*
# 
# *CustomerID: Unique Ids for bank customer identification*
# 
# *Surname: Customer's last name*
# 
# *CreditScore: Credit score of the customer*
# 
# *Geography: The country from which the customer belongs*
# 
# *Gender: Male or Female*
# 
# *Age: Age of the customer*
# 
# *Tenure: Number of years for which the customer has been with the bank*
# 
# *Balance: Bank balance of the customer*
# 
# *NumOfProductsNumber of bank products the customer is utilising*
# 
# *HasCrCardBinary Flag for whether the customer holds a credit card with the bank or not*
# 
# *IsActiveMemberBinary Flag for whether the customer is an active member with the bank or not*
# 
# *EstimatedSalaryEstimated salary of the customer in Dollars*
# 
# *ExitedBinary flag 1 if the customer closed account with bank and 0 if the customer is retained*
# 

# ## Data Collection and Preperation

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/anilak1978/customer_churn/master/Churn_Modeling.csv")
df.head()


# In[ ]:


# Looking for missing data
missing_data=df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# There are no missing data in our dataset. 

# In[ ]:


# Looking at data types
df.dtypes


# All data types are as expected and doesnt require updating. 

# ## Data Exploration

# In[ ]:


# Looking at the basic information
df.info()


# There are 1000 customers and 14 attributes for each customer. 

# In[ ]:


# looking at the summary
df.describe()


# The average estimated salary is around100K with around 58K standard deviation. Average age of the customers is 39 with standard deviation of 10. Minimum age is 18 and maximum is 92. 

# In[ ]:


# Looking at correlation between numerical variables
df.corr()


# From an overivew, there isnt a significant correlation between two variables.

# In[ ]:


# Visualizing relationship of Age and Estimated Salary
plt.figure(figsize=(20,20))
sns.relplot(x="Age", y="EstimatedSalary", hue="Geography", data=df)
plt.title("Age VS Estimated Salary")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")


# In[ ]:


# looking at age distribution
plt.figure(figsize=(10,10))
sns.distplot(df["Age"])


# The Age distribution is normal and very vey lightly right skewed. Based on the Age VS Estimated Salary figure, we can see there are some outliers above the 80 years old mark.

# In[ ]:


# Looking at Gender Distribution
plt.figure(figsize=(10,8))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")


# There are more male customers than female customers however the difference is not significant.

# In[ ]:


# Looking at Geography and Gender Distribution against Estimated Salary
plt.figure(figsize=(20,20))
sns.catplot(x="Geography", y="EstimatedSalary", hue="Gender", kind="box", data=df)
plt.title("Geography VS Estimated Salary")
plt.xlabel("Geography")
plt.ylabel("Estimated Salary")


# The estimated salary distribution for all 3 regions are similar and ballpark around 100K. Interesting fact to note is that females expected salary is more than the males in Germany.

# In[ ]:


# Looking at linear relationship between Age and CreditScore
plt.figure(figsize=(10,10))
sns.regplot(x="Age", y="CreditScore", data=df)


# The linear relationship is not very strong between Age and Credit Score. 

# In[ ]:


#looking at correlation between attributes in detail
corr=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)


# Overall, I dont see a significant correlation between two or more specific variables.

# ## Model Development and Evaluation
# 
# Based on my data exploration, I decided to create a classification model using Decision Trees. I will create two models, one with using Decision Tree Classifier and another one using Random Forest Classifier. 

# In[ ]:


df.columns


# In[ ]:


# Selecting and Preparing the Feature Set and Target
X = df[["CreditScore", "Geography", "Gender", "Age", "Tenure", "EstimatedSalary"]].values
y=df[["Exited"]]
X[0:5], y[0:5]


# In[ ]:


# preprocessing categorical variables
from sklearn import preprocessing
geography=preprocessing.LabelEncoder()
geography.fit(["France", "Spain", "Germany"])
X[:,1]=geography.transform(X[:,1])

gender = preprocessing.LabelEncoder()
gender.fit(["Female", "Male"])
X[:,2]=gender.transform(X[:,2])

X[0:5]


# In[ ]:


# split train and test data
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)


# In[ ]:


# create model using DecisionTree Classifier and fit training data
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_trainset, y_trainset)


# In[ ]:


# create prediction
dt_pred = dt_model.predict(X_testset)
dt_pred[0:5]


# In[ ]:


# Evaluating the prediction model
from sklearn import metrics
metrics.accuracy_score(y_testset, dt_pred)


# At this point, I created the prediction model by classification using Decision Tree Classifier, created prediction using feature test set and evaluated the model. The accuracy score is 0.732 out of maximum possible score of 1. I will further create another model using Random Forest to see if that gives me a better score.

# In[ ]:


# create Random Forest Decision Tree model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_trainset, y_trainset.values.ravel())


# In[ ]:


# create prediction using rf_model
rf_pred = rf_model.predict(X_testset)
rf_pred[0:5]


# In[ ]:


# evaluate the model
metrics.accuracy_score(y_testset, rf_pred)


# Using Random Forest gave me a slightly better accuracy score.

# ## Conclusion
# 
# By looking at customers Credit Score, where they are from, gender, age, estimated salary and how many they have been a customer, we can predict if they will continue their contract or leave for another bank. Our prediction using Random Forest Classifier will give 81 % prediction accuracy. I can create additional data requirements, collect them from various sources and analyze further in order to come up with a marketing approach to keep the customers that is predicted to leave the bank.
