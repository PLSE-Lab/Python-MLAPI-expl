#!/usr/bin/env python
# coding: utf-8

# # Project on Predicting the attrition of Employee in a Company

# In[ ]:


#importing the required libraries
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# importing the data set 

# In[ ]:


employee_df = pd.read_csv('../input/hrdepartment20_may_2020.csv')
employee_df.head()


# In[ ]:


employee_df.info()


# In[ ]:


employee_df.describe()


# In[ ]:


#replacing Yes/no in attrition column with 1/0
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[ ]:


employee_df.head()


# In[ ]:


employee_df.hist(figsize=(20,20))
plt.show()


# In[ ]:


#Removing EmployeeCount, EmployeeNumber, Over18, StandardHours as they dont add much value to the code

employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)


# In[ ]:


employee_df.shape


# Shape has reduced from 35 to 31 columns as we dropped 4 columns

# In[ ]:


#dataframes for people who stayed or left
left_df = employee_df[employee_df['Attrition'] == 1]
stayed_df = employee_df[employee_df['Attrition'] == 0]


# # Lets analyse the data now

# In[ ]:


#based on Age
plt.figure(figsize=[15, 8])
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df)
plt.show()


# From the graph above, it could be observed that there were higher percentage of people leaving the firm at lower age .
# This is quite evident from general intuition as well

# In[ ]:


plt.figure(figsize=[20,12])
plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)
plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)
plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)
plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)
plt.show()


# In[ ]:



plt.figure(figsize=(10,6))

sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home')


# From the curve, it seems employees staying far from home have higher chances of attrition

# In[ ]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'b')

plt.xlabel('Total Working Years')


# Employees with less total working years have higher attrition

# In[ ]:


#checking monthly income based on job role
plt.figure(figsize=(10, 6))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)


# There are some job groups having higher salary range. These job groups might have lower attrition compared to other groups 
# 

# In[ ]:


employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[ ]:


employee_df.OverTime


# In[ ]:


X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]

X= X_numerical
X


# In[ ]:


y= employee_df['Attrition']
y


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_numerical)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy of the Model is",100*accuracy_score(y_pred,y_test), '%')


# In[ ]:


# Testing Set Performance
ConfusionMatrix = confusion_matrix(y_pred, y_test)
sns.heatmap(ConfusionMatrix, annot=True,linewidths=1 )
plt.show()


# In[ ]:


ConfusionMatrix


# In[ ]:


print(classification_report(y_test, y_pred))


# Although the accuracy of the model is good , but here f1 score needs to be checked , which is on the lower side. The model needs to be further worked on to improve this number

# # Trying Random Forest to check the accuracy

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


RFConfusionMatrix = confusion_matrix(y_pred, y_test)
sns.heatmap(RFConfusionMatrix, annot=True)


# In[ ]:


print("Clasiification Report from Random Forest \n",classification_report(y_test, y_pred))


# Random Forest gives even lower f1 score,
# From the f1-scores, we can say that the Logistic Regression Model has worked better. Although, that also needs further improvement

# # Will further try further data analysis to improve the accuracy of the Model

# In[ ]:




