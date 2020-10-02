#!/usr/bin/env python
# coding: utf-8

# Attrition is the silent killer that can switly disable even the most successful and stable of the organizations in a shockingly spare amount of time.
# 
# Hiring new employees are extremely complex task that requires capital, time and skills.Also new employee costs a lot more than that Persons salary.
# 
# The cost of hiring an employee goes far beyond just paying for their salary to encompass recruiting, training, benefits, and more.
# 
# Small companies spent, on average, more than $1,500 on training, per employee, in 2019.
# 
# Integrating a new employee into the organization can also require time and expenditures.
# 
# It can take up to six months or more for a company to break even on its investment in a new hire.
# 
# [Cost of hiring new employees](https://www.investopedia.com/financial-edge/0711/the-cost-of-hiring-a-new-employee.aspx)

# # Import libraries and dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


employee_df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


employee_df


# In[ ]:


employee_df.head(3)


# In[ ]:


employee_df.tail(3)


# In[ ]:


employee_df.info()


# In[ ]:


employee_df.describe()


# In[ ]:


employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x=='Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x=='Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x:1 if x== 'Y' else 0)


# In[ ]:


employee_df.head(4)


# In[ ]:


sns.heatmap(employee_df.isnull(),yticklabels = False, cbar = False, cmap="Blues");


# In[ ]:


employee_df.hist(bins = 30, figsize = (20,20), color='r');


# In[ ]:


employee_df.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'], axis=1, inplace=True)


# In[ ]:


left_df = employee_df[employee_df['Attrition'] == 1]
stayed_df = employee_df[employee_df['Attrition'] == 0]


# In[ ]:


print("Total =",len(employee_df))
print("Number of employee who left the company = ",len(left_df))
print("Percentage of the employee who left the company =",len(left_df)/len(employee_df)*100.0,"%")

print("Number of employee who stayed the company = ",len(stayed_df))
print("Percentage of the employee who stayed the company =",len(stayed_df)/len(employee_df)*100.0,"%")


# In[ ]:


left_df.describe()


# In[ ]:


stayed_df.describe()


# In[ ]:


correlations = employee_df.corr()
f,ax = plt.subplots(figsize = (20,20))
sns.heatmap(correlations,annot = True);


# In[ ]:


plt.figure(figsize=[25,12])
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df);


# In[ ]:


plt.figure(figsize=[20,20])
plt.subplot(421)
g=sns.countplot(x= 'JobRole', hue = 'Attrition', data = employee_df)
g.set_xticklabels(g.get_xticklabels(),rotation=15)
plt.subplot(422)
sns.countplot(x= 'MaritalStatus', hue='Attrition', data = employee_df)
plt.subplot(423)
sns.countplot(x= 'JobInvolvement', hue='Attrition', data = employee_df)
plt.subplot(424)
sns.countplot(x= 'JobLevel', hue='Attrition', data = employee_df)
plt.subplot(425)
sns.countplot(x= 'PercentSalaryHike', hue='Attrition', data = employee_df)
plt.subplot(426)
sns.countplot(x= 'PerformanceRating', hue='Attrition', data = employee_df)
plt.subplot(427)
sns.countplot(x= 'YearsSinceLastPromotion', hue='Attrition', data = employee_df)
plt.subplot(428)
sns.countplot(x= 'TotalWorkingYears', hue='Attrition', data = employee_df)


# In[ ]:


plt.figure(figsize=(12,7))
sns.kdeplot(left_df['DistanceFromHome'],label= 'Employee who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label= 'Employee who stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home');


# In[ ]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['YearsWithCurrManager'], label= 'Employee who left', shade= 'True', color='r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label= 'Employee who stayed', shade= 'True', color='b')

plt.xlabel('Years With Current Manager');


# In[ ]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], label='People who left', shade= True, color='r')
sns.kdeplot(stayed_df['TotalWorkingYears'], label='People who stayed', shade=True, color='b')

plt.xlabel('Total Working Years');


# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x= 'MonthlyIncome', y='Gender', data = employee_df);


# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x='MonthlyIncome', y = 'JobRole', data = employee_df);


# In[ ]:


X_cat = employee_df[['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']]
X_cat.head(3)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()


# In[ ]:


X_cat.shape


# In[ ]:


X_cat = pd.DataFrame(X_cat)


# In[ ]:


X_cat


# In[ ]:


X_numerical = employee_df[['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
X_numerical


# In[ ]:


X_all = pd.concat([X_cat,X_numerical],axis=1)
X_all


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)


# In[ ]:


X


# In[ ]:


y = employee_df['Attrition']
y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %" .format(100 * accuracy_score(y_pred,y_test)))


# In[ ]:


cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_pred))


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_pred))


# # Deep Learning Model

# In[ ]:


import tensorflow as tf


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500,activation='relu',input_shape=(50,)))
model.add(tf.keras.layers.Dense(units=500,activation='relu'))
model.add(tf.keras.layers.Dense(units=500,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


epochs_hist = model.fit(X_train,y_train,epochs=100,batch_size=50)


# In[ ]:


y_pred = model.predict(X_test)
y_pred = (y_pred>0.5)


# In[ ]:


y_pred


# In[ ]:


epochs_hist.history.keys()


# In[ ]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])


# In[ ]:


plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[ ]:


print(classification_report(y_test,y_pred))


# **Acknowledgement** :
# Shared based on my learnings from Stemplicity
