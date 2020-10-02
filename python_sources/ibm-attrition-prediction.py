#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# # Data Exploration

# ### Age

# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.countplot(data.Age)


# ### Attrition

# In[ ]:


plt.figure(figsize=(8,5))
ax = sns.countplot(data.Attrition)


# ### Business Travel

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.BusinessTravel) 


# ### Daily Rate

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.DailyRate)


# ### Distance From Home

# In[ ]:


plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.DistanceFromHome)


# In[ ]:


# Right Skew


# ### Department

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.Department) 


# ### Education

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.Education) 


# ### Education Field

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.EducationField) 


# ### EmployeeCount

# In[ ]:


data.EmployeeCount.value_counts()


# In[ ]:


# Not useful


# ### Employee Number

# In[ ]:


# Not useful 


# ### Environment Satisfaction

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.EnvironmentSatisfaction)


# ### Gender

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.Gender) 


# ### Hourly Rate

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.HourlyRate)


# ### Job Involvement

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.JobInvolvement)


# ### Job Level

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.JobLevel)


# ### Job Role

# In[ ]:


plt.figure(figsize=(15,6))
ax = sns.countplot(data.JobRole)
t=plt.xticks(rotation=45) 


# ### Job Satisfaction

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.JobSatisfaction)


# ### Marital Status

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.MaritalStatus) 


# ### Monthly Income

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.MonthlyIncome)


# In[ ]:


# Right Skew


# ### Monthly Rate

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.MonthlyRate)


# ### Num Companies Worked

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.NumCompaniesWorked)


# ### Over 18

# In[ ]:


data.Over18.value_counts()


# In[ ]:


# Not useful 


# ### Over Time

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.OverTime) 


# ### Percent Salary Hike

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.PercentSalaryHike) 


# In[ ]:


# A little right skew


# ### Performance Rating

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.PerformanceRating)


# ### Relationship Satisfaction

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.RelationshipSatisfaction)


# ### Standard Hours

# In[ ]:


data.StandardHours.value_counts()


# In[ ]:


# Not useful 


# ### Stock Option Level

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.StockOptionLevel)


# ### Total Working Years

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.TotalWorkingYears) 


# In[ ]:


# A little right skew


# ### Training Times Last Year

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.TrainingTimesLastYear)


# ### Work Life Balance

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(data.WorkLifeBalance)


# ### Years At Company

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.YearsAtCompany) 


# In[ ]:


# Right skew


# ### Years In Current Role

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.YearsInCurrentRole)


# In[ ]:


# There are 2 curves : young people vs older people


# ### Years Since Last Promotion

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.YearsSinceLastPromotion) 


# In[ ]:


# Right skew


# ### Years With Current Manager

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(data.YearsWithCurrManager)


# In[ ]:


# Almost the same as YearsInCurrentRole


# ### Age vs YearsAtCompany

# In[ ]:


g=data.groupby('Age')['YearsAtCompany'].mean().plot()


# ### Age vs YearsInCurrentRole

# In[ ]:


g=data.groupby('Age')['YearsInCurrentRole'].mean().plot()


# # Data Featuring

# In[ ]:


# Normalize features columns
# Models performe better when values are close to normally distributed
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[ ]:


data['DistanceFromHome'] = scaler.fit_transform(data['DistanceFromHome'].values.reshape(-1, 1))
data['MonthlyIncome'] = scaler.fit_transform(data['MonthlyIncome'].values.reshape(-1, 1))
data['PercentSalaryHike'] = scaler.fit_transform(data['PercentSalaryHike'].values.reshape(-1, 1))
data['TotalWorkingYears'] = scaler.fit_transform(data['TotalWorkingYears'].values.reshape(-1, 1))
data['YearsAtCompany'] = scaler.fit_transform(data['YearsAtCompany'].values.reshape(-1, 1))
data['YearsSinceLastPromotion'] = scaler.fit_transform(data['YearsSinceLastPromotion'].values.reshape(-1, 1))


# In[ ]:


# Convert to categorical values
data['Attrition'] = data.Attrition.astype('category').cat.codes
data['BusinessTravel'] = data.BusinessTravel.astype('category').cat.codes
data['Department'] = data.Department.astype('category').cat.codes
data['EducationField'] = data.EducationField.astype('category').cat.codes
data['Gender'] = data.Gender.astype('category').cat.codes
data['JobRole'] = data.JobRole.astype('category').cat.codes
data['MaritalStatus'] = data.MaritalStatus.astype('category').cat.codes 
data['OverTime'] = data.OverTime.astype('category').cat.codes


# In[ ]:


# Check NA
data.isnull().sum(axis = 0)


# # Correlation

# In[ ]:


# Remove columns not useful
data = data.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1)


# In[ ]:


# Get columns with at least 0.1 correlation
data_corr = data.corr()['Attrition'] # Attrition : column to predict
cols = data_corr[abs(data_corr) > 0.1].index.tolist()
data = data[cols]


# In[ ]:


# plot the heatmap
data_corr = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(data_corr, 
        xticklabels=data_corr.columns,
        yticklabels=data_corr.columns, cmap=sns.diverging_palette(220, 20, n=200))


# In[ ]:


data.corr()['Attrition'].sort_values(ascending=False)


# In[ ]:


# Check correlations between columns
data['JobLevel'].corr(data['MonthlyIncome'])


# In[ ]:


# Too much correlation


# In[ ]:


data['YearsAtCompany'].corr(data['YearsWithCurrManager'])


# In[ ]:


data['JobLevel'].corr(data['TotalWorkingYears'])


# In[ ]:


data['YearsInCurrentRole'].corr(data['YearsWithCurrManager'])


# In[ ]:


# Remove columns with too much correlation
data = data.drop(["MonthlyIncome", "YearsAtCompany", "JobLevel", "YearsWithCurrManager"], axis=1)


# In[ ]:


data.columns


# # Creating the model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


X = data.drop("Attrition", axis=1)
Y = data["Attrition"]


# In[ ]:


# Split 20% test, 80% train

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# In[ ]:


# Logistic Regression

log = LogisticRegression(max_iter=1000)
log.fit(X_train, Y_train)
Y_pred_log = log.predict(X_test)
acc_log = accuracy_score(Y_pred_log, Y_test)
acc_log


# In[ ]:


t = tree.DecisionTreeClassifier()

# search the best params
grid = {'min_samples_split': [5, 10, 20, 50, 100]},

clf_tree = GridSearchCV(t, grid, cv=10)
clf_tree.fit(X_train, Y_train)

Y_pred_tree = clf_tree.predict(X_test)

# get the accuracy score
acc_tree = accuracy_score(Y_pred_tree, Y_test)
print(acc_tree)


# In[ ]:


rf = RandomForestClassifier()

# search the best params
grid = {'n_estimators':[100,200], 'max_depth': [2,5,10]}

clf_rf = GridSearchCV(rf, grid, cv=10)
clf_rf.fit(X_train, Y_train)

Y_pred_rf = clf_rf.predict(X_test)
# get the accuracy score
acc_rf = accuracy_score(Y_pred_rf, Y_test)
print(acc_rf)


# # Conclusion

# In[ ]:


# The best model is Logistic Regression

