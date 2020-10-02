#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns
import os


# In[ ]:


data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[ ]:


data.head(15).T


# In[ ]:


data.dtypes


# In[ ]:


data.describe().T


# In[ ]:


data['Attrition'].value_counts()


# In[ ]:


data.isnull().sum()


# NO Null values Present

# In[ ]:


cor = data.corr()


# In[ ]:


cor


# In[ ]:


data.shape


# In[ ]:


data['Over18'].value_counts()


# In[ ]:


data['StandardHours'].value_counts()


# In[ ]:


data['EmployeeCount'].value_counts()


# as Over18 ,StandardHours and EmployeeCount have constant value so we can remove them

# In[ ]:


data = data.drop(['EmployeeCount','StandardHours','Over18'],axis=1)


# In[ ]:


data.shape


# AGE

# In[ ]:


plt.boxplot(data['Age'])
plt.show()


# In[ ]:


plt.hist(data['Age'])
plt.show()


# In[ ]:


data['Age'].value_counts()


# Attrition

# In[ ]:


data['Attrition'].value_counts()


# In[ ]:


data.head()


# Business Travel

# In[ ]:


data['BusinessTravel'].value_counts()


# In[ ]:


plt.bar(data['BusinessTravel'],height= 1500)
plt.show()


# Daily Rate

# In[ ]:


plt.boxplot(data['DailyRate'])
plt.show()


# In[ ]:


plt.hist(data['DailyRate'])
plt.show()


# In[ ]:


plt.hist(data['DailyRate'])
plt.yscale('log')
plt.show()


# Department

# In[ ]:


data['Department'].value_counts()


# In[ ]:


plt.bar(data['Department'],height=1000)


# Distance From Home

# In[ ]:


data['DistanceFromHome'].value_counts()


# In[ ]:


plt.boxplot(data['DistanceFromHome'])
plt.show()


# In[ ]:


plt.hist(data['DistanceFromHome'])
plt.show()


# Education

# In[ ]:


data['Education'].value_counts()


# Education Field

# In[ ]:


data['EducationField'].value_counts()


# Employee Number

# In[ ]:


data['EmployeeNumber'].value_counts()


# so each employee has one unique EmployeeNumber

# In[ ]:


data = data.drop(['EmployeeNumber'],axis=1)


# In[ ]:


data.head().T


# In[ ]:


plt.boxplot(data['HourlyRate'])
plt.show()


# In[ ]:


plt.figure(1)
plt.subplot(1,3,1)
plt.hist(data['HourlyRate'])

plt.subplot(1,3,2)
plt.hist(data['HourlyRate'])
plt.yscale('log')


plt.subplot(1,3,3)
plt.hist(np.log(data['HourlyRate']))
plt.show()


# In[ ]:


data['JobRole'].value_counts()


# In[ ]:


plt.boxplot(data['MonthlyIncome'])
plt.show()


# In[ ]:


data['MonthlyIncome'].describe()


# In[ ]:


plt.hist(data['MonthlyIncome'])
plt.show()


# In[ ]:


iqr = ((np.percentile(data['MonthlyIncome'],75))- (np.percentile(data['MonthlyIncome'],25)))
limit = ((np.percentile(data['MonthlyIncome'],75)) + (1.5*iqr))
data.loc[data['MonthlyIncome']>limit,'MonthlyIncome'] = np.median(data['MonthlyIncome'])


# MonthlyIncome has some outliers , treating outliers now.

# In[ ]:


plt.boxplot(data['MonthlyRate'])
plt.show()


# In[ ]:


data['NumCompaniesWorked'].value_counts()


# In[ ]:


plt.hist(data['NumCompaniesWorked'])
plt.show()


# In[ ]:


data['OverTime'].value_counts()


# In[ ]:


data['PercentSalaryHike'].value_counts()


# In[ ]:


plt.boxplot(data['PercentSalaryHike'])
plt.show()


# In[ ]:


plt.hist(data['PercentSalaryHike'])
plt.show()


# In[ ]:


plt.hist(np.log(data['PercentSalaryHike']))
plt.yscale('log')
plt.show()


# In[ ]:


data['PerformanceRating'].value_counts()


# In[ ]:


data['RelationshipSatisfaction'].value_counts()


# In[ ]:


data['StockOptionLevel'].value_counts()


# In[ ]:


plt.hist(data['StockOptionLevel'])
plt.show()


# In[ ]:


data['TotalWorkingYears'].value_counts()


# In[ ]:


plt.hist(data['TotalWorkingYears'])
plt.show()


# In[ ]:


data['TrainingTimesLastYear'].value_counts()


# In[ ]:


data['WorkLifeBalance'].value_counts()


# In[ ]:


data['YearsAtCompany'].value_counts()


# In[ ]:


data['YearsInCurrentRole'].value_counts()


# In[ ]:


plt.hist(data['YearsSinceLastPromotion'])
plt.show()


# In[ ]:


data['YearsSinceLastPromotion'].value_counts()


# In[ ]:


data['YearsWithCurrManager'].value_counts()


# Bivariate Analysis

# In[ ]:


data.head().T


# In[ ]:


data.dtypes


# Label Encoding Attrition,BusinessTravel and OverTime

# In[ ]:


data.loc[data['Attrition']=='No','Attrition']=0
data.loc[data['Attrition']=='Yes','Attrition']=1


# In[ ]:


data.loc[data['BusinessTravel']=='Non-Travel','BusinessTravel']=0
data.loc[data['BusinessTravel']=='Travel_Rarely','BusinessTravel']=1
data.loc[data['BusinessTravel']=='Travel_Frequently','BusinessTravel']=2


# In[ ]:


data.loc[data['OverTime']=='No','OverTime']=0
data.loc[data['OverTime']=='Yes','OverTime']=1


# In[ ]:


data.dtypes


# In[ ]:


data = pd.get_dummies(data)


# In[ ]:


data.head(8).T


# In[ ]:


data = data.drop(['Department_Human Resources','EducationField_Life Sciences','Gender_Female','JobRole_Laboratory Technician','MaritalStatus_Divorced'],axis=1)


# In[ ]:


data.dtypes


# In[ ]:


data.shape


# In[ ]:


features = list((data.drop(['Attrition'],axis=1)).columns)
target = 'Attrition'
print(features)
print(target)
print(len(features))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[features],data[target],test_size=0.3,random_state=1)


# In[ ]:


print("X_train:",len(x_train))
print("X_test:",len(x_test))
print("Y_train:",len(y_train))
print("Y_test:",len(y_test))


# In[ ]:


y_test.value_counts()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=35,criterion="entropy")
rf.fit(x_train,y_train)


# In[ ]:


from sklearn import metrics
print("Random Forest")
print("Accuracy: ",rf.score(x_test,y_test))
y_pred = rf.predict(x_test)
print("Precision: ",metrics.precision_score(y_test,y_pred))
print("Recall: ",metrics.recall_score(y_test,y_pred))
print("Confusion Matrix: \n",metrics.confusion_matrix(y_test,y_pred))


# Handle Imbalanced  Classes

# In[ ]:


majority_class = data[data['Attrition']==0]
minority_class = data[data['Attrition']==1]
print(len(majority_class))
print(len(minority_class))


# In[ ]:


from sklearn.utils import resample
minority_class_upsampled = resample(minority_class,replace=True,n_samples=1233,random_state=1)


# In[ ]:


data_balanced = pd.concat([majority_class,minority_class_upsampled])


# In[ ]:


data_balanced['Attrition'].value_counts()


# In[ ]:


x_train1,x_test1,y_train1,y_test1 = train_test_split(data_balanced[features],data_balanced[target],test_size=0.3,random_state = 1)


# In[ ]:


print('RandomForest')
rf.fit(x_train1,y_train1)


# In[ ]:


y_test1.value_counts()


# In[ ]:


print("Random Forest")
print("Accuracy: ",rf.score(x_test1,y_test1))
y_pred1 = rf.predict(x_test1)
print("Precision: ",metrics.precision_score(y_test1,y_pred1))
print("Recall: ",metrics.recall_score(y_test1,y_pred1))
print("Confusion Matrix: \n",metrics.confusion_matrix(y_test1,y_pred1))


# In[ ]:


feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = features,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

