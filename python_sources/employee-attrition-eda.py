#!/usr/bin/env python
# coding: utf-8

# # IBM_HR_Analytics_Employee_Attrition_Performance 
# # Exploratory Data Analysis (EDA)

# ### The dataset is about employee attrition. This analysis can discover if any particular factors or patterns that lead to attrition. If so, employers can take certain precausion to prevent attrition which in employer of view, employee attrition is a loss to company, in both monetary and non-monetary. 

# ### **Importing the packages**

# In[ ]:


##Importing the packages
#Data processing packages
import numpy as np 
import pandas as pd 

#Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')


# ### **Importing the data**

# In[ ]:


#Import Employee Attrition data
data=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# ### **Basic Analysis**

# In[ ]:


#Find the size of the data Rows x Columns
data.shape


# **COMMENTS:** The data consists of 1470 rows and 35 columns

# In[ ]:


#Display first 5 rows of Employee Attrition data
data.head()


# In[ ]:


#Find Basic Statistics like count, mean, standard deviation, min, max etc.
data.describe()


# **COMMENTS:** 
# 1. Count of 1470 for all the fields indicates that there are no missing values in any of the field
# 2. Standard deviation (std) is ZERO for fields "EmployeeCount" and "StandardHours".  This indicates that all the values in the given field are same.
# 3. Minimum(min) and Maximum(max) defines the range of values for that field.
# 4. Mean(mean) indicates average of all the values in the field.  There is large variation of mean values of the fields so we need to scale the data.
# 5. 25%, 50%, 75% percentiles indicates the distribution of data

# In[ ]:


#Find the the information about the fields, field datatypes and Null values
data.info()


# **Category Columns**

# In[ ]:


cat_cols = data.columns[data.dtypes=='object']
data_cat = data[cat_cols]
print(cat_cols)
print(cat_cols.shape)
data_cat.head()


# In[ ]:


#A lambda function is a small anonymous function.
#A lambda function can take any number of arguments, but can only have one expression.
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)


# In[ ]:


data.head()


# In[ ]:


data[data.Attrition == 1].head()


# **Numerical Columns**

# In[ ]:


num_cols = data.columns[data.dtypes!='object']
data_num = data[num_cols]
print(num_cols)
print(num_cols.shape)
data_num.head()


# In[ ]:


data.corrwith(data.Attrition, axis = 0).sort_values().head()


# In[ ]:


data.corrwith(data.Attrition, axis = 0).sort_values(ascending = False).head()


# In[ ]:


sns.countplot(data.TotalWorkingYears, hue=data.Attrition)


# In[ ]:


sns.countplot(data.DistanceFromHome, hue=data.Attrition)


# **COMMENTS:**  Info fuction is used to list all the field names, their datatypes, count of elements in the field and if the field contacts Null values.

# In[ ]:


data.JobLevel.value_counts().plot.bar()


# In[ ]:


sns.countplot(data.JobLevel, hue=data.Attrition)


# In[ ]:


data[data.Attrition==1].JobLevel.value_counts(normalize=True, sort=False).plot.bar()


# In[ ]:


data[data.Attrition==1].DistanceFromHome.value_counts(normalize=True, sort=False).plot.bar()


# In[ ]:


plt.figure(figsize=(20, 20)) ; sns.heatmap(data_num.corr(), annot=True)


# In[ ]:


g = sns.pairplot(data_num.loc[:,'Age':'DistanceFromHome']); g.fig.set_size_inches(15,15)
#data_num.loc[:,'Age':'DistanceFromHome']


# In[ ]:


g = sns.pairplot(data_num.loc[:,'Education':'HourlyRate']); g.fig.set_size_inches(15,15)


# In[ ]:


g = sns.pairplot(data_num.loc[:,'JobInvolvement':'MontlyRate']); g.fig.set_size_inches(15,15)


# In[ ]:


g = sns.pairplot(data_num.loc[:,'NumCompaniesWorked':'StandardHours']); g.fig.set_size_inches(15,15)


# In[ ]:


g = sns.pairplot(data_num.loc[:,'StockOptionLevel':'YearsAtCompany']); g.fig.set_size_inches(15,15)


# In[ ]:


g = sns.pairplot(data_num.loc[:,'YearsInCurrentRole':'YearsWithCurrManager']); g.fig.set_size_inches(15,15)


# In[ ]:


g = sns.pairplot(data_num); g.fig.set_size_inches(15,15)


# In[ ]:


data_num.hist(layout = (9, 3), figsize=(24, 48), color='blue', grid=False, bins=15)


# ### **Visualizing the impact of Categorical Features on the Target**

# In[ ]:


#Find attrition size (Values)
data['Attrition'].value_counts()


# **COMMENTS:**  237 employees left the company out of total 1470 employees

# In[ ]:


pd.crosstab(data.BusinessTravel, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Frequent travelers  are more likely(25%) to leave the company as compared to Non Travellers

# In[ ]:


pd.crosstab(data.Department, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  The employees in "Sales" department are more likely(21%) to leave the company as compared to the employees of other department

# In[ ]:


pd.crosstab(data.Education, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  The employees who are least educated are more likely(18%) to leave the company and the employees who are highly qualified are less likely (10%) to leave the company.

# In[ ]:


pd.crosstab(data.EducationField, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  The employees in "Human Resource" education field are more likely(26%) to leave the company.  Next in the line are from the "Technical" field.

# In[ ]:


pd.crosstab(data.EnvironmentSatisfaction, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Lower the "Environment Satisfaction" higher the attrition rate(25%)

# In[ ]:


pd.crosstab(data.Gender, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Male employees have slightly higher attrition rate (17%) as compared to female employees.

# In[ ]:


pd.crosstab(data.JobInvolvement, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Lower the "JobInvolvement", higher the Attrition rate (34%)

# In[ ]:


pd.crosstab(data.JobLevel, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Lower the "JobLevel", higher the Attrition rate (26%)

# In[ ]:


pd.crosstab(data.JobRole, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Employees with with "Sales Representative" JobRole have the higher attrition rate (40%) as compared to others.

# In[ ]:


pd.crosstab(data.JobSatisfaction, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Lower the "Job Satisfaction" higher the attrition rate(23%)

# In[ ]:


pd.crosstab(data.MaritalStatus, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Employees who are Single have significantly higher attrition rate(26%)

# In[ ]:


pd.crosstab(data.OverTime, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Employees who do "OverTime" have significantly higher attrition rate(31%)

# In[ ]:


pd.crosstab(data.PerformanceRating, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Performance Rating has no effect on the Attrition rate.

# In[ ]:


pd.crosstab(data.RelationshipSatisfaction, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Employees with less Relationship Satisfaction are more likely to leave the company (21%)

# In[ ]:


pd.crosstab(data.StockOptionLevel, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Employees who are NOT given "Stock Option" are more likely(24%) to leave the company

# In[ ]:


pd.crosstab(data.WorkLifeBalance, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENTS:**  Employees with less "Work Life Balance" are more likely (31%) to leave the company.

# In[ ]:


#A lambda function can take any number of arguments, but can only have one expression.
#Change the Attrition from Yes/No to binary 1/0
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)


# ### **Visualizing the impact of Numerical Features on the Target**

# In[ ]:


#Comparing the numeric fields agains Attrition using boxplots
plt.figure(figsize=(24,12))
plt.subplot(231)  ; sns.boxplot(x='Attrition',y='Age',data=data)
plt.subplot(232)  ; sns.boxplot(x='Attrition',y='DailyRate',data=data)
plt.subplot(233)  ; sns.boxplot(x='Attrition',y='DistanceFromHome',data=data)
plt.subplot(234)  ; sns.boxplot(x='Attrition',y='HourlyRate',data=data)
plt.subplot(235)  ; sns.boxplot(x='Attrition',y='MonthlyIncome',data=data)
plt.subplot(236)  ; sns.boxplot(x='Attrition',y='PercentSalaryHike',data=data)


# In[ ]:


#Comparing the numeric fields agains Attrition using boxplots
plt.figure(figsize=(24,12))
plt.subplot(231)  ; sns.boxplot(x='Attrition',y='MonthlyRate',data=data)
plt.subplot(232)  ; sns.boxplot(x='Attrition',y='NumCompaniesWorked',data=data)
plt.subplot(233)  ; sns.boxplot(x='Attrition',y='TotalWorkingYears',data=data)
plt.subplot(234)  ; sns.boxplot(x='Attrition',y='TrainingTimesLastYear',data=data)
plt.subplot(235)  ; sns.boxplot(x='Attrition',y='YearsAtCompany',data=data)
plt.subplot(236)  ; sns.boxplot(x='Attrition',y='YearsInCurrentRole',data=data)


# In[ ]:


#Comparing the numeric fields agains Attrition using boxplots
plt.figure(figsize=(24,6))
plt.subplot(121)  ; sns.boxplot(x='Attrition',y='YearsSinceLastPromotion',data=data)
plt.subplot(122)  ; sns.boxplot(x='Attrition',y='YearsWithCurrManager',data=data)


# In[ ]:


#Correlation plot to find interelationship of the features
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True)


# In[ ]:


#sns.pairplot(data['BusinessTravel','Gender','Attrition'], hue='Attrition')
#sns.pairplot(data, vars=["Gender", "Attrition"])


# In[ ]:





# In[ ]:




