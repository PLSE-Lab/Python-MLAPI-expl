#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Data Aquisition and Read data**

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv') 
combine = [df] 
mpl.style.use("ggplot")
print("Done")


# In[ ]:


df.head(10)


# **Basic Insight Of Dataset.**

# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df.describe(include='all')


# In[ ]:


df.info


# **Identify and handle missing values.**

# In[ ]:


df.replace("?",np.nan,inplace= True) 
df.head()


# In[ ]:


missing_data = df.isnull() 
missing_data.tail()


# In[ ]:


for col in missing_data.columns.values.tolist(): 
    print(col) 
    print(missing_data[col].value_counts()) 
    print('-'*20)


# **Data Visualization**

# In[ ]:


print("Total Employees in each department :")
df.Department.value_counts().head() 


# In[ ]:


df['Department'].value_counts().head().plot(kind='bar',figsize=(10,6)) 
plt.ylabel("No. of employees") 
plt.xlabel("Department")
plt.title("Comparision of total no. of employees vs Department")


# In[ ]:


df_copy = df.copy() 
df_copy.head()


# In[ ]:



df_t = df_copy[['EmployeeNumber','MonthlyIncome']]
df_t.set_index('EmployeeNumber',inplace=True) 
df_t.head()


# In[ ]:


count,bin_edges=np.histogram(df_t) 
print(count) 
print(bin_edges)


# In[ ]:


df_t.plot(kind='hist', figsize=(8, 5),color='chartreuse') 
plt.xlabel("Monthly Salary")


#  We can infer that:
# 
# 1. 365 employees have monthly income between 1009 to 2908
# 2. 349 employees have monthly income between 2908 to 4807
# 3. 290 employees have monthly income between 4807 to 6706, and so on..
# 

# In[ ]:


df_b=df[['Education', 'MonthlyIncome']].groupby(['Education'], as_index=False).mean().sort_values(by='MonthlyIncome', ascending=False) 
df_b.set_index('Education',inplace=True) 
df_b.head()


# In[ ]:


df_b.plot(kind='bar',figsize=(10,6),color='Rybgm') 
plt.ylabel("Average Monthly income") 
plt.title("Comparison of average monthly income by education. \n\n Education: 1.'Below College', 2.'College', 3.'Bachelor', 4.'Master', 5.'Doctor'")


# In[ ]:


df_s=df[['YearsAtCompany', 'PercentSalaryHike']].groupby(['YearsAtCompany'], as_index=False).mean().sort_values(by='PercentSalaryHike', ascending=False) 
df_s.head()


# In[ ]:


df_s.plot(kind='scatter', x='YearsAtCompany', y='PercentSalaryHike', figsize=(10, 6), color='c')

plt.title('Comparison of PercentSalaryHike by YearsAtCompany.')
plt.xlabel('YearsAtCompany')
plt.ylabel('PercentSalaryHike')

plt.show()


# In[ ]:


for dataset in combine:
    dataset['Attrition'] = dataset['Attrition'].map( {'Yes': 1, 'No': 0} ).astype(int)

df.head()


# In[ ]:


for dataset1 in combine:
    dataset1['OverTime'] = dataset1['OverTime'].map( {'Yes': 1, 'No': 0} ).astype(int)

df.head()


# In[ ]:


for dataset2 in combine:
    dataset2['Gender'] = dataset2['Gender'].map( {'Female': 1, 'Male': 0} ).astype(int)

df.head()


# In[ ]:


sns.regplot(x='Age',y='Attrition',data=df) 
plt.ylim(0,)


# In[ ]:


df[['Age','Attrition']].corr() 
#weak linear relationship


# **Correlation and Causation.**

# In[ ]:


df[['EducationField', 'Attrition']].groupby(['EducationField'], as_index=False).mean().sort_values(by='Attrition', ascending=False)


# **Pearson Correlation**

# In[ ]:


from scipy import stats


# DistanceFromHome vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['DistanceFromHome'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# Age vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['Age'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# Education vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['Education'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# EnvironmentSatisfaction vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['EnvironmentSatisfaction'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# JobInvolvement vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['JobInvolvement'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# Overtime vs Attrition 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['OverTime'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# TotalWorkingYears vs Attrition

# In[ ]:



pearson_coef, p_value = stats.pearsonr(df['TotalWorkingYears'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# YearsSinceLastPromotion vs Attrition

# In[ ]:


#YearsSinceLastPromotion 
pearson_coef, p_value = stats.pearsonr(df['YearsSinceLastPromotion'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# YearsWithCurrManager vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['YearsWithCurrManager'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# NumCompaniesWorked vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['NumCompaniesWorked'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# YearsInCurrentRole vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['YearsInCurrentRole'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# MonthlyIncome vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['MonthlyIncome'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# JobLevel vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['JobLevel'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# Gender vs Attrition

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['Gender'], df['Attrition'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# **Data pre-processing and selection**

# In[ ]:


df= df[['JobInvolvement','Age','MonthlyIncome','YearsInCurrentRole','YearsWithCurrManager','TotalWorkingYears','HourlyRate','OverTime','Attrition']] 
df['Attrition'] = df['Attrition'].astype('int') 
df.head()


# In[ ]:


x = np.asarray(df[['JobInvolvement','MonthlyIncome','YearsInCurrentRole','YearsWithCurrManager','TotalWorkingYears','HourlyRate','OverTime','Age']]) 


# In[ ]:


y = np.asarray(df['Attrition'])


# In[ ]:


from sklearn import preprocessing 
x = preprocessing.StandardScaler().fit(x).transform(x) 
x[:5]


# **Prediction using Logistic Regression model.**

# In[ ]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4) 
print('Train set:  ', x_train.shape, y_train.shape) 
print('Test set: ', x_test.shape, y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)  
LR


# In[ ]:


yhat = LR.predict(x_test)
yhat


# **jaccard index**

# In[ ]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# **Prediction using Logistic Knn model.**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred


# In[ ]:


jaccard_similarity_score(y_test, y_pred)

