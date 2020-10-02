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


# **1. Library packages for data visualization and understanding data**

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[ ]:


#Read Data
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.info()
data.columns


# In[ ]:


# Change numeric data into catergoric data
data['cat_Education'] = pd.cut(data['Education'], 5, labels=['Below College','College','Bachelor','Master','Doctor'])
data['cat_EnvironmentSatisfaction'] = pd.cut(data['EnvironmentSatisfaction'], 4, labels=['Low','Medium','High','Very High'])
data['cat_JobInvolvement'] = pd.cut(data['JobInvolvement'], 4, labels=['Low','Medium','High','Very High'])
data['cat_JobSatisfaction'] = pd.cut(data['JobSatisfaction'], 4, labels=['Low','Medium','High','Very High'])
data['cat_PerformanceRating'] = pd.cut(data['PerformanceRating'], 4, labels=['Low','Good','Excellent','Outstanding'])
data['cat_RelationshipSatisfaction'] = pd.cut(data['RelationshipSatisfaction'], 4, labels=['Low','Medium','High','Very High'])
data['cat_WorkLifeBalance'] = pd.cut(data['WorkLifeBalance'], 4, labels=['Bad','Good','Better','Best'])
data['cat_Age'] = pd.cut(data['Age'], 4, labels=['Young', 'Middle', 'Senior','Super Senior'])
data['num_Attrition']=pd.get_dummies(data.Attrition, drop_first = True)


# **2. Data Visulization:**
# - Scatter plot
# - Histogram
# - Box plot
# - Count plot
# - Bar plot
# - Dist plot
# - Pie Chart
# - Correlation heat map 

# In[ ]:


#scatter plot
data.plot(kind='scatter', x='Age', y='DailyRate',alpha = 0.5,color = 'red')
plt.xlabel('Age', fontsize=16)              # label = name of label
plt.ylabel('DailyRate', fontsize=16)
plt.title('Age vs DailyRate Scatter Plot', fontsize=20)            # title = title of plot


# In[ ]:


#histogram
data.TotalWorkingYears.plot(kind = 'hist',bins = 10,figsize = (15,15))


# In[ ]:


#boxlot
sns.boxplot(data['Gender'], data['MonthlyIncome'])
plt.title('MonthlyIncome vs Gender Box Plot', fontsize=20)      
plt.xlabel('MonthlyIncome', fontsize=16)
plt.ylabel('Gender', fontsize=16)
plt.show()


# In[ ]:


#count plot
sns.countplot(data.JobLevel)
plt.title('JobLevel Count Plot', fontsize=20)      
plt.xlabel('JobLevel', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# In[ ]:


# BarPlot
sns.barplot(x = 'Attrition', y = 'DistanceFromHome', data = data)


# In[ ]:


#sns.distplot(data['Age'])
sns.distplot(data.Age, kde=True, label='YearsInCurrentRole', hist_kws={"histtype": "step", "linewidth": 3,
                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})

plt.title('Age Distribution plot', fontsize=20)      
plt.show()


# In[ ]:


# pie chart of workers
labels = ['Male', 'Female']
sizes = [data['Gender'].value_counts()[0],
         data['Gender'].value_counts()[1]
        ]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.title('Gender Pie Chart', fontsize=20)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,12))
sns.distplot(data.YearsWithCurrManager, hist=False, kde=True, label='YearsWithCurrManager', hist_kws={"histtype": "step", "linewidth": 3,
                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})
sns.distplot(data.YearsAtCompany, hist=False, kde=True, label='YearsAtCompany', hist_kws={"histtype": "step", "linewidth": 3,
                  "alpha": 1, "color": sns.xkcd_rgb["dark blue green"]})
sns.distplot(data.YearsInCurrentRole, hist=False, kde=True, label='YearsInCurrentRole', hist_kws={"histtype": "step", "linewidth": 3,
                  "alpha": 1, "color": sns.xkcd_rgb["fuchsia"]})
plt.suptitle('Total Years: Current Manager, Role and At Company', size=22, x=0.5, y=0.94)
plt.xlabel('Years', size=16)
plt.ylabel('Count', size=16)
plt.legend(prop={'size':26}, loc=1)
plt.show();


# In[ ]:


cont_col= ['DistanceFromHome', 'PerformanceRating','EnvironmentSatisfaction','Attrition','YearsWithCurrManager','NumCompaniesWorked']
sns.pairplot(data[cont_col], kind="reg", diag_kind = "kde" , hue = 'Attrition' )
plt.show()


# In[ ]:


#Plot a correlation map for all numeric variables
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.10, fmt= '.1f',ax=ax)


# **Inferrence of above charts:**
# - Inference here is that Daily rate is highly dense between ages 25 to 40
# - totalWorkingYears highest in 5 to 10 Years
# - average and range of salary is higher in Female compared to Male associates
# - largest population is between Joblevel 1 and 2
# - People are more likely to quit When 'DistanceFromHome' is between 9 to 12
# - Age is Normally distributed and Mean age is about 35 Years
# - Proportion of male is higher than female associates
# - Employees falling under the below categories are more likely to quit PerformanceRating' between 3.0 & 4.0 and 'DistanceFromHome' between 10 & 15'YearsWithCurrManager' less than 5
# - High correlations between variable (>=0.7) as per below, Montly Income to Job Level (1.0), Montly Income to Total working years (0.8), Performance Rating to Performance Salary Hike (0.8), Years in Company to Years with current manager (0.8), Years in Company to Years in current role (0.8); Job Leve and Monthly Income to Total working hours (0.8); Age to Total working hours (0.7); Years with current manager to Years in current role (0.7)

# **3. Create train and test splits**
# #Selecting numeric paremeters for Feature Engineering

# In[ ]:


df=data[['JobLevel','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','WorkLifeBalance','Attrition']]
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)


# In[ ]:


target_name = 'Attrition'
X = df.drop('Attrition', axis=1)
y=df[target_name]
X_train,X_test,y_train,t_test=train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
X_train.head()

