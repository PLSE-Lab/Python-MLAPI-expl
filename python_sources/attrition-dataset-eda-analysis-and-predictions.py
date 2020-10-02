#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


hra = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


hra.head()


# # EDA

# In[ ]:


sns.barplot(x = 'Attrition',y='DailyRate',data=hra,hue='Gender')


# In[ ]:


sns.swarmplot(x='Attrition',y='DistanceFromHome',hue='Gender',data=hra,split=True)


# In[ ]:


sns.boxplot(x='Attrition',y='YearsSinceLastPromotion',data=hra,hue='Department')


# In[ ]:


sns.barplot(x='YearsSinceLastPromotion',y='YearsInCurrentRole',data=hra)


# In[ ]:


sns.lmplot(x='YearsAtCompany',y='YearsWithCurrManager',data=hra)


# In[ ]:


hra.corr()


# In[ ]:


sns.heatmap(hra.corr())


# In[ ]:


df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


df.drop(columns='StandardHours',inplace=True)
df.drop(columns='EmployeeCount',inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


sns.stripplot(x='Attrition',y='HourlyRate',data=hra)


# In[ ]:


sns.stripplot(x='YearsWithCurrManager',y='StockOptionLevel',data=hra)


# In[ ]:


df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})
y = df.iloc[:,1].values
X = df.drop(columns='Attrition',inplace=True)


# In[ ]:


X = df.iloc[:,:].values


# In[ ]:


X = pd.get_dummies(data=df, columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],drop_first=True)
X.drop(columns='EmployeeNumber',axis=1,inplace=True)
X.drop(columns='Over18',axis=1,inplace=True)
X


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True)
regressor = LogisticRegression()


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


regressor.score(X_test,y_test)
regressor.score(X_train,y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred,labels=[0,1])
confusion_matrix


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Class Imbalance

# ## Using SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE

sm = SMOTE()

x1,y1 = sm.fit_sample(X,y)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state=10)


# In[ ]:


regressor = LogisticRegression()


# In[ ]:


regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))


# ## Oversampler

# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

x1,y1 = ros.fit_resample(X,y)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state=42)


# In[ ]:


reg = LogisticRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

reg.score(x_test,y_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# # Using Undersampling

# ## ClusterCentroids

# In[ ]:


from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=0)

x1,y1 = cc.fit_resample(X,y)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state=42)


# In[ ]:


reg = LogisticRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)


# In[ ]:


reg.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))


# ## Random Undersampler

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)

x1,y1 = rus.fit_resample(X,y)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state=42)


# In[ ]:


reg = LogisticRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))


# ## Near Miss

# In[ ]:


from imblearn.under_sampling import NearMiss

nm = NearMiss(version = 1)

x1,y1 = nm.fit_resample(X,y)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state=42)


# In[ ]:


reg = LogisticRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# # Final Analysis and Predictions

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


class Data_Preprocessing:
    df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    def initialize(self):
        self.df['Attrition'] = self.df['Attrition'].map({'Yes':1,'No':0})
        y = self.df['Attrition']
        X = self.df.drop(columns=['Attrition','Over18','StandardHours','EmployeeCount'],inplace=True)
        #X = self.df[['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
        #X.drop(columns = ['Over18'],axis=1,inplace=True)
        X = pd.get_dummies(data=self.df, columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],drop_first=True)
        return X,y
    def oversampling(self,X,y):
        sm = SMOTE()
        X1,y1 = sm.fit_sample(X,y)
        return X1,y1


# In[ ]:


class Train_Test:
    
    def train(self,X,y):
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        #regressor = LogisticRegression()
        #regressor = KNeighborsClassifier()
        regressor = RandomForestClassifier()
        regressor.fit(x_train,y_train)
        return regressor,x_test,y_test
    
    def test(self,x_test,y_test,regressor):
        y_pred = regressor.predict(x_test)
        acc = regressor.score(x_test,y_test)
        confusion_mat = confusion_matrix(y_test,y_pred)
        print(classification_report(y_test,y_pred))
        return acc,confusion_mat


# In[ ]:


process = Data_Preprocessing()


# In[ ]:


X,y = process.initialize()


# In[ ]:


X,y = process.oversampling(X,y)


# In[ ]:


train_test = Train_Test()
reg,x_test,y_test = train_test.train(X,y)
accuracy, confusion_matrix = train_test.test(x_test,y_test,reg)


# In[ ]:


print(confusion_matrix)

