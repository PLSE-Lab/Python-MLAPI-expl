#!/usr/bin/env python
# coding: utf-8

# In[62]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[64]:


df = pd.read_csv("../input/general_data.csv",sep=",")


# In[65]:


df.head()


# In[66]:


print(df.columns)


# ## Data Cleaning:

# In[67]:


df.isnull().any()


# In[68]:


df.fillna(0,inplace=True)


# In[69]:


#drop the useless columns:

df.drop(['EmployeeCount','EmployeeID','StandardHours'],axis=1, inplace = True)


# ## Data Visualization :

# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#  ### Find the correlation b/w all the columns:

# In[71]:


corr_cols = df[['Age','Attrition','BusinessTravel','DistanceFromHome','Education', 'EducationField','Gender', 'JobLevel', 'JobRole',
       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]


# In[72]:


corr = corr_cols.corr()
plt.figure(figsize=(16,7))
sns.heatmap(corr,annot=True)
plt.show()


# In[73]:


print(len(df))
print(len(df[df['Attrition']=='Yes']))
print(len(df[df['Attrition']=='No']))
print("percentage of yes Attrition is:",(len(df[df['Attrition']=='Yes'])/len(df))*100,"%")
print("percentage of no Attrition is:",(len(df[df['Attrition']=='No'])/len(df))*100,"%")


# In[74]:


sns.countplot(x = "Attrition",data=df)
plt.show()


# In[75]:


sns.countplot(x = "Attrition",data=df,hue="Gender")
plt.show()


# In[76]:


sns.countplot(x = "Attrition",data=df,hue="JobLevel")
plt.show()


# In[77]:


#function to creat group of ages, this helps because we have 78 differente values here
def Age(dataframe):
    dataframe.loc[dataframe['Age'] <= 30,'Age'] = 1
    dataframe.loc[(dataframe['Age'] > 30) & (dataframe['Age'] <= 40), 'Age'] = 2
    dataframe.loc[(dataframe['Age'] > 40) & (dataframe['Age'] <= 50), 'Age'] = 3
    dataframe.loc[(dataframe['Age'] > 50) & (dataframe['Age'] <= 60), 'Age'] = 4
    return dataframe

Age(df); 


# In[78]:


sns.countplot(x = "Attrition",data=df,hue="Age")
plt.show()


# ## Convert all the Categorical data into numerical data 

# In[79]:


print(df['BusinessTravel'].unique())
print(df['Department'].unique())
print(df['EducationField'].unique())
print(df['Gender'].unique())
print(df['JobRole'].unique())
print(df['MaritalStatus'].unique())
print(df['Over18'].unique())


# In[80]:


from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
df['BusinessTravel'] = labelEncoder_X.fit_transform(df['BusinessTravel'])
df['Department'] = labelEncoder_X.fit_transform(df['Department'])
df['EducationField'] = labelEncoder_X.fit_transform(df['EducationField'])
df['Gender'] = labelEncoder_X.fit_transform(df['Gender'])
df['JobRole'] = labelEncoder_X.fit_transform(df['JobRole'])
df['MaritalStatus'] = labelEncoder_X.fit_transform(df['MaritalStatus'])
df['Over18'] = labelEncoder_X.fit_transform(df['Over18'])


# In[81]:


#Attriton is dependent var
from sklearn.preprocessing import LabelEncoder
label_encoder_y=LabelEncoder()
df['Attrition']=label_encoder_y.fit_transform(df['Attrition'])


# In[82]:


df.head()


# In[83]:


corr_cols = df[['Age','Attrition','BusinessTravel','DistanceFromHome','Education', 'EducationField','Gender', 'JobLevel', 'JobRole',
       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]


# In[84]:


corr = corr_cols.corr()
plt.figure(figsize=(18,7))
sns.heatmap(corr, annot = True)
plt.show()


# # Split data into training and Testing set:

# ### Choose dependent and independent var:
#  here dependent var is **Attrition** and rest of the var are indepdent var.

# In[85]:


y = df['Attrition']
x = df.drop('Attrition', axis = 1)


# In[86]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)


# In[87]:


from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)


# In[88]:


#import some comman libs:
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


# # Logistic Regression:

# In[89]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[90]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




