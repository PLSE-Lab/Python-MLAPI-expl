#!/usr/bin/env python
# coding: utf-8

# # IBM_HR_Analytics_Employee_Attrition_Performance 

# ### The dataset is about employee attrition. This analysis can discover if any particular factors or patterns that lead to attrition. If so, employers can take certain precausion to prevent attrition which in employer of view, employee attrition is a loss to company, in both monetary and non-monetary. 

# #### import modules and data input 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# In[ ]:


data=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


data.head()


# Attrition will be the label column and remaining will be feature

# In[ ]:


data.describe()


# EmployeeCount can be deleted as its value always equal 1

# In[ ]:


del data['EmployeeCount']


# #### Check if any missing values

# In[ ]:


data.isnull().any()


# There is no missing value

# #### Data exploration

# In[ ]:


data.shape


# There are 1470 records and 35 variables

# In[ ]:


data.groupby('Attrition').size()


# There are 237 attritions in dataset 

# Let's change Attrition to binary: 1 is Yes , 0 is No 

# In[ ]:


data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)


# There are many reasons causing attrition, from personal related (eg age, family) to job related (eg salary, bored in position)
# <br> Before digging in deep, let's have a prelimary finding first

# First is about age (Maybe old enough to retire)

# In[ ]:


sns.boxplot(x='Attrition',y='Age',data=data)


# In average, attrited employees are younger than non-attrited employees

# Second is salary (Unsatisfactory in salary??)(DailyRate in data) 

# In[ ]:


sns.boxplot(x='Attrition',y='DailyRate',data=data)


# There is less difference for daily rate between attrited and non-attrited employees than age.
# <br>Maybe wage is not as an important factor than expected

# Next is duration of work in company (Bored, no further excitment from work?)

# In[ ]:


sns.boxplot(x='Attrition',y='YearsAtCompany',data=data)


# There are many extreme cases in both attrited and non-attrited employees. Hard to determine if duration of work is related at this moment

# #### Data transformation

# There are three text categories, BusinessTravel, Department and EducationField. Moreover, Education, EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, PerformanceRating, RelationshipSatisfaction and WorkLifeBalance are also categorical data. First need to separate into dummy variables

# In[ ]:


num_cat=['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction' , 'WorkLifeBalance']
for i in num_cat:
    data[i]=data[i].astype('category')


# In[ ]:


data=pd.get_dummies(data)


# In[ ]:


data.info()


# From the last result, apparently there is only one variable for over18. Can double check and if so then it can be deleted

# In[ ]:


data['Age'].describe()


# All employees are over 18. So it can be deleted 

# In[ ]:


del data['Over18_Y']


# In[ ]:


data.shape


# After transformation, there are 74 variables 

# #### Modelling

# First is to separate feature set and label set

# In[ ]:


X=data[data.columns.difference(['Attrition'])]
y=data['Attrition']


# Second is to separate training set and test set 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=42)


# Then is to standardize numeric variables. Back to previous finding, attrition is also numeric variable. Therefore need to special handle

# In[ ]:


numeric_variables = list(data.select_dtypes(include='int64').columns.values)


# In[ ]:


numeric_variables.remove('Attrition')


# In[ ]:


numeric_variables


# In[ ]:


#First is to reset index for X_train and X_test
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
#Separate into two dataframes for numeric and non-numeric variables 
X_train_num=X_train[numeric_variables]
X_train_nnum=X_train[X_train.columns.difference(numeric_variables)]
X_test_num=X_test[numeric_variables]
X_test_nnum=X_test[X_train.columns.difference(numeric_variables)]
#Set standard scaler 
scaler=StandardScaler()
#Fit and transform training set 
X_train_scaled=scaler.fit_transform(X_train_num)
X_train_scaled=pd.DataFrame(data=X_train_scaled,columns=X_train_num.columns)
X_train_scaled=pd.concat([X_train_scaled,X_train_nnum],axis=1)
#Transform training set
X_test_scaled=scaler.transform(X_test_num)
X_test_scaled=pd.DataFrame(data=X_test_scaled,columns=X_test_num.columns)
X_test_scaled=pd.concat([X_test_scaled,X_test_nnum],axis=1)


# #### 1. K Nearest Neighbors
# <br>K Nearest Neighbors is a simple non-parametric method for classification. The label will be determined by neighbors

# In[ ]:


knn=KNeighborsClassifier()
knn.fit(X_train_scaled,y_train)


# In[ ]:


knn.score(X_test_scaled,y_test)


# K Nearest Neighbors has 87% of accuracy. Let's have a look for confusion for better understanding

# In[ ]:


y_predict = knn.predict(X_test_scaled)
confusion_matrix(y_test,y_predict)


# Because attrition is not balanced, by just looking at score will cause bise on performance
# <br>Apparently KNN wrongly predicts attrition=1 as 0 frequently. Only 6 attrition records are correctly predicted. 
# <br>A very bad prediction model 

# #### 2. LogisticRegression

# In[ ]:


logis=LogisticRegression()
logis.fit(X_train_scaled,y_train)
logis.score(X_test_scaled,y_test)


# In[ ]:


y_predict = logis.predict(X_test_scaled)
confusion_matrix(y_test,y_predict)


# Better than KNN as there are 17 attrited records are correctly predicted. 
# And score is also higher than KNN

# In[ ]:


logis.coef_


# In[ ]:


print('The most positive influent coefficient is {0}, with value equal to {1}'.format(X_test_scaled.columns[np.argmax(logis.coef_)],logis.coef_.max()))


# In[ ]:


print('The most negative influent coefficient is {0}, with value equal to {1}'.format(X_test_scaled.columns[np.argmin(logis.coef_)],logis.coef_.min()))


# Apparently if your employees give you 'Low' in Job Satisfcation, this is already a signal that he or she may leave company
# <br>And definitely employees enjoy no overwork, this greatly reduce chance of attrition.

# In[ ]:


pd.crosstab(data.JobInvolvement_1,data.Attrition)


# Over one third of employees choosing JobInvolvement=1 were attrited. This is a very good sign to predict employee leaving 

# In[ ]:


pd.crosstab(data.OverTime_No,data.Attrition)


# No overtime can greatly reduce chance of attrition by two third. Another good policy to be enforced for reducing attrition
