#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


Data = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


Data.head()


# In[ ]:


Data.tail()


# In[ ]:


Data.describe()


# In[ ]:


Data.info()


# In[ ]:


columns = Data.columns.tolist()


# In[ ]:


columns


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools


# In[ ]:


categorical=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','sales','salary']


# In[ ]:


fig = plt.subplots(figsize = (20,20))  ## plots sub-plots of size 20x20
length = len(categorical)
for i,j in itertools.zip_longest(categorical,range(length)): ## Similar to indexing the categorical data 
    ### i -> categorical , j -> range(length)
    plt.subplot(np.ceil(length/2),2,j+1)  ## (j+1) is given since indexing starts from 0 and we need to assign graph numbers from 1
    plt.subplots_adjust(hspace = 0.5)  ## Adjusting horizontal space by a factor 0.5 to get better visibility
    sns.countplot(x=i, data=Data)  ### (sns.countplot) gives better visualization in bar graph format
plt.subplot(np.ceil(length/2),2,6)  ## Re-aligning the labels of 6th sub-plot
plt.xticks(rotation=90) ## Re-align done by 90 degrees


# In[ ]:


len(Data[Data['left']==1]) ### Number of employees who left the company


# In[ ]:


len(Data[Data['promotion_last_5years']==1])  ## Promotions in the last 5 yrs


# In[ ]:


categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary']

## Omitting the (left) column from the previous (categorical) list


# In[ ]:


fig=plt.subplots(figsize=(20,20))
length = len(categorical)
for i,j in itertools.zip_longest(categorical,range(length)):
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=0.5)
    sns.countplot(x=i,data = Data,hue="left")  ### In left, if 0 -> sticking on with the same company, 1 -> left the company
    
plt.subplot(np.ceil(length/2),2,5)
plt.xticks(rotation=90)


# # Observations : 
# 
# * Those who were promoted in last 5 years didn't leave the company
# * Those who have spent 5 years in company are more in proportion compared to people staying back
# * More than 6 years, people dont leave company anymore
# * People who have done more than 7 projects have left the company
# * More than 6 years experienced are not leaving company & more than 7 projects completed are not staying back. So, folks who have done many projects faster are leaving company

# In[ ]:


## Calculating Proportions (Percentage) of people who left in each category

categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary']
length = len(categorical)

fig=plt.subplots(figsize=(20,20))
for i,j in itertools.zip_longest(categorical,range(length)):
    
     # only counting the number who left 
    Proportion_of_data = Data.groupby([i])['left'].agg(lambda x: (x==1).sum()).reset_index()
    
    # Counting the total number 
    Proportion_of_data1=Data.groupby([i])['left'].count().reset_index() 
    
    # mergeing two data frames
    Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on=i)
    
    # Now we will calculate the % of employee who left category wise
    Proportion_of_data2["Proportion"]=(Proportion_of_data2['left_x']/Proportion_of_data2['left_y'])*100 
    
    #sorting by percentage
    Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)

    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    sns.barplot(x=i,y='Proportion',data=Proportion_of_data2)
    plt.xticks(rotation=90)
    plt.title("percentage of employee who left")
    plt.ylabel('Percentage')


# In[ ]:


Proportion_of_data  

### The number of people who left


# In[ ]:


Proportion_of_data1

### Total number of people


# In[ ]:


Proportion_of_data2

### Percentage of people based on high, medium and low salaries


# ## Feature Engineering
# 
# * To find the important features for prediction
# * Using Random Forests

# In[ ]:


# Let's plot the correlation Matrix

corr = Data.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cbar=True, cmap='coolwarm')
plt.xticks(rotation=90)


# In[ ]:


# For changing categorical variable into int
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score 
le=LabelEncoder()
Data['salary']=le.fit_transform(Data['salary'])
Data['sales']=le.fit_transform(Data['sales'])


# In[ ]:


# we can select importance features by using Randomforest Classifier
from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100)

#Using all features except for left
feature_var = Data.ix[:,Data.columns != "left"]

#Target feature
pred_var = Data.ix[:,Data.columns=='left']

#Training
model.fit(feature_var,pred_var.values.ravel())


# In[ ]:


featimp = pd.Series(model.feature_importances_,index=feature_var.columns).sort_values(ascending=False)
print(featimp)

## Here, notice that satisfaction level , time spent in company and number of projects are three top most criteria.


# In[ ]:


## ML Models 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.svm import SVC


# In[ ]:


def Classification_model(model,Data,x,y): # here x is the variable which are used for prediction
    # y is the prediction variable
    train,test = train_test_split(Data,test_size= 0.33)
    train_x = Data.ix[train.index,x] # Data for training only with features
    train_y = Data.ix[train.index,y] # Data for training only with predcition variable
    test_x = Data.ix[test.index,x] 
    test_y = Data.ix[test.index,y]
    
    model.fit(train_x,train_y.values.ravel())
    
    pred=model.predict(test_x)
    
    accuracy=accuracy_score(test_y,pred)
    return accuracy


# In[ ]:


All_features=['satisfaction_level',
'number_project',
'time_spend_company',
'average_montly_hours',
'last_evaluation',
'sales',
'salary',
'Work_accident',       
'promotion_last_5years']


# In[ ]:


Important_features = ['satisfaction_level',
'number_project',
'time_spend_company',
'average_montly_hours',
'last_evaluation']

#Target Variable
Pred_var = ["left"]


# In[ ]:


### Checking the accuracy with all the features

models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]
Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]
Model_Accuracy = []
for model in Classification_models:
    Accuracy=Classification_model(model,Data,All_features,Pred_var)
    Model_Accuracy.append(Accuracy)


# In[ ]:


Model_Accuracy


# In[ ]:


### Checking the accuracy with only important features

models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]
Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]
Model_Accuracy = []
for model in Classification_models:
    Accuracy=Classification_model(model,Data,Important_features,Pred_var)
    Model_Accuracy.append(Accuracy)


# In[ ]:


Model_Accuracy


# In[ ]:




