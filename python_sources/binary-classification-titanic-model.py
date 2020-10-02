#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train["Flag"]='train'
test["Flag"]='test'
data=pd.concat([train,test],axis=0)


# So we have read the data and added flags to both the train and test data.
# Next we have combined the test and train data together. 
# 
# The next important step is Analyzing our data and feature engineering.
# 
# Some of the initial steps in Data Exploration process are follows:-
# We will look into some of these.
# 
# 1. Removing unwanted columns.
# 2. Presence of missing values.
# 3.Presence of outliers.
# 4 Categorical variables and convert them to dummy variables or map them into required form 
# 5. Creating Derived Variables
# 6. Variable Transformation

# Removing columns needs to be done carefully...We should try not to remove some important feature.
# 
# 1. Columns which contain redundant information can be removed. However this step must be done carefully.Correlation can be checked for such features and needs to be carefully done.
# 2. Any randomly generated fields can be removed. For e.g here Ticket is one such field.Ticket column has no such information as such.
# 3. Any field which leaks data from future. For e.g if there was a field say : Compensation to families was a feature in this dataset which had yes or no answer based on whether there were deaths in the family it had to be removed as it is based on "Survived Column" of our dataset.
# 
# But there is no such column at present.
# 
# Based on above I have decided to drop off columns: Survived(Predictor column), Ticket (randomly generated field),PassengerId(randomly generated),Name
# 

# In[ ]:


columns_to_drop=["PassengerId","Name","Ticket"]
data.drop(columns_to_drop,axis=1,inplace=True)

#However we need to look into Fare,SibSp,Parch,Cabin features as well


# **DATA VISUALIZATION**
# 
# It is always good to understand your data through some visualizations.Better to think and draw your graphs.
# 
# Lets have a look at some graphs.

# I first need to understand how my predictor variable is distributed. So I am using a simple bar graph for this purpose.So we know the data is not imbalanced.

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,8))
plt.subplot2grid((3,4),(0,0))
train["Survived"].value_counts().plot(kind="bar")


# In[ ]:


def val_count_plot(feature,i,j):
    plt.subplot2grid((2,2),(i,j))
    train[feature].value_counts(normalize=True).plot(kind='bar')

val_count_plot('Survived',0,0)
val_count_plot('Sex',0,1)
val_count_plot('Embarked',1,0)
val_count_plot('Pclass',1,1)


# Lets look at hoe the age and sex variable affected the survival of a passenger.

# Lets have a look at how these features are related to their survival feature.

# In[ ]:


plt.subplot2grid((2,2),(0,0))
train[(train['Sex']=='male')&(train['Pclass']==1)]['Survived'].value_counts(normalize=True).plot(kind='bar')
plt.title('Survived male')

# SO it seems the male survived less as compared to females.


# In[ ]:


data.head()


# 

# The next thing we will check for is missing values. These missing values can then be imputed using one of the following methods:-
# 
# 1. Deletion:- 
# If a variable is missing in most of the samples point and is not conveying any useful info we may consider removing that variable. 
# Also those rows which have most of the missing details can be removed but this decreases the sample size.
# 
# 2. Mode/Median/Mean 
# The missing values may be imputed with the mode/median/ mean of the remaining values.
# But before that always pay attention as to why the data is missing. 
# 
# 
# This can be done in two ways:
# Either we replace the missing values by non missing mean/median/mode.
# or we can choose another feature in the data available and based on that try to predict the missing values.
# 
# When to use median?- Though the way you handle the msiing data depends completely on the data. But we prefer replacing by median when there are cases of outliers.
# 
# Let's analyze our data for missing values:-
# 

# In[ ]:


print(data.isnull().values.any())


# In[ ]:


data.apply(lambda x: sum(x.isnull()))


# It can be seen values are missing in Age, Cabin, Embarked, Fare.We need to handle these.

# In[ ]:


#Age 
#Since age seems to be an important feature because we need to decide whether 
# the passsenger survived or not. And age is an important feature.

# I am replacing age with mean of ages.

data["Age"].fillna(data["Age"].mean(),inplace=True)

#Since two value is missing in embarked I am replacing it with mode
data["Embarked"].fillna(data["Embarked"].mode,inplace=True)
print(data.apply(lambda x:sum(x.isnull())))

# but in fare I am not thinking of applying mean because the fare will depend on passenger class,
# So any passenger with higher class will be having a higher fare.


#Introduction to Ensembling/Stacking in Python kernel has many interesting steps for data exploration .

# I have considered the idea of creating another field from fare from this kernel using qcut
data['Fare'] = data['Fare'].fillna(train['Fare'].median())
data['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# This divides into 4 ranges as can be seen 
print(data['CategoricalFare'].head(10))

# I can map the fares into one of these ranges


# Next only Cabin is the feature that is left 

# if the cabin is present I am mapping that to 1 else 0

data["Has_Cabin"]=data["Cabin"].apply(lambda x: 1 if type(x) == float else 0)
print(data["Has_Cabin"].head(5))


# In[ ]:


data.drop(["CategoricalFare","Cabin"],axis=1,inplace=True)


# Next step is handling all categorical features.
# In this kernel I am using mapping method, However you can also use Label encoder/ get_dummy method for converting to non categorical formats.
# 
# Next:
# one hot encoding and label encoding- Difference between two 
# get dummies
# 
# Detecting outliers
# creating derived variables
# 
# variable transformation 
# 
# read top 5 kernels completely 
# separate train and test data 
# 
# run models
# cross validation 

# 

# Converting all categorical features to numerical for running the models.
# I will be first using the mapping technique to convert the features into categorical.
# But before that , have a look at which features to be converted

# In[ ]:


print(data.dtypes)# helpd in identifying the types of data in the dataset, the object is data type is categorical
# to get the names of the categorical columns in the dataset we can use
data.select_dtypes(include='object').columns # so flag , sex, embarked are the categorical data.
# to convert them into numerical data I am using mapping method


# In[ ]:


# 1.MAPPING METHOD
#dict={"male":0,1:"female"}
#data["Sex"].map(dict)
#print(data["Sex"].value_counts())
# similiarly for column Embarked the values can be mapped


# In[ ]:


#2. using the Label Encoder
# this method will transform non-numerical labels to numerical labels.
# To underatnd this consider a category say region:- which may be east/west/north/south.
#Label encoder will assign 4 different numerical values to each one of the regions.
# this is a disadvantage esp in case of nominal categorical variables where there are no levels.


# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#label_enc=LabelEncoder()
#data['Sex_l_enc']=label_enc.fit_transform(data['Sex'])
#data['Sex_l_enc'].value_counts()
# so the male and female category has been encoded.


# To overcome the disadvantage of Label Encoder , one can use another library of scikit learn -which is ONE HOT ENCODER- or get dummies method of pandas.

# In[ ]:


data=pd.get_dummies(data,columns=['Sex'])
data.drop(['Embarked'],axis=1,inplace=True)
# so now we have our dataset , next we can separate it back to training data and test data
# we have a flag column which will help in this.
train_new = data[data.Flag=='train']
test_new=data[data.Flag=='test']


# In[ ]:


y=train['Survived']# output data
train_new.drop(['Flag','Survived'],axis=1,inplace=True)
test_new.drop(['Flag','Survived'],axis=1,inplace=True)
print(train_new.head(10))
print(train_new.columns)


# 

# In[ ]:


X=train_new
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=7)

decision_model=DecisionTreeClassifier()
decision_model.fit(X_train,y_train)
dt_predict= decision_model.predict(X_test)
# accuracy 
print("Accuracy of decision tree :",metrics.accuracy_score(y_test,dt_predict))


# 

# In[ ]:


# using LOgistic Regression 
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
logistic_predict = logmodel.predict(X_test)
print("Accuracy of logistic regression is:", metrics.accuracy_score(y_test,logistic_predict))

# using Random Forests 
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()  
random_forest.fit(X_train, y_train)  
randomf_predict = random_forest.predict(X_test)  
print("Accuracy of RandomForest is:", metrics.accuracy_score(y_test,randomf_predict))

# using XGboost 
from xgboost import XGBClassifier

XGB_model = XGBClassifier()
XGB_model.fit(X_train, y_train)
XGB_predict= XGB_model.predict(X_test)
print("Accuracy of XGB is:", metrics.accuracy_score(y_test,XGB_predict))


# So the XG boost works the best in this case , Logistic Regression and Random Forests have similiar accuracy.
# Decision trees didn't work well for this case.

# Though there is more scope of improvement in this kernel , it will surely help in giving a start to the Titanic Dataset problem. The trained model can then be applied to predict the test data scores.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




