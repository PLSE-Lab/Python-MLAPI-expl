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
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #  Data  Import

# In[ ]:


train_data=pd.read_csv('/kaggle/input/titanic/train.csv') # training data
test_data=pd.read_csv('/kaggle/input/titanic/test.csv') # test data


# # **Understand the Data**

# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.describe(include='all')


# In[ ]:


test_data.describe(include='all')


# In[ ]:


train_data.nunique()


# * Survived column in train_data is output column
# * train_data has 891 samples, test_data has 418 samples
# * train_data has 11 features, test_data has 11 features, same as training data
# * Age, Cabin feature has missing values in training and train data
# * Embarked feature has missing value in trianing data
# * Fare feature has missing value in test data
# * Sex, Name Ticket, Cabin, Embarked are object data types, will be an issue in model fitting
# 

# # **Visualize the data distribution**

# In[ ]:


train_data.hist(color='Blue',bins=20, figsize=(14,14))
plt.show()


# In[ ]:


train_data.groupby(['Pclass']).Survived.mean().sort_values(ascending=False)


# Pclass is a feature to be included: Class 1 tends to survive

# In[ ]:


train_data.groupby(['Sex']).Survived.mean().sort_values(ascending=False)


# In[ ]:


train_data.groupby(['Embarked']).Survived.mean().sort_values(ascending=False)


# Embarked is a feature to be included: Higher chance of survival form C

# In[ ]:


train_data.groupby(['Parch']).Survived.mean().sort_values(ascending=False)


# Parch is a feature to be included:most passengers do not have parents or children and have low survival rate

# In[ ]:


train_data.groupby(['SibSp']).Survived.mean().sort_values(ascending=False)


# SibSp is a feature to be included:50% of passengers do not have sibblings or spouse and have low survival rate

# In[ ]:


#visualize the data shown above in barplots
sns.barplot(x="Pclass", y="Survived", data=train_data)
plt.show()
sns.barplot(x="Sex", y="Survived", data=train_data)
plt.show()
sns.barplot(x="Embarked", y="Survived", data=train_data)
plt.show()
sns.barplot(x="Parch", y="Survived", data=train_data)
plt.show()
sns.barplot(x="SibSp",y="Survived",data=train_data)
plt.show()


# In[ ]:


sns.swarmplot(x=train_data['Survived'],y=train_data['Age'])


#  Age is a feature to be included: Age0-10 and 80+ have higher survival rate, age between 40-60 have a 50% survival rate and 20-40 yrs age group have lowest survival rate
#  We can consider grouping Age into such bands if necessary

# In[ ]:


sns.swarmplot(x=train_data['Survived'],y=train_data['Fare'])


# Fare is a feature to be considered: higher fares have higher survival rates

# In[ ]:


cormat=train_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cormat,annot=True, linewidth=1, linecolor='black',cbar=True, square=True)


# * SibSp and Parch have high correlation
# * Pclass has high correlation with Age abd Fare
# * Survival has high correlation with Fare and Pclass
# 
# Assumptions:
# * This means better economic status ppl might have higher survival rate
# * Females have higher survival rate
# * 0-10 and 80+  age group ppl might have higher survival rate
# * Passengers travelling with 1-2 family members like spouse/children have higher survival rate
# 
# * Keep Name, Sex, Embarked,Age, Fare, Pclass, Parch, SibSp column in train and test data

# In[ ]:


# Remove Cabin, PassengerId, Ticket column in train data

train_data=train_data.drop(['Ticket','Cabin','PassengerId'],axis=1)
train_data.info()


# Do not remove PassengerId in test data and it is required for submission

# In[ ]:


# Remove Cabin, Ticket column in test data. Do not remove PassengerId as it is required for submission
test_data=test_data.drop(['Ticket','Cabin'],axis=1)
test_data.info()


# # Plan:
# 1. Map female to 1 and male to 0
# 2. Fill missing embarked value and map S to 1 C to 2 and Q to 3
# 3. Fill in missing Age values
# 4. Fill in missing Fare value
# 5. Find the Title of names and store in new column, drop Name column
# 6. Combine Sibsp and Parch to family size
# 7. Introduce Ageband to get better correlation with survival column
# 8. Introduce Fareband to get better correlation with survival column

# # 1. convert female to 1 and male to 0 value in the column Sex in train and test data

# In[ ]:



combine=[train_data,test_data]
for ds in combine:
    ds['Sex']=ds.Sex.map({'male':0,'female':1})


# In[ ]:


train_data.groupby(['Sex']).Survived.mean().sort_values(ascending=False)


# # 2. Working on Embarked Column and converting it into int and fill the missing value
# Find the mode of embarked and replace nan value with mode in Embarked column

# In[ ]:


emb_mod=train_data.Embarked.mode()[0]
combine=[train_data,test_data]
for ds in combine:
    ds['Embarked']=ds.Embarked.fillna(emb_mod)
    ds['Embarked']=ds.Embarked.map({'S': 1, 'C': 2, 'Q': 3} )


# In[ ]:


train_data.groupby(['Embarked']).Survived.mean().sort_values(ascending=False)


# # 3. Fill in missing Age value in train and test data with mean value of train data

# In[ ]:



age_mean=train_data.Age.mean()
print(age_mean)


# In[ ]:


# Find the mean age of Pclass and sex will result in 0.4% inc in accuracy
a=np.zeros((3,2))
for i in range(3):
    for j in range(2):
        a[i,j]=train_data.loc[(train_data.Pclass==i+1) & (train_data.Sex==j),'Age'].mean()
print(a)


# In[ ]:


for ds in combine:
    for i in range(3):
        for j in range(2):
            ds.loc[(ds.Age.isnull()) & (ds.Pclass==i+1) & (ds.Sex==j),'Age']=a[i,j]


# In[ ]:


train_data.loc[[5,17,19],:]


# # 4. Fill in missing Fare value in train and test data with mean value of train data

# In[ ]:



Fare_mean=train_data.Fare.mean()
print(Fare_mean)


# In[ ]:


for ds in combine:
    ds['Fare']=ds.Fare.fillna(Fare_mean)


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# Start adding more features will result in 3% inc in model accuracy 

# # 5. find titles in ames, search for ending with'.' and starting with caps for train and test data

# In[ ]:


for ds in combine:
    ds['Title']=ds['Name'].str.extract('([A-Z][a-z]+)\.',expand=False)


# In[ ]:


train_data.head()


# In[ ]:


print(train_data['Title'].value_counts())


# In[ ]:


#Convert and group titles in train and test data
for ds in combine:
    ds['Title']=ds['Title'].replace(['Ms','Mlle'],'Miss')
    ds['Title']=ds['Title'].replace('Mme','Mrs')
    ds['Title']=ds['Title'].replace(['Don','Sir' ,'Lady','Countess','Jonkheer'],'Nobles') 
    ds['Title']=ds['Title'].replace(['Dr','Capt','Col','Major','Rev'],'Others') 
    
    

print(train_data['Title'].value_counts())


# In[ ]:


train_data.groupby('Title').Survived.mean().sort_values(ascending=False)


# In[ ]:


for ds in combine:
    ds['Title']=ds['Title'].map({'Mrs':3,'Miss':2,'Mr':1,'Master':4, 'Others':5,'Nobles':6})
    ds['Title']=ds['Title'].fillna(0)


# In[ ]:


train_data.head()


# In[ ]:


train_data['Title'].unique()


# In[ ]:


train_data.groupby('Title').Survived.mean().sort_values(ascending=False)


# In[ ]:


#Drop name feature in train and test data
train_data=train_data.drop(['Name'],axis=1)
test_data=test_data.drop(['Name'],axis=1)


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# # 6. Explore if family size is a feature

# In[ ]:


# Family size is 1 if alone 
combine=[train_data, test_data]
for ds in combine:
    ds['Fam_Size']=ds.Parch+ds.SibSp+1


# In[ ]:


train_data.groupby(['Fam_Size']).Survived.mean()


# In[ ]:


for ds in combine:
    ds['Single']=0
    ds.loc[ds.Fam_Size==1,'Single']=1


# In[ ]:


train_data.groupby(['Single']).Survived.mean()


# In[ ]:


# Plot heatmap to visual the factors
cormat_2=train_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cormat_2,annot=True, linewidth=1, linecolor='black',cbar=True, square=True)


# Keep Single column and get rid of SibSP, Parch, Fam_Size since Single column has higher correlation with survived column

# In[ ]:



train_data=train_data.drop(['SibSp','Parch','Fam_Size'],axis=1)
test_data=test_data.drop(['SibSp','Parch','Fam_Size'],axis=1)


# # 7. Introduce Ageband to get better correlation with survival column

# In[ ]:



train_data['Age_band']=pd.cut(train_data.Age, bins=5)
train_data.groupby(['Age_band']).Survived.mean()


# In[ ]:


combine=[train_data, test_data]
for ds in combine:
    ds.loc[(ds.Age<=16.336),'Age']=1
    ds.loc[(ds.Age>16.336) & (ds.Age<=32.252),'Age']=2
    ds.loc[(ds.Age>32.252) & (ds.Age<=48.168),'Age']=3
    ds.loc[(ds.Age>48.168) & (ds.Age<=64.084),'Age']=4
    ds.loc[(ds.Age>64.084) ,'Age']=5


# In[ ]:


train_data=train_data.drop(['Age_band'],axis=1)


# In[ ]:


test_data.Age.unique()


# In[ ]:


train_data.groupby(['Pclass','Age']).Survived.mean()


# In[ ]:


combine=[train_data, test_data]
for ds in combine:
    ds['Age_Pclass']=ds.Age*ds.Pclass


# In[ ]:


train_data.groupby('Age_Pclass').Survived.mean()


# # 8. Introduce Fareband to get better correlation with survival column

# In[ ]:



train_data['Fare_band']=pd.qcut(train_data.Fare,4)
train_data.groupby('Fare_band').Survived.mean()


# if cut is used,the bands do not represent ideal bands as fare si skewed. hence, use qcut here as it helps in dividing fare feature but not for age feature as we might loose certain variations in survival rate.
# Keeping fareband has higher correlation with survival rate. keep fare band and drop continuous fare data 

# In[ ]:


combine=[train_data, test_data]
for ds in combine:
    ds.loc[(ds.Fare<=7.91),'Fare']=1
    ds.loc[(ds.Fare>7.91) & (ds.Fare<=14.454),'Fare']=2
    ds.loc[(ds.Fare>14.454) & (ds.Fare<=31.0),'Fare']=3
    ds.loc[(ds.Fare>31.0),'Fare']=4


# In[ ]:


train_data=train_data.drop(['Fare_band'],axis=1)


# In[ ]:


# Plot heatmap to visual the factors
cormat_2=train_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cormat_2,annot=True, linewidth=1, linecolor='black',cbar=True, square=True)


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# # Model Selection

# In[ ]:


# Obtain train and test data
X_train=train_data.drop(['Survived'],axis=1)
y_train=train_data['Survived']
X_test=test_data.drop(['PassengerId'],axis=1)


# In[ ]:


X_train.shape,y_train.shape,X_test.shape


# In[ ]:


#Split train and CV sets
train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state = 0)
train_X.shape, val_X.shape, train_y.shape, val_y.shape


# In[ ]:


#Logistic Regression
num_iter=[50,100,200,500]
C_Values=[0.01, 0.1, 0.3, 1, 3, 10,30, 100,300]
p=np.zeros((len(num_iter),len(C_Values)))
J_cv=pd.DataFrame(p,columns=C_Values, index=num_iter)
             
for i in range(len(num_iter)): 
    for j in range(len(C_Values)): 
        model=LogisticRegression(max_iter=num_iter[i],C=C_Values[j])
        model.fit(train_X,train_y)
        y_train_pred=model.predict(train_X)
        y_pred=model.predict(val_X)
        J_cv.iloc[i,j]=mean_absolute_error(val_y, y_pred)


J_cv        


# **Select C and num_iter based on min value of Jcv, i.e max_iter=200,C=0.1**

# In[ ]:


model=LogisticRegression(max_iter=200,C=0.1)
#model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
y_train_pred=model.predict(train_X)
y_cv_pred=model.predict(val_X)

print('J_cv=',mean_absolute_error(val_y, y_cv_pred))
print('J_train=',mean_absolute_error(y_train_pred, train_y))
print('Accuracy Score:',model.score(val_X,val_y))
y_pred=model.predict(X_test)


# The accuracy score increased from 0.78 to 0.793 with additional features

# In[ ]:


#Accuracy score
cm = confusion_matrix(val_y, y_cv_pred)
print(cm)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_pred
    })


# In[ ]:


#submission.to_csv(''../output/submission_lr.csv',index=False)

