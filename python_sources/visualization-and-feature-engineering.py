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
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

import scipy as sp
import seaborn as sns
import re
import matplotlib.pyplot as plt

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')


# # Creating new variables

# In[ ]:


#creating new variables family and family_size which weather the passenger has a family 
#and total no of family members onboard.
#if the passenger has a family onboard then it is 1 else 0
train['family']=train.SibSp+train.Parch
train.family=train.family.apply(lambda x: 1 if x>0 else 0 )
#total no of family members onboard
train['family_size']=train.SibSp+train.Parch+1
def cap(x):
    if x>6:
        return 6
    else:
        return x
train.family_size=train.family_size.apply(cap)

test['family']=test.SibSp+test.Parch
test.family=test.family.apply(lambda x: 1 if x>0 else 0 )
test['family_size']=test.SibSp+test.Parch+1
def cap(x):
    if x>6:
        return 6
    else:
        return x
test.family_size=test.family_size.apply(cap)


# # Data Visualization

# In[ ]:


#Age distribution for Survival
g= sns.FacetGrid(train, col="Survived", col_wrap=4, size=5)
g.map(sns.distplot,"Age" )


# In[ ]:


#Age distribution over Sex
g= sns.FacetGrid(train, col="Sex", col_wrap=4, size=5)
g.map(sns.distplot,"Age" )


# In[ ]:


#Survived passenger's Age distribution over Sex
g= sns.FacetGrid(train[train.Survived==1], col="Sex", col_wrap=4, size=5)
g.map(plt.hist,"Age" )


# In[ ]:


#Dead passenger's Age distribution over Sex
g= sns.FacetGrid(train[train.Survived==0], col="Sex", col_wrap=4, size=5)
g.map(plt.hist,"Age" )


# In[ ]:


#Survived passenger's distribution over Pclass
g= sns.FacetGrid(train[train.Survived==1], col="Pclass", col_wrap=4, size=5)
g.map(plt.hist,"Age" )


# In[ ]:


#females are more likely to survive
a=train.groupby('Sex').Survived.value_counts()/train.groupby('Sex').Survived.count()
a.unstack().plot(kind='bar',stacked=True,color=['r','b'])


# In[ ]:


#higher class passengers are more likely to survive
a=train.groupby('Pclass').Survived.value_counts()/train.groupby('Pclass').Survived.count()
a.unstack().plot(kind='bar',stacked=True,color=['r','b'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=12)


# # Feature Engineering

# In[ ]:


#adding survived column to 0 for appending it with train data
test['Survived']=0

#Creating a new variable which denotes the salutation
def parse_des(x):
    x=re.findall(r',\s\w+.',x)
    return (re.findall(r'\w+',str(x)))[0]
train['desig']=train['Name'].apply(parse_des)
test['desig']=test['Name'].apply(parse_des)

data=train.append(test)
data.set_index(np.arange(0,np.size(data,axis=0)),inplace=True)
#checking all the salutation that are extracted from the data
train.desig.value_counts()


# In[ ]:


data.isnull().sum()


# # Missing Value Treatment

# In[ ]:


#Creating a reference table based on mean partitioned on the basis of other variables
impute_ref=pd.DataFrame(data.groupby(['desig','Pclass']).Age.mean().dropna())

#checking types of missing ages info Ms is similar to Miss
data[data.desig=="Ms"]=data[data.desig=="Ms"].replace("Ms","Miss")
pd.DataFrame(data[data.Age.isnull()].groupby(['desig','Pclass']).Age.max())


# In[ ]:


#filling the missing value in Fare
data.Fare.fillna(np.mean(data.Fare),inplace=True)


# In[ ]:


#filling the values according to the reference table
for items in data[data.Age.isnull()].index:
    data.Age[items]=np.floor(impute_ref.ix[data.desig[items],data.Pclass[items]].Age)


# In[ ]:


#looking at the data we can say that these two passengers somehow related
data[data.Embarked.isnull()]


# In[ ]:


#as we can see from both the grouping, both of them have probably boarded from S.
#filling the missing value with S because it can be seen from grouping
data.Embarked.fillna('S',inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


#label encoded for categorical variables 
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
list=['Embarked','Parch','Pclass','SibSp','desig','family','family_size','Sex']
for items in list:
    le.fit(data[items])
    data[items]=le.transform(data[items])


# In[ ]:


#separating train and test data
train_x=data[['Age','Fare','Embarked','Parch','Pclass','SibSp','desig','family','family_size','Sex']][0:np.size(train,axis=0)]
label_x=train['Survived']
test_x=data[['Age','Fare','Embarked','Parch','Pclass','SibSp','desig','family','family_size','Sex']][891:1309]


# In[ ]:


#scaling the data
scaler=preprocessing.StandardScaler()
train_x=scaler.fit_transform(train_x)
test_x=scaler.fit_transform(test_x)


# In[ ]:


#making  a logistic classiffication model
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(penalty='l2',C=1)
clf.fit(train_x,label_x)
pred=clf.predict(test_x)


# In[ ]:


result=pd.DataFrame(pred,columns=['Survived'])


# In[ ]:


result['PassengerId']=test.PassengerId


# In[ ]:




