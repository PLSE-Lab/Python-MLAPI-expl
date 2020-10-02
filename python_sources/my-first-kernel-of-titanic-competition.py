#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualizations
import seaborn as sns #for interactive visualizations
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#importing datasets
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
totaldata = [train, test]


# In[ ]:


#check both train and test datasets
train.head()


# In[ ]:


test.head()


# In[ ]:


train.tail()


# In[ ]:


test.tail()


# In[ ]:


#finding columns in train data set
train.columns


# In[ ]:


#the above code gives you in Index form
#this gives you in an array form
train.columns.values


# In[ ]:


train.columns[0]


# In[ ]:


train.columns.values[0]


# In[ ]:


#finding the shape(no.of.rows, no.of.columns) in train
train.shape


# In[ ]:


#finding the shape(no.of.rows, no.of. columns) in test
test.shape


# In[ ]:


train.info()


# Cabin and age has missing values, out of 12 coulmns 5 are object(text), 5 are int64(numbers) and 2 are float(in decimal points)
# Name, Sex, Ticket, Cabin, Embarked are text data type.
# remaining are numericals.
# Age,Cabin  and Embarked has missing values.

# In[ ]:


test.info()


# so we had missing values in Age,1 value in Fare and some values in Cabin

# In[ ]:


train.describe()


# **Data preprocessing **

# handling missing values.
# 
# from the train.info( ) we can say that we had missing values in Age, Cabin , Embarked columns as they have less rows out of 891.
# 

# In[ ]:


#again checking missing values using isna() or isnull() method which most of the people do
train.isna().sum
#using this code snippet it only shows boolean values. if the column contains missing values then 
#it shows True, else it shows False.


# In[ ]:


train.isna().sum()


# In[ ]:


#same code as above but using isnull() method
train.isnull().sum()


# In[ ]:


#from the above code snippet, its clear that we had missing values in Age, Cabin, Embarked columns.
#Cabin has 687 missing values, Age has 177 missing values, Embarked has 2 missing values.
#now try to fill those missing vlues.


# In[ ]:


test.isnull().sum()


# We had 86 misssing values in Age, 327 missing values in Cabin, 1 in Fare

# In[ ]:


#lets replace missing values of Embarked column in trian data set
#trying to find out what and how many are the values present in Embarked column
train.Embarked.value_counts()


# In[ ]:


#mostly missing values are replaced with Either of Mean,Median,Mode
#so totally we have 'S' repeated more times which is Mode case.
#so fill the missing 2 values with most repeated value i.e "S"
train["Embarked"]  = train["Embarked"].fillna("S")


# In[ ]:


#after filling missing values in Embarked column, verifyinig still we have missing values in 
#Embarked column or not.
train.isna().sum()


# In[ ]:


#so now we do not have any missing values in Embarked columns. 
#fill missing value of Age with Median.in both test and train.
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


#lets fill the 1 missing value in Fare column of test data set
test['Fare'] = test["Fare"].fillna("0")


# In[ ]:


test.info()


# In[ ]:


#As Cabin Column has most data missing ,
#there wont be any use filling missing values as it would be hard to fill those missing values using
#using any of the techniques like Mean, Medain and  Mode.we have to delete it. 
train = train.drop("Cabin",axis = 1)
test = test.drop("Cabin", axis = 1)
train.info()


# In[ ]:


test.info()


# In[ ]:


#now we had another problem, i.e after filling all the missing values, now we have to convert text 
#data to numerical data, because that is what a machine understands at the end of the day.
#for that we have to use label encoder
#but before that lets find can we convert all the text values to numerical ones.
train.Ticket.value_counts()


# In[ ]:


train['Ticket'].describe()


# In[ ]:


#its clear that we had 681 unique values, so its impossible to encode it. so delete Ticket column
train = train.drop(['Ticket'],axis = 1)
train.info()


# In[ ]:


test['Ticket'].describe()


# In[ ]:


test = test.drop(['Ticket'],axis = 1)
test.info()


# In[ ]:


#also there is no use with name, so lets delete it
train = train.drop(["Name"], axis = 1)
test = test.drop(['Name'], axis =1)
train.info()


# In[ ]:


test.info()


# In[ ]:


#lets encode remaning 2 text columns to numerical ones
#lets write function for label encoder
from sklearn.preprocessing import LabelEncoder


# In[ ]:


def encode_features(dataset,featurenames):
    for featurename in featurenames:
        LE = LabelEncoder()
        LE.fit(dataset[featurename])
        dataset[featurename] = LE.transform(dataset[featurename])
    return dataset    


# In[ ]:



train = encode_features(train, ['Sex','Embarked'])
test = encode_features(test, ['Sex','Embarked'])


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


#lets convert Fare from float to int
train['Fare'] = train['Fare'].astype(int)
test["Fare"] = test['Fare'].astype(int)
train.info()


# In[ ]:


test.info()


# **Data visualization**

# As our ML models fastens the processing if our data is in  0,1,2....., in simplie numeric form. so lets change Fare and Age to such format.
# now we will find the correlation between Age and survival and between Fare and survival.

# In[ ]:





# In[ ]:


test.head()


# In[ ]:


train = train.drop(['PassengerId'], axis=1)
train.head()


# **ML models**

# In[ ]:


X_train = train.drop(['Survived'], axis = 1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# In[ ]:


X_test.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#Logistic regression
LR = LogisticRegression()
LR.fit(X_train, Y_train)
y_pred_LR = LR.predict(X_test)
LR.score(X_train,Y_train)*100


# In[ ]:


#Support vector machine 
svm = SVC()
svm.fit(X_train, Y_train)
y_pred_svm = svm.predict(X_test)
svm.score(X_train,Y_train)*100


# In[ ]:


#RandomForestClassirier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
y_pred_rfc = rfc.predict(X_test)
rfc.score(X_train,Y_train)*100


# In[ ]:


#KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)
y_pred_knc = knc.predict(X_test)
knc.score(X_train,Y_train)*100


# In[ ]:


#KNeighborsClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
y_pred_dtc = dtc.predict(X_test)
dtc.score(X_train,Y_train)*100


# **If you find any mistakes or if you have any suggestions, please comment down. I would like to learn from my mistakes.**

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_dtc
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




