#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[ ]:


train=pd.read_csv("../input/titanic/train.csv" )
tit1=train.select_dtypes(include=['float64','int64','object'])
train.info()

test=pd.read_csv("../input/titanic/test.csv")
tit2=test.select_dtypes(include=['float64','int64','object'])
test.info()


# **We should know the size of the data we are working with.**

# In[ ]:


print("train shape:",train.shape)
print("test shape :",test.shape)


# In[ ]:


tit1.head()


# In[ ]:


tit2.head()


# **Adding a column for Survived which has to be predicted in the test data.**

# In[ ]:


tit2['survived']=np.nan
tit2.head()


# # **EXPLORING FEATURES**

# **First we start by checking the counts of survived(1) and dead(0).
# thus from the below graoh it is clear that there were more deaths than the ratio of survivors.
# We also plot of graph for the division of genders, to see the ratio between men and women.
# when the graph i splotted we see that the range of women seem more equivalent to the range of survivors and the range of deaths seem more closely related to the range of men.
# So thus that mean that there were more women who survuived?
# We shall se that further.**

# In[ ]:


plt.figure(figsize=(4,4))
plt.title('SURVIVED',size=20)
tit1.Survived.value_counts().plot.bar(color=['red','green'])

plt.figure(figsize=(4,4))
plt.title('SEX',size=20)
tit1.Sex.value_counts().plot.bar(color=['skyblue','pink'])


# **Lets find out the Survival Rate**

# In[ ]:


percent=round(np.mean(train['Survived']),3)*100
print("Percentage of Survivors:",percent)


# **Lets find out the percentage of Women and Men**

# In[ ]:


total=train['Survived'].sum()
total
men=train[train['Sex']=='male']
women=train[train['Sex']=='female']
m=men['Sex'].count()
w=women['Sex'].count()
print("male:",m)
print("female:",w)
print("percentage of women:",round(w/(m+w)*100))
print("percentage of men:",round(m/(m+w)*100))


# **Lets check for the number of Null Values in our DATA SET**

# In[ ]:


train.isnull().sum()


# **AGE and CABIN have the higest number of Null Values,so they will not be of major help since most of the values are missing,especially CABIN.
# But lets see the Maximum age groups present.**

# **Lets assign X value to all the NAN values**

# In[ ]:


train['Cabin'] = train['Cabin'].fillna('X')
test['Cabin']=test['Cabin'].fillna('X')


# In[ ]:


train['Age'].hist(bins=40,color='salmon')
plt.title("AGE",size=20)


# **Lets examine the types of classes that were present**

# In[ ]:


plt.figure(figsize=(5,5))
plt.title("CLASS DIVISION",size=20)
tit1.Pclass.value_counts().plot.bar(color=['olive','coral','gold'])


# **Checking out the distribution of Fares**

# In[ ]:


train['Fare'].hist(bins = 80, color = 'orange')
plt.title("FARE",size=20)


# **Checking out Embarked Attribute.
#   It has 3 discrete Divisions,namely S , C ,Q.**

# In[ ]:


plt.figure(figsize=(5,5))
plt.title("Embarked",size=20)
tit1.Embarked.value_counts().plot.bar(color=['olive','coral','gold'])


# **Visualizing the data in our dataframe into a correlation heatmap **

# In[ ]:


sns.heatmap(train.corr(), annot = True)


# # **CLEANING DATA**

# **Since we have explored all the features in our dataset,now we shall draw close comparisons with "SURVIVED" feature,to help us draw some inference.**

# In[ ]:



plt.figure(figsize=(5,5))
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
plt.title("SURVIVED AND SEX",size=20)


# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
plt.title("SURVIVED AND PCLASS",size=20)


# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot(x = 'Survived', hue = 'Embarked', data = train)
plt.title("SURVIVED AND EMBARKED",size=20)


# **Calculating median values of "Age" by using "Pclass" and "Embarked" to fill up the missing values.**

# In[ ]:


age_group = train.groupby("Pclass")["Age"]
print(age_group.median())


# In[ ]:


age_group = train.groupby("Embarked")["Age"]
print(age_group.median())


# In[ ]:


train.loc[train.Age.isnull(),'Age']=train.groupby("Pclass").Age.transform('median')
test.loc[test.Age.isnull(),'Age']=test.groupby("Pclass").Age.transform('median')
print(train['Age'].isnull().sum())


# **Now we have no missing values for AGE**

# **Lets work out with the Cabin numbers**

# In[ ]:


test['Cabin'].unique().tolist()


# In[ ]:


cab = test.groupby("Cabin")["Age"]
print(cab.median())


# In[ ]:



train['Cabin'].unique().tolist()


# **We will be searching for the initials of the cabin numbers like A,B,C,etc**

# In[ ]:



import re

test['Cabin'] = test['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
test['Cabin'].unique().tolist()


# **Now we are assigning values to the initials that we had found in the above step and replace them with integers by mapping them.
# Same step will be repeated for train and test data**

# In[ ]:


category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'X':8}
test['Cabin'] = test['Cabin'].map(category)
test['Cabin'].unique().tolist()


# In[ ]:


cab = train.groupby("Cabin")["Age"]
print(cab.median())


# In[ ]:




train['Cabin'] = train['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train['Cabin'].unique().tolist()


# In[ ]:


category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'X':8, 'T':9}
train['Cabin'] = train['Cabin'].map(category)
train['Cabin'].unique().tolist()


# **Lets check out the missing values again**

# In[ ]:


print(train.isnull().sum())


# **Now only "Embarked" has two missing values in it.**

# In[ ]:


from statistics import mode
train["Embarked"] = train["Embarked"].fillna(mode(train["Embarked"]))


# **So now we have filled the NAN values of embarked too.Lets check the null values again!**

# In[ ]:


print(train.isnull().sum())


# **GG!!So no more missing values in our dataset**

# **Lets convert our categorical data to numeric form**

# In[ ]:


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


# **Lets create a new column of fam using SibSp which means number of Siblings or Spouse and Parch which means number of Parents or Children,later we will be dropping SibSp and Parch from our data set since these values are alreday being used in Fam**

# In[ ]:


train['fam']=train['SibSp']+train['Parch']+1
test['fam']=test['SibSp']+test['Parch']+1


# **Lets play a little with Age as well**

# **First we are converting float to string for both the datasets namely test and train**

# In[ ]:


train['Age']=train['Age'].astype(str)
test['Age']=test['Age'].astype(str)


# **Now we will do the same thing that we did with cabin so that we are left with the initials and can assign them numeric values accordingly.**

# In[ ]:


import re
train['Age'] = train['Age'].map(lambda x: re.compile("[0-9]").search(x).group())
train['Age'].unique().tolist()


# In[ ]:




cat={'2':1, '3':2 , '5':3, '1':4 ,'4':5,'8':6,'6':7,'7':8,'0':9,'9':10}
train['Age']=train['Age'].map(cat)
train['Age'].unique().tolist()


# In[ ]:


test['Age'] = test['Age'].map(lambda x: re.compile("[0-9]").search(x).group())
test['Age'].unique().tolist()


# **Mapping values**

# In[ ]:



cat={'2':1, '3':2 , '5':3, '1':4 ,'4':5,'8':6,'6':7,'7':8,'0':9,'9':10}
test['Age']=test['Age'].map(cat)
test['Age'].unique().tolist()


# **Lets play around with Name as well!**

# **Searching for the titles and extracting them from the names in the given data**

# In[ ]:


train['Title'] = train['Name'].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())
test['Title'] = test['Name'].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())
print(train['Title'].unique())
    


# In[ ]:


print(test['Title'].unique())


# **Mean survival rate according to the Titles assigned**

# In[ ]:



    train['Title'] = train['Title'].replace(['Lady.', 'Capt.', 'Col.',
    'Don.', 'Dr.', 'Major.', 'Rev.', 'Jonkheer.', 'Dona.'], 'Rare.')
    
    train['Title'] = train['Title'].replace(['Countess.', 'Lady', 'Sir'], 'Royal')
    train['Title'] = train['Title'].replace('Mlle.', 'Miss.')
    train['Title'] = train['Title'].replace('Ms.', 'Miss.')
    train['Title'] = train['Title'].replace('Mme.', 'Mrs.')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# **Replacing to make the categories narrower and accurate**

# In[ ]:



    test['Title'] = test['Title'].replace(['Lady.', 'Capt.', 'Col.',
    'Don.', 'Dr.', 'Major.', 'Rev.', 'Jonkheer.', 'Dona.'], 'Rare.')
    
    test['Title'] = test['Title'].replace(['Countess.', 'Lady.', 'Sir.'], 'Royal.')
    test['Title'] = test['Title'].replace('Mlle.', 'Miss')
    test['Title'] = test['Title'].replace('Ms.', 'Miss.')
    test['Title'] = test['Title'].replace('Mme.', 'Mrs.')


# **Mapping new numerical values onto Titles**

# In[ ]:


title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Royal.": 5, "Rare.": 6}

train['Title'] = train['Title'].map(title_mapping)
train['Title'] = train['Title'].fillna(0)

train.head()


# In[ ]:


title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Royal.": 5, "Rare.": 6}

test['Title'] = test['Title'].map(title_mapping)
test['Title'] = test['Title'].fillna(0)

   


# **Lets check whether the conversion has worked or not**

# In[ ]:


print(train['Age'])


# In[ ]:


print(train['Cabin'])


# In[ ]:


print(train['Sex'])


# In[ ]:


print(train['Embarked'])


# In[ ]:


print(train['fam'])


# **The Data that we are dropping from the dataset**

# In[ ]:



test = test.drop(['Ticket'], axis = 1)
test = test.drop(['Name'], axis = 1)
test = test.drop(['Parch'], axis = 1)
test = test.drop(['Fare','SibSp'], axis = 1)

train.drop(['Name', 'Ticket'], axis = 1, inplace = True)
train = train.drop(['Parch'], axis = 1)
train = train.drop(['Fare','SibSp'], axis = 1)


# **Lets start predicting,we will be using Logistic Regression.Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.**

# **train_test_split :Split arrays or matrices into random train and test subsets**

# In[ ]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived','PassengerId'], axis=1), 
                                                    train['Survived'], test_size = 0.2, 
                                                    random_state = 0)


# # **LOGISTIC REGRESSION**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression(max_iter = 30000)
logisticRegression.fit(X_train, y_train)



# **Making and Printing our predictions**

# In[ ]:


predictions = logisticRegression.predict(X_test)
acc_LOG = round(accuracy_score(predictions, y_test) * 100, 2)
print(acc_LOG)
print(predictions)



# In[ ]:


round(np.mean(predictions), 3)


# **This mean is pretty close to the one that we had calculated earlier(0.384)**

# **Lets take help of confusion matrix to find out TP TN FP FN.A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix. The confusion matrix shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.
# **
# 

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))


# ![image.png](attachment:image.png)
# 

# In[ ]:


accuracy=((88+50)/(88+50+22+19))
print('accuracy is: ', (round(accuracy, 2)*100))


# # **RANDOM FOREST**

# **Lets try using Random Forest.A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees.**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Define our optimal randomForest algo
randomforest = RandomForestClassifier(random_state = 5, criterion = 'gini', max_depth = 10, max_features = 'auto', n_estimators = 500)
randomforest.fit(X_train, y_train)
pred = randomforest.predict(X_test)
acc_randomforest = round(accuracy_score(pred, y_test) * 100, 2)
print(acc_randomforest)


# # **GRADIENT BOOSTING**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(pred, y_test) * 100, 2)
print(acc_gbk)


# **Checking out the accuracy of our predictions**

# In[ ]:


see={'TECHNIQUE':['RANDOM FOREST','LOGISTIC REGRESSION','GRADIENT BOOSTING'],'ACCURACY':[acc_randomforest,acc_LOG,acc_gbk]}
mod=pd.DataFrame(see)
mod


# Doing Rank Averaging ensembling modelling

# In[ ]:


mod['Rank']=[2,1,3]
mod


# In[ ]:


mod['Weighted Rank']=[2/6,1/6,3/6]
mod


# In[ ]:


final_pred=np.dot(mod['ACCURACY'],mod['Weighted Rank'])
final_pred


# In[ ]:


train.head()


# In[ ]:


test.head()


# **SUBMISSION FILE(choosing Gradient Boosting)**

# In[ ]:


ids = test['PassengerId']
predictions = randomforest.predict(test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

