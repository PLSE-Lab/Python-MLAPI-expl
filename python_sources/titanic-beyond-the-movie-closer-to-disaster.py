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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


RawData = pd.read_csv('../input/train.csv');
#X = RawData.copy().drop('Survived', 1);
X = RawData.copy();
Y = RawData['Survived'];
print(X.head())
print(list(X))


# **Let's try to understand the statistic of those who survived**

# In[ ]:


print(X[X['Survived'] == 1].describe())


# **Let's try to understand the statistic of those who died.**

# In[ ]:


#print(X[X['Survived'] == 0].describe())


# **Looks like those who were not able to survive were close to 3rd class as mean is ~2.5**
# > Bechara garib hamesha maara jata hai, aur baki ko aashiqui le doobti hai
# 
# **Chalo, let's go deeper into the analysis of the data classwise.**

# In[ ]:


#print(pd.DataFrame({'count' : X[['Fare', 'Pclass']].groupby( [ "Pclass", 'Fare'] ).size()}))
X_groupBy_Class = X.groupby( [ "Pclass"] );
#print(X_groupBy_Class.mean()) #head(10))


# **People in 1st class were more likely to survive, and they were old too..**
# > Paisa ekdum nahi aata, time lagta hai
# 
# ** We know that women and kids were given preference over men, does our data show that fact. **

# In[ ]:


X_groupBy_Class_And_Gender = X.groupby( [ "Pclass", "Sex"] );
print(X_groupBy_Class_And_Gender.mean()) #head(10))


# In[ ]:


X_categorized_by_age = pd.cut(X['Age'], np.arange(0, 90, 10)); # return array of half open bins to which `age` belongs.
#print(X.groupby([X_categorized_by_age, 'Pclass', 'Sex'])['Survived', 'SibSp', 'Parch'].mean())


# In[ ]:


#X.groupby([X_categorized_by_age]).mean()['Survived'].plot.bar()


# **What does our initial analysis says**
# 1. It looks like being either kid or girl or 1st class ticket holder raises the chance of survival.

# In[ ]:


X.corr()


# **Feature Selection**
# * Remove un-necessary columns
# * User Wrapper method. **[TODO]**
# 

# In[ ]:


print(list(X))
Tickets_df = X[['Ticket', 'Pclass']]
X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# **Feature Extraction - Create new feature  [TODO]**
# * It looks like `pClass` and `Fare` can be combined. They are intuitively and statically correlated.
# * Also, `parch` and `sibsp` can be combined, as they both belongs to `Family` of person.
# * Can we get something fruitful out of `Name`, `Cabin` and `Ticket number` which we can use in our algorithm as input.
# 
# *We have to define mathematical function using which they can be combined.*
# 

# > Analysing Ticket

# In[ ]:


Tickets_df.groupby('Pclass').describe()


# **Preprocessing**
# * Convert categorical data into continuous - One Hot encoder or dummy variables.
# * `Age` seems to be bit out of scale, hence needs to be scaled. **[TODO]** 

# In[ ]:


X = pd.get_dummies(X)
#print(X.head())


# **Imputation - Handle missing values**
# * Try to fill in some logical value that makes sense, but not hamper the predictablity as well. Use Imputers.
# * Delete the rows against which values can not be imputed. [Column is significant, unable to impute value, few missing values]
# * Delete the column where missing values exists. [Only when column is either not so significant, or too many missing values]

# In[ ]:


X = X.dropna(subset=['Age'])
X.count()
Y = X['Survived']
Y.head()
#X = X.drop(['Survived'], axis=1)
#X.count()


# **Try SVM**

# In[ ]:


from sklearn import svm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
scores = []
C_Vals = []
# Hyper parameter tuning 
for idx in np.arange(0, 5, 1):
    C_Val = .01*10**idx
    C_Vals.append(C_Val)
    clf = svm.SVC(C_Val)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))     
print(pd.DataFrame({'score': scores, 'C':C_Vals}))


# **Try LR approach**

# **Try Random forest approach**

# **Try Decision Tree**

# **Try K-Means**

# **Try Neural Network ( if possible )**

# **Try Gradient Boosting Algorithm**
