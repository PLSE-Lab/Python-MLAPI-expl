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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# * Data Loading and Description

# In[ ]:



train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()


# - The dataset consists of the information about people boarding the famous RMS Titanic. Various variables present in the dataset includes data of age, sex, fare, ticket etc. 
# - The training dataset comprises of __891 observations of 12 columns__. Below is a table showing names of all the columns and their description.
# - The testing dataset comprises of __418 observations of 11 columns__.

# | Column Name   | Description                                               |
# | ------------- |:-------------                                            :| 
# | PassengerId   | Passenger Identity                                        | 
# | Survived      | Whether passenger survived or not                         |  
# | Pclass        | Class of ticket                                           | 
# | Name          | Name of passenger                                         |   
# | Sex           | Sex of passenger                                          |
# | Age           | Age of passenger                                          |
# | SibSp         | Number of sibling and/or spouse travelling with passenger |
# | Parch         | Number of parent and/or children travelling with passenger|
# | Ticket        | Ticket number                                             |
# | Fare          | Price of ticket                                           |
# | Cabin         | Cabin number                                              |

# In[ ]:


train.info()


# In[ ]:


train.describe()


# __Preprocessing the data__

# In[ ]:


train.isna().sum()


# - Dealing with missing values<br/>
#     - Dropping/Replacing missing entries of __Embarked.__
#     - Replacing missing values of __Age__ and __Fare__ with median values.
#     - Dropping the column __'Cabin'__ as it has too many _null_ values.

# In[ ]:


train.Embarked = train.Embarked.fillna(train['Embarked'].mode()[0])


# In[ ]:


median_age = train.Age.median()
train.Age.fillna(median_age, inplace = True)

train.drop('Cabin', axis = 1,inplace = True)


# - Creating a new feature named __FamilySize__.

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch']+1


# - Segmenting __Sex__ column as per __Age__, Age less than 15 as __Child__, Age greater than 15 as __Males and Females__ as per their gender.

# In[ ]:


train['GenderClass'] = train.apply(lambda x: 'child' if x['Age'] < 15 else x['Sex'],axis=1)


# In[ ]:


train[train.Age<15].head(2)


# - __Dummification__ of __GenderClass__ & __Embarked__.

# In[ ]:


train = pd.get_dummies(train, columns=['GenderClass','Embarked'], drop_first=True)


# - __Dropping__ columns __'Name' , 'Ticket' , 'Sex' , 'SibSp' and 'Parch'__ 

# In[ ]:


train = train.drop(['Name','Ticket','Sex','SibSp','Parch'], axis = 1)


# In[ ]:


train.head()


# Drawing __pair plot__ to know the joint relationship between __'Fare' , 'Age' , 'Pclass' & 'Survived'__

# In[ ]:


import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
sns.pairplot(train[["Fare","Age","Pclass","Survived"]],vars = ["Fare","Age","Pclass"],hue="Survived", dropna=True,markers=["o", "s"])
plt.title('Pair Plot')


# Observing the diagonal elements,
# - More people of __Pclass 1__ _survived_ than died (First peak of red is higher than blue)
# - More people of __Pclass 3__ _died_ than survived (Third peak of blue is higher than red)
# - More people of age group __20-40 died__ than survived.
# - Most of the people paying __less fare died__.

# Establishing __coorelation__ between all the features using __heatmap__.

# In[ ]:


corr = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,vmax=.8,linewidth=.01, square = True, annot = True,cmap='YlGnBu',linecolor ='black')
plt.title('Correlation between features')


# - __Age and Pclass are negatively corelated with Survived.__
# - FamilySize is made from Parch and SibSb only therefore high positive corelation among them.
# - __Fare and FamilySize__ are __positively coorelated with Survived.__
# - With high corelation we face __redundancy__ issues.

# __Preparing X and y using pandas__

# In[ ]:


X = train.loc[:,train.columns != 'Survived']
X.head()


# In[ ]:


y = train.Survived 


# In[ ]:


y.head()


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# __Logistic regression in scikit-learn__

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# __Using the Model for Prediction__

# In[ ]:


y_pred_test = logreg.predict(X_test)  


# __model accuracy__

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))


# __ processinng of test data__

# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


median_age = test.Age.median()
test.Age.fillna(median_age, inplace = True)


# In[ ]:


test.drop('Cabin', axis = 1,inplace = True)


# In[ ]:


median_fare = test.Fare.median()
test.Fare.fillna(median_fare, inplace = True)


# In[ ]:


test['FamilySize'] = test['SibSp'] + test['Parch']+1


# In[ ]:


test['GenderClass'] = test.apply(lambda x: 'child' if x['Age'] < 15 else x['Sex'],axis=1)


# In[ ]:


test = pd.get_dummies(test, columns=['GenderClass','Embarked'], drop_first=True)


# In[ ]:


test_processing = test.drop(['Name','Ticket','Sex','SibSp','Parch'], axis = 1)
test_processing.head()


# __Model Evaluation for test data__

# In[ ]:


y_pred_test = logreg.predict(test_processing)  


# In[ ]:


y_pred_test


# In[80]:


y_pred_test.shape


# In[82]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_test
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




