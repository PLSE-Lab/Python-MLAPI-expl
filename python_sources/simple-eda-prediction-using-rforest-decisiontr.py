#!/usr/bin/env python
# coding: utf-8

# **For detailed understanding of EDA steps and exhaustive visualizations on the Titanic dataset refer** [Exploratory Data Analysis and Visualization](https://www.kaggle.com/krrai77/exploratory-data-analysis-and-visualization)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


trainnew=train.drop(['Name','Ticket','Cabin'], axis=1)


# In[ ]:


testnew=test.drop(['Name','Ticket','Cabin'], axis=1)


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


trainnew['Embarked'].value_counts()


# In[ ]:


#Handling missing values in Training data

trainnew['Embarked'].fillna('S',inplace=True)
trainnew['Age'].fillna(29,inplace=True)


# In[ ]:


#Handling Missing values in Test data

testnew['Age'].fillna(30,inplace=True)
testnew['Fare'].fillna(35,inplace=True)


# In[ ]:


train.SibSp.unique()


# In[ ]:


train.Parch.unique()


# In[ ]:


plt.figure(figsize=(10,10))
corr = trainnew.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(30, 320, n=200),
    square=True, annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=45,

);


# In[ ]:


sns.pairplot(trainnew)


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x=trainnew['Fare'])


# In[ ]:


relplot = sns.catplot(x="Pclass", hue="Sex", col="Survived",  
      data=trainnew, kind="count",  
    height=4, aspect=.7);  
relplot


# In[ ]:


relplot = sns.catplot(x="Sex", hue="Embarked", col="Survived",  
      data=trainnew, kind="count",  
    height=4, aspect=.7);  
relplot


# In[ ]:


import plotly.express as px
fig = px.box(trainnew,x='Survived',y='Age', color='Sex')
fig.show()


# In[ ]:


fig = px.box(trainnew,x='Survived',y='SibSp', color='Parch')
fig.show()


# In[ ]:


#Handling Outliers in Training data

i=trainnew[trainnew.Fare == 512.329200].index
j=trainnew[trainnew.Age == 80.000000 ].index
train1=trainnew.drop(i)
train1=train1.drop(j)


# In[ ]:


ax = sns.countplot(x = 'Survived',data = train1) 
total = float(len(train1))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.show()


# In[ ]:


#Encode categorical variables

encoded_training = pd.get_dummies(train1)
encoded_testing = pd.get_dummies(testnew)


# In[ ]:



#Models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X=encoded_training.drop(['Survived'],axis=1)
Y=encoded_training.Survived


# In[ ]:


X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=500)


# In[ ]:


#Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[ ]:


#KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test) 
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test) 
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Decision Tree

decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test) 
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[ ]:


#Best model

results = pd.DataFrame({
    'Model': [ 'KNN', 'Random Forest', 'Naive Bayes',  
               'Decision Tree'],
    'Score': [ acc_knn, acc_random_forest, acc_gaussian,  
              acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()


# In[ ]:


#Prediction using Random Forest
submission = pd.DataFrame({
    "PassengerId": testnew["PassengerId"],
    "Survived": random_forest.predict(encoded_testing)
})


# In[ ]:


submission['Survived'].value_counts()


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:


#Prediction using Decision Tree
submission1 = pd.DataFrame({
    "PassengerId": testnew["PassengerId"],
    "Survived": decision_tree.predict(encoded_testing)
})


# In[ ]:


submission1['Survived'].value_counts()


# In[ ]:


submission.to_csv("submission1.csv",index=False)

