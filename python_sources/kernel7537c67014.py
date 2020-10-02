#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()
test.shape


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


train["Survived"].value_counts()


# In[ ]:


# sns.pairplot(train)
# plt.show()


# In[ ]:


# sns.heatmap(train)


# In[ ]:


890*.5


# In[ ]:


def bar_graph(features):
    survived = train[train["Survived"]==1][features].value_counts()
    dead = train[train["Survived"]==0][features].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index=['survived' , 'dead']
    df.plot(kind='bar',stacked=True)


# In[ ]:


bar_graph("Sex")


# In[ ]:


# (train[train["Survived"]==1]["Sex"].value_counts())


# In[ ]:


bar_graph("Pclass")


# In[ ]:


bar_graph("SibSp")


# In[ ]:


bar_graph("Parch")


# In[ ]:


bar_graph("Embarked")


# In[ ]:


train["Age"].fillna(train["Age"].mean() , inplace=True)
test["Age"].fillna(test["Age"].mean() , inplace=True)
# train["Age"] =train["Age"].fillna(train["Age"].mean() )
train.isnull().sum()


# In[ ]:


train.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1 , inplace =True)
test.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1 , inplace =True)


# In[ ]:


train.dropna(axis = 0,inplace=True)
test.dropna(axis = 0,inplace=True)


# In[ ]:


train


# In[ ]:


train.loc[train["Sex"]=="male" , "Sex"]=0
train.loc[train["Sex"]=="female" , "Sex"]=1

train.loc[train["Embarked"]=="S" , "Embarked"]=0
train.loc[train["Embarked"]=="Q" , "Embarked"]=1
train.loc[train["Embarked"]=="C" , "Embarked"]=2


test.loc[test["Sex"]=="male" , "Sex"]=0
test.loc[test["Sex"]=="female" , "Sex"]=1

test.loc[test["Embarked"]=="S" , "Embarked"]=0
test.loc[test["Embarked"]=="Q" , "Embarked"]=1
test.loc[test["Embarked"]=="C" , "Embarked"]=2


# In[ ]:


train.columns


# In[ ]:





# In[ ]:





# In[ ]:


train.head()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])


# In[ ]:


prediction = clf.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']])


# In[ ]:


clf.score(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])


# In[ ]:





# ## Decision tree
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


sk  = DecisionTreeClassifier(criterion="entropy" , max_depth=5)
sk.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])
sk.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']])


# In[ ]:


sk.score(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])


# In[ ]:


get_ipython().system('pip install pydotplus')
get_ipython().system('pip install --upgrade scikit-learn==0.20.3')
get_ipython().system('pip install mglearn')


# In[ ]:


import pydotplus

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz


# In[ ]:


dot_data = StringIO()
export_graphviz(sk,out_file=dot_data,filled=True,rounded=True)


# In[ ]:


['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']


# In[ ]:



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[ ]:


from sklearn.ensemble import  RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=10 , criterion="entropy" , max_depth=5)


# In[ ]:


rf.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])


# In[ ]:


rf.score(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])


# In[ ]:




