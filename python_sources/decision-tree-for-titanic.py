#!/usr/bin/env python
# coding: utf-8

# ## **Import the libraries and load the datas**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# # **Missing Data**

# Let's look all the missing data we have

# In[ ]:


train.isnull().sum(axis=0)


# Let's check the Embarked

# In[ ]:


sns.countplot(dataset['Embarked'])


# How we can see, the most commun is the Embarked S, it shows frequentely so we will fill the missing feactures with the S

# In[ ]:


sns.barplot('Pclass', 'Survived', data = train)


# In[ ]:


train = train.fillna({"Embarked": "S"})


# We will make a One Hot Encoding because is avaliable for Sex and Embarked

# In[ ]:


train = pd.get_dummies(train, columns = ['Sex'])
train = pd.get_dummies(train, columns = ['Embarked'])
train.head()


# And for Age: 

# In[ ]:


data = [train, test]
for dataset in data:
    mean = train["Age"].mean()
    std = test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)


# Cabin we will drop out the column

# In[ ]:


train.drop('Cabin',axis = 1,inplace = True)


# In[ ]:


train.drop('Ticket',axis = 1,inplace = True)


# In[ ]:


train.drop('Name',axis = 1,inplace = True)


# In[ ]:


train.drop('PassengerId',axis = 1,inplace = True)


# Let's divide in train and test for the training of the model

# In[ ]:


train_c = train[['Survived']]
train_f = train.drop(['Survived'], axis = 1)
train_f.head()


# ## The Decision Tree Model

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(train_f, train_c, test_size = 0.20,random_state = 42)


# In[ ]:


clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X_train, Y_train)
export_graphviz(clf, out_file = 'Tree.dot')

pred = clf.predict(X_test)


# In[ ]:


hit_rate = accuracy_score(Y_test, pred)
error_rate = 1 - hit_rate


# In[ ]:


hit_rate * 100


# In[ ]:


print(classification_report(Y_test,pred))


# In[ ]:


print(confusion_matrix(Y_test,pred))


# In[ ]:


sub = pd.DataFrame({ "PassengerId": test['PassengerId'], "Survived": pred})

sub.head()

sub.to_csv('MySub.csv', index = False)

