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


# # Starter libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Hello everyone!
# In this kernel you will not see anything new, but you did it yourself.
# Thank you and like it.

# In[ ]:


titanic = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


# data loading
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
df = train.append(test, ignore_index=True)
passenger_id = df[891:].PassengerId
df.head()


# In[ ]:


def description(df):
    print(f'Dataset Shape:{df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values   
    summary['Uniques'] = df.nunique().values
    return summary
print('Data Description:')
description(df)


# ## What we see from the data:
# 1. There are unnecessary columns. We will remove them.
# 2. The Sex field will be tediously converted to int format
# 3. Supplement Age; empty values are present
# 4. Convert Embarked field to int format
# 5. Let's come back to this later.
# 
# # Visualization!

# ## Countplot

# In[ ]:


sns.countplot('Survived', data=df, palette='Set2')
plt.ylabel('Number of survivors')
plt.title('Distribution of survivors');


# ## Boxplot 

# In[ ]:


sns.boxplot(x="Survived", y="Age", data=df, palette='rainbow');


# ## Violinplot

# In[ ]:


sns.violinplot('Pclass','Age', hue='Survived',
               data=df,palette="Set2", split=True,scale="count");


# ## Swarmplot

# In[ ]:


sns.swarmplot(x='Embarked', y='Fare', data=df);


# ## Swarmplot + violin plot

# In[ ]:


ax = sns.violinplot(x="Sex", y="Age", data=df, inner=None)
ax = sns.swarmplot(x="Sex", y="Age", data=df,
                   color="white", edgecolor="gray")


# ## Use Catplot to combine a swarmplot() and a FacetGrid.

# In[ ]:


sns.catplot(x="Pclass", y="Fare",
            hue="Survived", col="Sex",
            data=df, kind="swarm");


# ## Jointplot

# In[ ]:


sns.jointplot("Age", "Pclass", data=df,
                  kind="kde", space=0, color="g");


# ## Heatmap

# In[ ]:


plt.figure(figsize=(17,10))
matrix = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, mask=matrix,cmap= 'coolwarm');


# # Let's go back to working with data

# # Data Preparation

# We remove the speakers that seem to us a little informative
# 
# * PassengerID
# * Name
# * Ticket
# * Cabin
# * Fare

# In[ ]:


df = df.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis=1)
df.head()


# In[ ]:


# Remember what needs to be done
description(df)


# ### Replace Sex with LabelEncoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelEnc = LabelEncoder()
df.Sex=labelEnc.fit_transform(df.Sex)


# ### Replace the average age

# In[ ]:


df['Age'] = df.Age.fillna(df.Age.mean())


# ## Replace the average Embarked

# In[ ]:


df.Embarked.value_counts()


# In[ ]:


df['Embarked'] = df.Embarked.fillna('S')


# # One_hot

# In[ ]:


Embarked = pd.get_dummies(df.Embarked , prefix='Embarked' )
Embarked.head()


# In[ ]:


Pclass = pd.get_dummies(df.Pclass, prefix='Pclass')
SibSp = pd.get_dummies(df.SibSp, prefix='SibSp')
Parch = pd.get_dummies(df.Parch, prefix='Parch')
df_new = pd.concat([df, Embarked, Pclass, SibSp, Parch], axis=1)


# In[ ]:


df_new = df_new.drop(['Pclass', 'SibSp','Parch', 'Embarked'], axis=1)
description(df_new)


# ## Moving on to machine learning

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_valid_X = df_new.drop(['Survived'],axis=1)[ 0:891]
train_valid_y = df_new.Survived[ 0:891]
test_X = df_new.drop(['Survived'],axis=1)[891:]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_valid_X, train_valid_y, test_size=.7)


# # Training model

# ## Extracting the best fitted DecisionTreeClassifier after Grid Search

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
clf_tree= DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,parameters)
clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, clf.predict(X_test))


# In[ ]:


test_Y = clf.predict(test_X)
test = pd.DataFrame({'PassengerId': passenger_id,'Survived': test_Y})
test.to_csv('/kaggle/working/titanic_pred.csv', index = False)

