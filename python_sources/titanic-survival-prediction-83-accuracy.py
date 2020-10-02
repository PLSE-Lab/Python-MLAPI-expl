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


# **Import Libraries-**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
Passenger_ID = test['PassengerId']


# In[ ]:


sns.heatmap(train.isnull(), cmap='plasma')


# In[ ]:


sns.countplot('Survived', data = train, hue='Sex')


# As You can see, Female Survivals are more than Male

# In[ ]:


sns.countplot('SibSp', data = train)


# **Data Cleaning**

# In[ ]:


train.sort_values(by=['SibSp'], ascending=False).head(10)


# In[ ]:


outliner_SibSp = train.loc[train['SibSp'] == 8]


# In[ ]:


outliner_SibSp


# In[ ]:


train = train.drop(outliner_SibSp.index, axis=0)


# In[ ]:


train.sort_values(by=['Fare', 'Pclass'], ascending=False)


# **Here three people have Much Higher Fare 500+. As You can see the Huge differennce in Below Boxplot**

# In[ ]:


sns.boxplot(train['Fare'],orient='v')


# In[ ]:


outliner_Fare = train.loc[train['Fare']>500]


# In[ ]:


outliner_Fare


# In[ ]:


train = train.drop(outliner_Fare.index, axis=0)


# In[ ]:


dataset = pd.concat([train, test], ignore_index=True)


# In[ ]:


dataset.head()


# In[ ]:


sns.heatmap(dataset.isnull(), cmap='viridis')


# In[ ]:


dataset.shape


# In[ ]:


dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# *As You can see 'Embarked' column has 2 null value and 'Fare' has 1 null value, So Let's fill these Null Values*

# In[ ]:


dataset.loc[dataset['Embarked'].isnull()]


# In[ ]:


sns.countplot(dataset['Embarked'])


# *As You can See 'Embarked' column has 'S' that is Mode.*

# In[ ]:


dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


dataset.loc[dataset['Fare'].isnull()]


# In[ ]:


temp = dataset[(dataset['Pclass'] == 3) & (dataset['Parch'] ==0) & (dataset['SibSp'] == 0) 
               & (dataset['Fare']>0)].sort_values(by='Fare', ascending=False)


# In[ ]:


temp.mean()


# In[ ]:


temp['Fare'].mean()


# In[ ]:


dataset['Fare'] = dataset['Fare'].fillna(temp['Fare'].mean())


# In[ ]:


dataset.isnull().sum()


# *Now we fill 'Embarked' and 'Fare' columns Null value. Now lets deal with another columns Null Values ['Age', 'Cabin', 'Survived']*

# In[ ]:


dataset[(dataset['Survived'] == 0) & (dataset['Sex'] == 'male')]['Age'].mean()


# In[ ]:


nullAge = dataset.loc[dataset['Age'].isnull()]


# In[ ]:


nullAge.shape


# *We have 256 Null Values in 'Age' Column. Let's deal with them*

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age, axis=1)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


sns.catplot(data = dataset, x = 'Pclass', y = 'Survived', kind='bar')


# *Here First Class people have more count of Survival.*

# In[ ]:


g = sns.FacetGrid(data = dataset[dataset['Survived'] == 1], col='Pclass')
g.map(sns.countplot, 'Sex')


# *As You can Above plot, Female are more survived than Man*

# In[ ]:


X=dataset.drop(['Cabin','Name','PassengerId','Survived','Ticket'],axis=1)
Y=dataset['Survived']


# In[ ]:


X.head(3)


# **Let's Perfrom Data Preprocessing & Machine Leaning**

# In[ ]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# In[ ]:


X['Embarked']=LabelEncoder().fit_transform(X['Embarked'])
X['Sex']=LabelEncoder().fit_transform(X['Sex'])
X['Age']=StandardScaler().fit_transform(np.array(X['Age']).reshape(-1,1))
X['Fare']=StandardScaler().fit_transform(np.array(X['Fare']).reshape(-1,1))


# In[ ]:


X.head(3)


# *Here You can See the Difference between the above X.head() and this X.head(3), Sex columns become Numerical. *

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


trainDataX=X[:train.shape[0]]
trainDataY=Y[:train.shape[0]].astype('int32')
testDataX=X[train.shape[0]:]


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(trainDataX,trainDataY,test_size=0.1,random_state=101)


# **Let's Perform Some Algorithms-**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier().fit(X_train, Y_train)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(Y_train, dtree.predict(X_train)))


# In[ ]:


print(accuracy_score(Y_test, dtree.predict(X_test)))


# ![](http://)**Random Forest-**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


randomForest = RandomForestClassifier().fit(X_train, Y_train)


# In[ ]:


print(accuracy_score(Y_train, randomForest.predict(X_train)))


# In[ ]:


print(accuracy_score(Y_test, randomForest.predict(X_test)))


# In[ ]:


submission = pd.DataFrame(columns=['PassengerId','Survived'])
submission['PassengerId'] = Passenger_ID
submission['Survived'] = dtree.predict(testDataX)


# In[ ]:


submission.head()


# In[ ]:


filename = 'submit.csv'


# In[ ]:


submission.to_csv(filename, index=False)


# In[ ]:


from IPython.display import FileLink
FileLink(filename)

