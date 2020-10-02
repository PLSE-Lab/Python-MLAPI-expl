#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 


import pandas as pd 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.info()


# In[ ]:


total = train_df.isnull().sum() 
total


# In[ ]:


train_df = train_df.drop(['PassengerId'],axis=1)


# In[ ]:


test_passenger_id = pd.DataFrame(test_df.PassengerId)
test_passenger_id.head()


# In[ ]:


test_df=test_df.drop(['PassengerId'],axis=1)


# In[ ]:


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[ ]:


train_df.Age.fillna(train_df.Age.median(),inplace=True)
test_df.Age.fillna(test_df.Age.median(),inplace=True)


# In[ ]:


data = [train_df, test_df] 
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset["IsAlone"] = np.where(dataset["relatives"] > 0, 0,1)
train_df['IsAlone'].value_counts()  


# In[ ]:


for dataset in data:
    dataset.drop(['SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


top_value = 'S'
data = [train_df,test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(top_value)


# In[ ]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


gender = {'male':0,'female':1}
data = [train_df, test_df]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(gender)


# In[ ]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# In[ ]:


ports = {'S':0,'C':1,'Q':77}
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[ ]:


train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)


# In[ ]:


X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123)


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
accu_dt = dt.score(X_train,y_train)
accu_dt = round(accu_dt*100,2)
accu_dt


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
accu_rf = rf.score(X_train,y_train)
accu_rf = round(accu_rf*100,2)
accu_rf


# In[ ]:


y_final = rf.predict(test_df)
submission = pd.DataFrame({
    'PassengerId': test_passenger_id['PassengerId'],
    'Survived': y_final
})
submission.head()
submission.to_csv('titanic.csv', index=False)


# In[ ]:




