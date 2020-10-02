#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
train = pd.read_csv('../input/train.csv')
train.head()


# Exploratory Data Analysis: Missing Data

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False)


# Filling the missing data: Age 

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
    else: return Age
    
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False)


# Removing data: Cabin

# In[ ]:


train.drop('Cabin',axis = 1, inplace = True)


# Converting Categorical Features

# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'], axis =1, inplace = True)
train = pd.concat([train,sex, embark],axis = 1)
train.head()


# Training using Decision Tree

# In[ ]:


from sklearn.model_selection import KFold


# 10-Fold Cross Validation

# In[ ]:


kf=KFold(n_splits=10)


# In[ ]:


train1=train.drop('Survived', axis = 1)
labels=train['Survived']

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier()

kf=KFold(n_splits=10, shuffle=True, random_state=False)
outcomes=[]
for train_id, test_id in kf.split(train1,labels):
    X_train, X_test = train1.values[train_id], train1.values[test_id]
    y_train, y_test = labels.values[train_id], labels.values[test_id]
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
plt.plot(range(10),outcomes)
print(np.mean(outcomes))


# Training using Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
Rf=RandomForestClassifier(n_estimators=400)
outcomesRf=[]
for train_id, test_id in kf.split(train1,labels):
    X_train, X_test = train1.values[train_id], train1.values[test_id]
    y_train, y_test = labels.values[train_id], labels.values[test_id]
    Rf.fit(X_train,y_train)
    predictions = Rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomesRf.append(accuracy)
plt.plot(range(10),outcomesRf)
plt.ylabel=('accuracy')
print(np.mean(outcomesRf))


# For submission, import and process test dataset

# In[ ]:


test = pd.read_csv('../input/test.csv')
#sns.heatmap(test.isnull(),yticklabels=False)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis = 1)
test.drop('Cabin',axis = 1, inplace = True)
#sns.heatmap(test.isnull(),yticklabels=False)
sex = pd.get_dummies(test['Sex'], drop_first = True)
embark = pd.get_dummies(test['Embarked'],drop_first = True)
test.drop(['Sex','Embarked','Name','Ticket'], axis =1, inplace = True)
test = pd.concat([test,sex, embark],axis = 1)
test.dropna(axis=0, how='all')
Fare_avg = test['Fare'].mean()
def impute_Fare(col):    
    if pd.isnull(col):        
            return Fare_avg
    else: return col
test['Fare'] = test['Fare'].apply(impute_Fare)


# In[ ]:


Rf.fit(train1,labels)
predictions = Rf.predict(test)
Survived=pd.DataFrame(data=predictions,columns=['Survived'])
pred_final=pd.concat([test['PassengerId'],Survived],axis =1)
pred_final.to_csv('Titanic_predict_Random_Forest.csv',index=False)


# In[ ]:




