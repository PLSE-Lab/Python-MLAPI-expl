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


train_data=pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.head()


# In[ ]:


test_data=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train_data


# In[ ]:


test_data


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


pd.set_option('precision', 2)


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(train_data.isnull(),cmap='viridis',yticklabels=False)


# In[ ]:


sns.countplot(train_data['Sex'])


# In[ ]:


sns.countplot(x='Survived',data=train_data,hue='Sex')


# In[ ]:


women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# In[ ]:


sns.countplot(x='Survived',data=train_data,hue='Pclass')


# In[ ]:


pclass1 = train_data.loc[train_data.Pclass == 1]['Survived']
rate_Pclass1 = sum(pclass1)/len(pclass1)

pclass2 = train_data.loc[train_data.Pclass == 2]['Survived']
rate_Pclass2 = sum(pclass2)/len(pclass2)

pclass3 = train_data.loc[train_data.Pclass == 3]['Survived']
rate_Pclass3 = sum(pclass3)/len(pclass3)

print("% of First P Class who survived:", rate_Pclass1)
print("% of Second P Class who survived:", rate_Pclass2)
print("% of Third P Class who survived:", rate_Pclass3)


# In[ ]:


sns.countplot(x='Survived',data=train_data,hue='SibSp')


# In[ ]:


sns.countplot(x='Survived',data=train_data,hue='Parch')


# In[ ]:


train_data['Fare'].hist(bins=40,figsize=(10,4))


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train_data)


# In[ ]:


train_data


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(),annot=True,cmap='viridis')


# In[ ]:


train_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train_data.corr()


# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train_data['Age']=train_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train_data['Embarked'].unique()


# In[ ]:


train_data.loc[train_data.Embarked.isnull()]


# In[ ]:


train_data.dropna(inplace=True)


# In[ ]:


sex=pd.get_dummies(train_data['Sex'],drop_first=True)


# In[ ]:


sex


# In[ ]:


embark=pd.get_dummies(train_data['Embarked'],drop_first=True)


# In[ ]:


train_data=pd.concat([train_data,sex,embark],axis=1)


# In[ ]:


train_data


# In[ ]:


train_data.drop(['Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[ ]:


X_train=train_data.drop(['Survived','PassengerId'],axis=1)
y_train=train_data['Survived']


# In[ ]:


train_data.head()


# In[ ]:


#X_train=X_train.drop(['Cabin'],axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data.loc[test_data.Embarked.isnull()]


# In[ ]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


test_data['Age']=test_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


test_data.groupby('Pclass').mean()['Fare']


# In[ ]:


def impute_fare(cols):
    Fare=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Fare):
        if Pclass==1:
            return 94.28
        elif Pclass==2:
            return 22.20
        else:
            return 12.45
    else:
        return Fare


# In[ ]:


test_data['Fare']=test_data[['Fare','Pclass']].apply(impute_fare,axis=1)


# In[ ]:


sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


test_data['Embarked'].unique()


# In[ ]:


test_sex=pd.get_dummies(test_data['Sex'],drop_first=True)


# In[ ]:


test_sex


# In[ ]:


test_embark=pd.get_dummies(test_data['Embarked'],drop_first=True)


# In[ ]:


test_data=pd.concat([test_data,test_sex,test_embark],axis=1)


# In[ ]:


test_data.drop(['Sex','Ticket','Embarked','Cabin'],axis=1,inplace=True)


# In[ ]:





# In[ ]:


test_data.head()


# In[ ]:


X_test=test_data.drop('PassengerId',axis=1)


# In[ ]:





# In[ ]:


X_test


# In[ ]:


X_train['Title']=X_train['Name'].apply(lambda name: name.split(',')[1].split()[0])
dummies=pd.get_dummies(X_train['Title'],drop_first=True)
X_train=pd.concat([X_train,dummies],axis=1)


# In[ ]:


X_test['Title']=X_test['Name'].apply(lambda name: name.split(',')[1].split()[0])
dummies=pd.get_dummies(X_test['Title'],drop_first=True)
X_test=pd.concat([X_test,dummies],axis=1)


# In[ ]:


X_train


# In[ ]:


X_train=X_train.drop('Name',axis=1)


# In[ ]:


X_train=X_train.drop('Title',axis=1)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


X_test=X_test.drop(['Name','Title'],axis=1)


# In[ ]:


X_test


# In[ ]:


X_train.info()


# In[ ]:


X_train=X_train.drop(['the','Sir.','Mlle.','Mme.','Major.','Jonkheer.','Col.','Lady.','Don.'],axis=1)


# In[ ]:


X_test=X_test.drop('Dona.',axis=1)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


X_train=X_train.values
y_train=y_train.values
X_test=X_test.values


# In[ ]:


X_test.shape


# In[ ]:





# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


X_train=scaler.fit_transform(X_train)


# In[ ]:


X_test=scaler.transform(X_test)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[ ]:





# In[ ]:


model=Sequential()
model.add(Dense(15,activation='relu'))

model.add(Dense(10,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')


# In[ ]:


model.fit(x=X_train,y=y_train,epochs=300)


# model_los=pd.DataFrame(model.history.history)

# In[ ]:


model_los=pd.DataFrame(model.history.history)


# In[ ]:


model_los.plot()


# In[ ]:


predictions=model.predict_classes(X_test)


# In[ ]:


pred=predictions.tolist()


# In[ ]:


pred=pd.Series(pred)


# In[ ]:


pred2=pred.apply(lambda x: x[0])


# In[ ]:


pred2


# In[ ]:


test_data.PassengerId


# In[ ]:


#from sklearn.linear_model import LogisticRegression


# In[ ]:


#logmodel=LogisticRegression()


# In[ ]:


#logmodel.fit(X_train,y_train)


# In[ ]:


#predictions=logmodel.predict(X_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred2})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




