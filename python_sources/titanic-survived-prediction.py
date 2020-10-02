#!/usr/bin/env python
# coding: utf-8

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train_data)


# In[ ]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# In[ ]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# In[ ]:


#visualizing % of Male and female survived
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train_data)


# In[ ]:


#cheking for missing values
train_data.isnull().sum()


# In[ ]:


#cheking for missing values
test_data.isnull().sum()


# **Data Cleaning**

# In[ ]:


sns.distplot(train_data['Age'].dropna(), kde=False, color='blue', bins=40)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train_data)


# In[ ]:


#visualization for relationship between Age and Pclass
plt.figure(figsize=(15,6))
sns.boxplot(x='Pclass', y='Age', data = train_data, palette='winter')


# In[ ]:


#since the age is corealted with Pclass we have to add mean or average for the age

def  age_impute(cols):
  Age = cols[0]
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


#impute age
train_data['Age'] = train_data[['Age', 'Pclass']].apply(age_impute, axis=1)
test_data['Age'] = train_data[['Age', 'Pclass']].apply(age_impute, axis=1)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


#droping cabin coloum in train and test set 
train_data.drop('Cabin',axis=1, inplace=True)
test_data.drop('Cabin',axis=1, inplace=True)


# In[ ]:


train_data['Embarked'].isnull().sum()


# In[ ]:


train_data['Embarked']=train_data['Embarked'].replace(np.NaN, train_data['Embarked'].mode())


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data['Fare'].mean()


# In[ ]:


test_data['Fare']=test_data['Fare'].replace(np.NaN, train_data['Fare'].mean())


# In[ ]:


test_data.isnull().sum()


# In[ ]:


#caring categorical values -----Train set------
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embarked = pd.get_dummies(train_data['Embarked'],drop_first=True)


# In[ ]:


print(sex.head())
print(embarked.head())


# In[ ]:


#droping unwanted columns
train_data.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


#Concat
train_data = pd.concat([train_data,sex,embarked],axis=1)
train_data.head()


# In[ ]:


#caring categorical values -----Test set------
gender = pd.get_dummies(test_data['Sex'],drop_first=True)
embark = pd.get_dummies(test_data['Embarked'],drop_first=True)


# In[ ]:


print(gender.head())
print(embark.head())


# In[ ]:


#droping unwanted columns
test_data.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
test_data.head()


# In[ ]:


#Concat
test_data = pd.concat([test_data,gender,embark],axis=1)
test_data.head()


# Train test split

# In[ ]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['Survived','PassengerId'], axis=1), 
                                                    train_data['Survived'], test_size = 0.2, 
                                                    random_state = 0)


# **Random Forest**

# In[ ]:


#model building
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500, criterion='entropy')
classifier.fit(X_train,y_train)


# In[ ]:


predictions = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
cm


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
accuracy


# **XGBoot**

# In[ ]:


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train,y_train)


# In[ ]:


xgb_predictions = xgb_classifier.predict(X_test)


# In[ ]:


#confusion matrix
xgb_cm = confusion_matrix(y_test, xgb_predictions)
xgb_cm


# In[ ]:


#checking accuracy
xgb_accuracy = accuracy_score(y_test,xgb_predictions)
xgb_accuracy


# **Gradient Boosting Classifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc= GradientBoostingClassifier()
gbc.fit(X_train,y_train)


# In[ ]:


#prediciion
gbc_predictions = gbc.predict(X_test)


# In[ ]:


#confusion_matrix
gbc_cm = confusion_matrix(y_test, gbc_predictions)
gbc_cm


# In[ ]:


gbc_accuracy = accuracy_score(y_test, gbc_predictions)
gbc_accuracy


# In[ ]:


from sklearn.model_selection import cross_val_score
crossval = cross_val_score(estimator = classifier , X= X_train ,y= y_train, cv = 10)
crossval.mean()


# In[ ]:


#checking accuracy of each model
models={'MODEL':['RANDOM FOREST','XG BOOSTING','GRADIENT BOOSTING'],'ACCURACY':[accuracy,xgb_accuracy,crossval.mean()]}
model_accuracy=pd.DataFrame(models)
model_accuracy


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# **Submission(choosing GRADIENT BOOSTING	)**

# In[ ]:


passenger_id = test_data['PassengerId']
predict_values = gbc.predict(test_data.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : passenger_id, 'Survived': predict_values })
output.to_csv('submission.csv', index=False)


# In[ ]:


predictions

