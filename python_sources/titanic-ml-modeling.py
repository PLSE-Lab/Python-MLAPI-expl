#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv", index_col='PassengerId')
train_data


# In[ ]:


for col in train_data.columns:
    print(col, len(train_data[col].unique()))


# In[ ]:


train_data.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
train_data


# In[ ]:


total = train_data.isnull().sum()

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1))

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data


# In[ ]:


age_mean = train_data.Age.mean()
train_data.Age.fillna(age_mean, inplace=True)

train_data.isna().sum()


# In[ ]:


train_data['Embarked'].describe()


# In[ ]:


train_data.Embarked.fillna(train_data.Embarked.mode()[0], inplace=True)
train_data.isna().sum()


# In[ ]:


train_data.Embarked.replace('C', 0, inplace=True)
train_data.Embarked.replace('S', 1, inplace=True)
train_data.Embarked.replace('Q', 2, inplace=True)

train_data


# In[ ]:


train_data.Age.dtype


# In[ ]:


test_data = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')


# In[ ]:


data = [train_data, test_data]

for dataset in data:
    dataset['Age'] = dataset['Age'].fillna(0)
    dataset['Age'] = dataset['Age'].astype(int)
    
train_data    


# In[ ]:


test_data.Age


# In[ ]:


train_data.info()


# In[ ]:


train_data.Sex.replace('male', 0, inplace=True)
train_data.Sex.replace('female', 1, inplace=True)

train_data


# In[ ]:


print(train_data.Sex.value_counts())
print('----------------------')
print(train_data.groupby('Sex').Survived.value_counts())


# **Chart says that more female passengers are survived compared to males..**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x='Sex', y='Survived', data=train_data)


# **Chart below describe how many passengers are there in every boarding location (Embarked)**

# In[ ]:


train_data.Embarked.value_counts().plot(kind='bar')
plt.title("Passengers per boarding location");


# **Information below shows the survival probability of each classes according to their gender..**

# In[ ]:


Survived_Pcalss = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_data, kind="bar")
Survived_Pcalss = Survived_Pcalss.set_ylabels("survival probability")


# In[ ]:


test_data = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')

test_data.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
test_data


# In[ ]:


test_data.isna().sum()


# In[ ]:


test_data.Age.fillna(age_mean, inplace=True)
test_data.Fare.fillna(test_data.Fare.mean(), inplace=True)

test_data


# In[ ]:


test_data.Sex.replace('male', 0, inplace=True)
test_data.Sex.replace('female', 1, inplace=True)

test_data.Embarked.replace('C', 0, inplace=True)
test_data.Embarked.replace('S', 1, inplace=True)
test_data.Embarked.replace('Q', 2, inplace=True)

test_data


# **Building ML Model**

# In[ ]:


X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]

X_train


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=1)

RFC_model = RandomForestClassifier(criterion='gini', 
                             n_estimators=100,
                             random_state=1,
                             n_jobs=-1)
RFC_model.fit(X_train, y_train)

y_prediction = RFC_model.predict(X_test)

RFC_model.score(X_train, y_train)
acc_RFC = round(RFC_model.score(X_train, y_train) * 100, 2)

acc_RFC


# In[ ]:


from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

y_predict = RFC_model.predict(X_test)
accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
y_prediction = decision_tree.predict(X_test) 

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree


# In[ ]:


y_predict = decision_tree.predict(X_test)
accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))


# In[ ]:


results = pd.DataFrame({
    'Model': ['Random Forest','Decision Tree'], 'Score': [acc_RFC, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(2)


# In[ ]:


f_model = DecisionTreeClassifier()
f_model.fit(X_train, y_train)

preds = f_model.predict(test_data)

test_data.shape


# In[ ]:


test_output = pd.DataFrame({
    'PassengerId': test_data.index, 
    'Survived': preds
})
test_output.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')

submission

