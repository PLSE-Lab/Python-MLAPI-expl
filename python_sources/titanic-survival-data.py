#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Competiton

# ## Table of Contents
# 
# 
# * [Import Libraries](#import)
# * [Going through the Data](#going_data)
# * [Data Cleaning](#cleaning)
# * [Feature Engineering](#feature)
# * [Analysis](#analysis)
#     - [Visualizations](#visual)
#     - [Models](#models)
#         - [Random Forest](#random)
#         - [Gradient](#gradient)
#         - [Logistic](#logistic)
#         - [Decision Tree](#decision)
#     - [Final Accuracies](#final)
# * [Submisson](#submission)    
#     

# <a id="import"></a>
# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor


labelencoder = LabelEncoder()
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_boston
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "going_data"></a>
# # Going through the Data

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# <a id = "cleaning"></a>
# # Data Cleaning

# In[ ]:


cols = train_data.columns[:30] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(train_data[cols].isnull(), cmap=sns.color_palette(colours))


# <a id = "feature"></a>
# # Feature Engineering

# In[ ]:


train_data['Title'] = train_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
test_data['Title'] = test_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train_data.head()
test_data.head()


# Droping Missing Data of cabin and replaceing missing age with median

# In[ ]:


train_data.drop('Cabin', axis=1, inplace=True)
train_data.drop('Name', axis=1, inplace=True)
#train_data.drop('Embarked', axis=1, inplace=True)

med = train_data['Age'].median()
print(med)
train_data['Age'] = train_data['Age'].fillna(med)
train_data.head()


# In[ ]:


test_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)
#test_data.drop('Embarked', axis=1, inplace=True)

med = test_data['Age'].median()
print(med)
test_data['Age'] = test_data['Age'].fillna(med)
test_data.head()


# In[ ]:


test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data.isnull().sum()


# In[ ]:


def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df


# In[ ]:


train_data = dummyEncode(train_data)
train_data.Embarked.head()


# In[ ]:


test_data = dummyEncode(test_data)
test_data.head()


# <a id = "analysis"></a>
# # Analysis

# <a id = "visual"></a>
# ## Visualizations

# In[ ]:


sns.boxplot(x="Survived", y="Fare", data=train_data)


# In[ ]:


sns.boxplot(x="Title", y="Fare", data=train_data)


# In[ ]:


sns.boxplot(x="Title", y="Embarked", data=train_data)


# In[ ]:


plt.figure(figsize=(5,5))
barSvS = sns.countplot(x = 'Survived', hue = 'Sex', data = train_data)
plt.title("SURVIVED AND SEX",size=15)
barSvS.legend(["Male", "Female"])


# In[ ]:


plt.figure(figsize=(5,5))
barSvP =sns.countplot(x = 'Survived', hue = 'Pclass', data = train_data)
plt.title("SURVIVED AND PCLASS",size=15)
barSvP.legend(["First Class", "Second Class", "Third Class"])


# In[ ]:


plt.figure(figsize=(5,5))
barSvP =sns.countplot(x = 'Survived', hue = 'Embarked', data = train_data)
plt.title("SURVIVED AND EMBARKED",size=15)


# In[ ]:


sns.relplot(x="Embarked", y="Sex", hue="Survived", data=train_data)


# In[ ]:


sns.relplot(x="Embarked", y="Title", hue="Survived", data=train_data)


# <a id = "models"></a>
# ## Models

# In[ ]:


dataset = pd.get_dummies(train_data, columns = ["Title", "Embarked"])
train_data.Survived = train_data.Survived.astype('int')

train = dataset[:len(train_data)]
test = dataset[len(test_data):]

test.drop(labels=['Survived'], axis=1, inplace=True)
y=train.Survived
X=train.drop('Survived', axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 473, random_state = 2)


# <a id = "random"></a>
# ### Random Forest

# In[ ]:


model_1 = RandomForestClassifier(n_estimators=100)
model_1.fit(X_train, y_train)

predict1 = model_1.predict(X_val)
acuracy1 = accuracy_score(predict1, y_val)
print('Accuracy: ', acuracy1)


# <a id = "gradient"></a>
# ### Gradient Regression

# In[ ]:


model_2 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
model_2.fit(X_train, y_train)

predict2 = model_2.predict(X_val)
acuracy2 = accuracy_score(predict2, y_val)
print('Accuracy: ', acuracy2)


# <a id = "logistic"></a>
# ### Logistic Regression 

# In[ ]:


model_3 = LogisticRegression(random_state=0)
model_3.fit(X_train, y_train)

predict3 = model_3.predict(X_val)
acuracy3 = accuracy_score(predict3, y_val)
print('Accuracy: ', acuracy3)


# <a id = "decision"></a>
# ### Decision Tree

# In[ ]:


model_4 =  DecisionTreeClassifier()
model_4.fit(X_train, y_train)

predict4 = model_4.predict(X_val)
acuracy4 = accuracy_score(predict4, y_val)
print('Accuracy: ', acuracy4)


# <a id = "final"></a>
# ## Final Accuracies 

# In[ ]:


df = pd.DataFrame({'Random Forest': acuracy1, 'Gradient': acuracy2, 'Logistic': acuracy3, ' Decision Tree': acuracy4} , index=[0])
df.rename(index={0:'Accuracy'}, inplace=True)
df


# <a id = "submission"></a>
# # Submission

# In[ ]:


y_train = train_data['Survived']
X_train = train_data[['Title','Embarked', 'Pclass', 'Age', 'Sex']]

X_test = test_data[['Title','Embarked', 'Pclass', 'Age', 'Sex']]
selected_columns = X_train[['Title', 'Pclass', 'Age', 'Sex']]
df1 = selected_columns.copy()
df2 = []
X_train['Embarked'] = X_train['Embarked'].fillna('S')
for dataset in X_train['Embarked']:
    if dataset == 'S':
        df2.append(0)
    if dataset == 'C':
        df2.append(1)
    if dataset == 'Q':
        df2.append(2)

df1['Embarked'] = df2
y_train = y_train.reindex(X_test.index)
X_train = X_train.reindex(X_test.index)
df1 = X_test.reindex(X_test.index)


# In[ ]:


final_model = GradientBoostingClassifier(n_estimators=419, max_depth=7, learning_rate=0.1)
final_model.fit(df1, y_train)

final_predictions = final_model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predictions})
final_accuracy = accuracy_score(final_predictions, y_train)
print('Accuracy: ', final_accuracy)

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
output.head()

