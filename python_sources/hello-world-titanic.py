#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This is my first work of machine learning. the notebook is written in python

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import re as re
from random import randint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# ### Load in the train datasets

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.head(3)


# ### Load in the test datasets

# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
test.head(3)


# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]

y_train = train['Survived'].values
passengerId = test['PassengerId']

dataset = pd.concat((train, test))


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe(include='all')


# # Feature Engineering

# ## Pclass

# In[ ]:


dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[ ]:


pd.crosstab(dataset.Pclass, dataset.Survived, margins=True)


# Pclass 1 and 2 have similar values for survived

# ## Sex

# In[ ]:


dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# ## SibSp and Parch

# In[ ]:


dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()


# In[ ]:


dataset['IsAlone'] = 0
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
dataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# ## Fare

# In[ ]:


dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
dataset['Fgroup'] = pd.qcut(dataset['Fare'], 10, labels=range(10))
dataset[['Fgroup', 'Survived']].groupby(['Fgroup'], as_index=False).mean()


# ## Name

# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''


dataset['Title'] = dataset['Name'].apply(get_title)
pd.crosstab(dataset['Title'], dataset['Sex'])


# In[ ]:


dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


def get_initial_name(name):
    initial_search = re.search('([A-Za-z]+)', name)
    if initial_search:
        return initial_search.group(1)
    return ''


dataset['LastName'] = dataset['Name'].apply(get_initial_name)
dataset['NumName'] = dataset['LastName'].factorize()[0]
    
dataset[['NumName', 'Survived']].groupby(['NumName'], as_index=False).mean()


# ## Age

# In[ ]:


print('Oldest Passenger was', dataset['Age'].max(), 'Years')
print('Youngest Passenger was', dataset['Age'].min(), 'Years')
print('Average Age on the ship was', int(dataset['Age'].mean()), 'Years')


# In[ ]:


dataset.groupby('Title').agg({'Age': ['mean', 'count']})


# In[ ]:


dataset = dataset.reset_index(drop=True)
dataset['Age'] = dataset.groupby('Title')['Age'].apply(lambda x: x.fillna(x.mean()))


# In[ ]:


dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona' , 'Dr', 'Jonkheer', 'Lady', 
                                             'Major', 'Master',  'Miss'  ,'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'], 
                                            ['Sacrificed', 'Respected', 'Nobles', 'Mr', 'Mrs', 'Respected', 'Mr', 'Nobles', 
                                             'Respected', 'Kids', 'Miss', 'Nobles', 'Nobles', 'Mr', 'Mrs', 'Nobles', 'Sacrificed', 'Nobles'])
dataset['Title'] = dataset['Title'].replace(['Kids', 'Miss', 'Mr', 'Mrs', 'Nobles', 'Respected', 'Sacrificed'], [4, 4, 2, 5, 6, 3, 1])


# In[ ]:


dataset['TempAgroup'] = pd.qcut(dataset['Age'], 10)

dataset[['TempAgroup', 'Survived']].groupby(['TempAgroup'], as_index=False).mean()


# In[ ]:


dataset['Agroup'] = pd.qcut(dataset['Age'], 10, labels=range(10))
dataset[['Agroup', 'Survived']].groupby(['Agroup'], as_index=False).mean()


# In[ ]:


pd.crosstab(dataset.Pclass, dataset.Agroup, margins=True)


# # Data Cleaning

# ### Mapping Gender class

# In[ ]:


dataset['Gclass'] = 0
dataset.loc[((dataset['Sex'] == 'male') & (dataset['Pclass'] == 1)), 'Gclass'] = 1
dataset.loc[((dataset['Sex'] == 'male') & (dataset['Pclass'] == 2)), 'Gclass'] = 2
dataset.loc[((dataset['Sex'] == 'male') & (dataset['Pclass'] == 3)), 'Gclass'] = 2
dataset.loc[((dataset['Sex'] == 'female') & (dataset['Pclass'] == 1)), 'Gclass'] = 3
dataset.loc[((dataset['Sex'] == 'female') & (dataset['Pclass'] == 2)), 'Gclass'] = 4
dataset.loc[((dataset['Sex'] == 'female') & (dataset['Pclass'] == 3)), 'Gclass'] = 5
dataset.loc[(dataset['Age'] < 1), 'Gclass'] = 6


# ### Mapping Sex

# In[ ]:


dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# ### Mapping Priority
# 1. Nobles 
# 2. Women in Pclass 1 
# 3. Babies under 1 
# 4. Kids under 17 in Pclass 1
# 5. Kids under 17 in Pclass 2 
# 6. Women in Pclass 3

# In[ ]:


dataset['Priority'] = 0
dataset.loc[(dataset['Title'] == 6), 'Priority'] = 1
dataset.loc[(dataset['Gclass'] == 3), 'Priority'] = 2
dataset.loc[(dataset['Gclass'] == 6), 'Priority'] = 3
dataset.loc[(dataset['Pclass'] == 1) & (dataset['Age'] <= 17), 'Priority'] = 4
dataset.loc[(dataset['Pclass'] == 2) & (dataset['Age'] <= 17), 'Priority'] = 5
dataset.loc[(dataset['Pclass'] == 3) & (dataset['Sex'] == 1), 'Priority'] = 6
dataset.loc[(dataset['Fgroup'] == 9), 'Priority'] = 7


# ### Mapping FH
# Female Higher Survival Group

# In[ ]:


dataset['FH'] = 0
dataset.loc[(dataset['Gclass'] == 1), 'FH'] = 0
dataset.loc[(dataset['Gclass'] == 2), 'FH'] = 0
dataset.loc[(dataset['Gclass'] == 3), 'FH'] = 1
dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 2), 'FH'] = 2
dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 3), 'FH'] = 3
dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 4), 'FH'] = 4
dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 1), 'FH'] = 5
dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 2), 'FH'] = 6
dataset.loc[(dataset['Gclass'] == 4) & (dataset['Fgroup'] == 3), 'FH'] = 7
dataset.loc[(dataset['Gclass'] == 4) & (dataset['Fgroup'] >= 5), 'FH'] = 8


# ### Mapping MH
# Male Higher Survival Group

# In[ ]:


dataset['MH'] = 0
dataset.loc[(dataset['Sex'] == 1), 'MH'] = 0
dataset.loc[(dataset['Gclass'] == 1), 'MH'] = 1
dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 2), 'MH'] = 2
dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 3), 'MH'] = 3
dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 4), 'MH'] = 4
dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 1), 'MH'] = 5
dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 2), 'MH'] = 6
dataset.loc[(dataset['Gclass'] == 1) & (dataset['Fgroup'] == 3), 'MH'] = 7
dataset.loc[(dataset['Gclass'] == 1) & (dataset['Fgroup'] >= 5), 'MH'] = 8


# ### Mapping FL
# Female Lower Surival Group

# In[ ]:


dataset['FL'] = 0
dataset.loc[(dataset['Gclass'] != 5), 'FL'] = 0
dataset.loc[(dataset['Gclass'] == 5) & (dataset['Fgroup'] < 5), 'FL'] = 1
dataset.loc[(dataset['Gclass'] == 5) & (dataset['Fgroup'] != 3), 'FL'] = 2
dataset.loc[(dataset['Gclass'] == 5) & (dataset['FH'] == 1), 'FL'] = 3
dataset.loc[(dataset['Gclass'] == 5) & (dataset['FamilySize'] < 2), 'FL'] = 4
dataset.loc[(dataset['Gclass'] == 5) & (dataset['FamilySize'] > 4), 'FL'] = 5
dataset.loc[(dataset['Gclass'] == 5) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 3), 'FL'] = 6


# ### Mapping FH
# Female Higher Survival Group

# In[ ]:


dataset['ML'] = 0
dataset.loc[(dataset['Gclass'] == 2) & (dataset['Fgroup'] < 5), 'ML'] = 1
dataset.loc[(dataset['Gclass'] == 2) & (dataset['Fgroup'] != 3), 'ML'] = 2
dataset.loc[(dataset['Gclass'] == 2) & (dataset['MH'] < 7), 'ML'] = 3
dataset.loc[(dataset['Gclass'] == 2) & (dataset['FamilySize'] < 2), 'ML'] = 4
dataset.loc[(dataset['Gclass'] == 2) & (dataset['FamilySize'] > 4), 'ML'] = 5
dataset.loc[(dataset['Gclass'] == 2) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 3), 'ML'] = 6
dataset.loc[(dataset['Gclass'] == 3) & (dataset['Fgroup'] < 5), 'ML'] = 1
dataset.loc[(dataset['Gclass'] == 3) & (dataset['Fgroup'] != 3), 'ML'] = 2
dataset.loc[(dataset['Gclass'] == 3) & (dataset['MH'] < 7), 'ML'] = 3
dataset.loc[(dataset['Gclass'] == 3) & (dataset['FamilySize'] < 2), 'ML'] = 4
dataset.loc[(dataset['Gclass'] == 3) & (dataset['FamilySize'] > 4), 'ML'] = 5
dataset.loc[(dataset['Gclass'] == 3) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 3), 'ML'] = 6


# In[ ]:


dataset.columns


# In[ ]:


dfl = pd.DataFrame()
good_columns = ['Priority', 'Gclass', 'Title','NumName', 'FL','IsAlone','ML', 'FH', 'MH', 'Fgroup', 'FamilySize']
dfl[good_columns] = dataset[good_columns]


# In[ ]:


corrMatrix = pd.concat([dfl[:ntrain], train['Survived']], axis=1).corr()
fig, ax = plt.subplots(figsize=(10,10))     
sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
plt.show()


# In[ ]:


dfl = pd.DataFrame()
good_columns = ['Priority', 'Gclass', 'Title','NumName', 'FL','IsAlone','ML', 'FH', 'MH', 'Fgroup', 'FamilySize']
dfl[good_columns] = dataset[good_columns]


# In[ ]:


corrMatrix = pd.concat([dfl[:ntrain], train['Survived']], axis=1).corr()
fig, ax = plt.subplots(figsize=(10,10))     
sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
plt.show()


# In[ ]:


dfh = dfl.copy()
dfl_enc = dfl.apply(LabelEncoder().fit_transform)
one_hot_cols = dfh.columns.tolist()
dfh_enc = pd.get_dummies(dfh, columns=one_hot_cols)


# In[ ]:


X_train = dfh_enc[:ntrain]
X_test = dfh_enc[ntrain:]


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Prediction

# In[ ]:


model = SVC(probability=True, gamma=0.001, C=10)
scores = cross_val_score(model, X_train, y_train, cv=7)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


model = SVC(probability=True, gamma=0.001, C=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': passengerId, 'Survived': predictions})
output


# In[ ]:


output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

