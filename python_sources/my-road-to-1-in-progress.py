#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# [Link to Kaggle Competition](https://www.kaggle.com/c/titanic)
# 
# Author: Diego Rodrigues [@polaroidz](https://github.com/polaroidz)

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings # just for ignoring annoying warnings
warnings.filterwarnings('ignore')


# In[ ]:


TARGET = 'Survived'
ID = 'PassengerId'


# In[ ]:


dataset = pd.read_csv('../input/train.csv')


# In[ ]:


dataset.head()


# ### Numerical Variables

# In[ ]:


desc = dataset.describe()
desc


# In[ ]:


numerical_columns = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]


# In[ ]:


for col in numerical_columns:
    dataset[col].hist()
    plt.title(col + " distribution")
    plt.show();
    
    st = dataset[[col, TARGET]].groupby(col, as_index=False).mean()
    plt.bar(st[col], st[TARGET])
    plt.title(col + " survival count")
    plt.show();

del st


# In[ ]:


numerical_dataset = numerical_columns + [TARGET]
numerical_dataset = dataset[numerical_dataset]
numerical_dataset.corr()


# In[ ]:


numerical_dataset = numerical_dataset.drop('Survived', axis=1)


# In[ ]:


numerical_dataset.head()


# ### Categorical Variables

# In[ ]:


categorical_variables = [
    'Sex',
    'Embarked'
]


# In[ ]:


categorical_dataset = dataset[categorical_variables + [TARGET]]
categorical_dataset.head()


# In[ ]:


for col in categorical_dataset.columns:
    print(categorical_dataset[col].value_counts())
    print("\n")


# In[ ]:


categorical_dataset = categorical_dataset.drop('Survived', axis=1)


# ### Feature Engineering - Numerical

# In[ ]:


num_feat_dataset = pd.DataFrame()


# In[ ]:


num_feat_dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[ ]:


num_feat_dataset['FamilySize'].hist()


# ### Feature Engineering - Categorical

# In[ ]:


cat_feat_dataset = pd.DataFrame()


# In[ ]:


cat_feat_dataset[TARGET] = dataset[TARGET]


# In[ ]:


cat_feat_dataset['IsAlone'] = np.ones(dataset.shape[0])


# In[ ]:


cat_feat_dataset['IsAlone'].loc[num_feat_dataset['FamilySize'] > 1] = 0


# In[ ]:


cat_feat_dataset['AgeBin'] = pd.qcut(dataset['Age'].fillna(dataset['Age'].mean()).astype(int), 5)


# In[ ]:


cat_feat_dataset['FareBin'] = pd.qcut(dataset['Fare'].fillna(dataset['Fare'].mean()).astype(int), 4)


# In[ ]:


cat_feat_dataset['Title'] = dataset['Name'].str.extract(r',\s([A-Z].*)\.')


# In[ ]:


cat_feat_dataset['Title'].value_counts()


# In[ ]:


conditions = [
    cat_feat_dataset['Title'] == 'Mr',
    cat_feat_dataset['Title'] == 'Mrs',
    cat_feat_dataset['Title'] == 'Miss',
    cat_feat_dataset['Title'] == 'Master',
    cat_feat_dataset['Title'] == 'Dr',
    cat_feat_dataset['Title'] == 'Rev'
]

choices = [6, 5, 4, 3, 2, 1]

cat_feat_dataset['Title'] = np.select(conditions, choices, default=0)


# In[ ]:


st = cat_feat_dataset[['Title', TARGET]].groupby('Title', as_index=False).mean()
plt.bar(st['Title'], st[TARGET]);


# In[ ]:


dataset['Cabin'][dataset['Cabin'].isna()].shape


# In[ ]:


cat_feat_dataset['HasCabinInfo'] = dataset['Cabin'].isnull()


# In[ ]:


st = cat_feat_dataset[['HasCabinInfo', TARGET]].groupby('HasCabinInfo', as_index=False).mean()
plt.bar(st['HasCabinInfo'], st[TARGET]);


# In[ ]:


cat_feat_dataset['Deck'] = dataset['Cabin'].str.slice(0,1)


# In[ ]:


cat_feat_dataset['Deck'].value_counts()


# In[ ]:


st = cat_feat_dataset[['Deck', TARGET]].groupby('Deck', as_index=False).mean()
plt.bar(st['Deck'], st[TARGET]);


# In[ ]:


conditions = [
    (cat_feat_dataset['Deck'] == 'A') | (cat_feat_dataset['Deck'] == 'B') | (cat_feat_dataset['Deck'] == 'C'),
    (cat_feat_dataset['Deck'] == 'D') | (cat_feat_dataset['Deck'] == 'E'),
    (cat_feat_dataset['Deck'] == 'F') | (cat_feat_dataset['Deck'] == 'G') | (cat_feat_dataset['Deck'] == 'T')
]

choices = [1, 2, 3]

cat_feat_dataset['Deck'] = np.select(conditions, choices, default=0)


# In[ ]:


cat_feat_dataset['CabinPos'] = dataset["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")


# In[ ]:


cat_feat_dataset['CabinPos'].hist()


# In[ ]:


cat_feat_dataset['CabinPos'] = pd.qcut(cat_feat_dataset['CabinPos'], 3)


# In[ ]:


cat_feat_dataset['CabinPos'].value_counts()


# In[ ]:


cat_feat_dataset[['CabinPos', TARGET]].groupby('CabinPos', as_index=False).mean()


# In[ ]:


cat_feat_dataset[['CabinPos', 'Deck', TARGET]].groupby(['CabinPos', 'Deck'], as_index=False).mean()


# In[ ]:


cat_feat_dataset = cat_feat_dataset.drop(TARGET, axis=1)


# In[ ]:


cat_feat_dataset['CabinPos'] = cat_feat_dataset['CabinPos'].astype(str)


# In[ ]:


cat_feat_dataset.head()


# In[ ]:


numerical_dataset = pd.concat([numerical_dataset, num_feat_dataset], axis=1)
categorical_dataset = pd.concat([categorical_dataset, cat_feat_dataset], axis=1)


# ### Missing Values

# In[ ]:


for col in categorical_dataset.columns:
    print(col)
    print(categorical_dataset[categorical_dataset[col].isna()].index)


# In[ ]:


for col in numerical_dataset.columns:
    print(col)
    print(numerical_dataset[numerical_dataset[col].isna()].index)


# In[ ]:


dataset[dataset['Age'].isna()].head()


# In[ ]:


numerical_dataset['Age'] = numerical_dataset['Age'].fillna(dataset['Age'].mean())


# In[ ]:


rows_to_drop = [
    61, 829
]


# In[ ]:


categorical_dataset = categorical_dataset.drop(categorical_dataset.index[rows_to_drop])


# In[ ]:


numerical_dataset = numerical_dataset.drop(numerical_dataset.index[rows_to_drop])


# ### Applying Transformations

# In[ ]:


numerical_scaler = StandardScaler()
numerical_scaler.fit(numerical_dataset)


# In[ ]:


numerical_dataset = numerical_scaler.transform(numerical_dataset)
numerical_dataset = pd.DataFrame(numerical_dataset)


# In[ ]:


one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(categorical_dataset)


# In[ ]:


categorical_dataset = one_hot_encoder.transform(categorical_dataset)


# In[ ]:


categorical_dataset = pd.DataFrame(categorical_dataset.toarray())


# ### Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


X = pd.concat([numerical_dataset, categorical_dataset], axis=1)
y = dataset[TARGET]


# In[ ]:


y = y.drop(y.index[rows_to_drop])


# In[ ]:


X = X.fillna(0)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
xgb = XGBClassifier()
gbc = GradientBoostingClassifier()
svc = SVC(probability=True)
ext = ExtraTreesClassifier()
ada = AdaBoostClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier()


# In[ ]:


X_train.columns = range(X_train.columns.shape[0])


# In[ ]:


models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
scores = []
names = []

for model in models:
    model_name = model.__class__.__name__
    
    model.fit(X_train, y_train)
    acc = cross_val_score(model, X_train, y_train, scoring = "accuracy", cv = 10)
    scores.append(acc.mean())
    names.append(model_name)


# In[ ]:


results = pd.DataFrame({
    'Model': names,
    'Score': scores
})
results = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
results.head(len(models))


# In[ ]:


final_model = VotingClassifier(
    estimators=[(model.__class__.__name__, model) for model in models],
    voting='soft'
)


# In[ ]:


final_model.fit(X_train, y_train)


# In[ ]:


import pickle


# In[ ]:


with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, file=f)


# ### Submission

# In[ ]:


test_dataset = pd.read_csv('../input/test.csv')


# In[ ]:


test_numset = test_dataset[numerical_columns]
test_numset['FamilySize'] = test_numset['SibSp'] + test_numset['Parch'] + 1


# In[ ]:


test_catset = test_dataset[categorical_variables]
test_catset['IsAlone'] = np.ones(test_dataset.shape[0])
test_catset['IsAlone'].loc[test_numset['FamilySize'] > 1] = 0

test_catset['AgeBin'] = pd.qcut(test_numset['Age'].fillna(test_numset['Age'].mean()).astype(int), 5)
test_catset['FareBin'] = pd.qcut(test_numset['Fare'].fillna(test_numset['Fare'].mean()).astype(int), 4)

test_catset['Title'] = test_dataset['Name'].str.extract(r',\s([A-Z].*)\.')
conditions = [
    test_catset['Title'] == 'Mr',
    test_catset['Title'] == 'Mrs',
    test_catset['Title'] == 'Miss',
    test_catset['Title'] == 'Master',
    test_catset['Title'] == 'Dr',
    test_catset['Title'] == 'Rev'
]

choices = [6, 5, 4, 3, 2, 1]

test_catset['Title'] = np.select(conditions, choices, default=0)

test_catset['HasCabinInfo'] = test_dataset['Cabin'].isnull()

test_catset['Deck'] = test_dataset['Cabin'].str.slice(0,1)

conditions = [
    (test_catset['Deck'] == 'A') | (test_catset['Deck'] == 'B') | (test_catset['Deck'] == 'C'),
    (test_catset['Deck'] == 'D') | (test_catset['Deck'] == 'E'),
    (test_catset['Deck'] == 'F') | (test_catset['Deck'] == 'G') | (test_catset['Deck'] == 'T')
]

choices = [1, 2, 3]

test_catset['Deck'] = np.select(conditions, choices, default=0)

test_catset['CabinPos'] = test_dataset["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
test_catset['CabinPos'] = pd.qcut(test_catset['CabinPos'], 3)
test_catset['CabinPos'] = test_catset['CabinPos'].astype(str)


# In[ ]:


test_numset = numerical_scaler.transform(test_numset)
test_numset = pd.DataFrame(test_numset)


# In[ ]:


test_catset = one_hot_encoder.transform(test_catset)
test_catset = pd.DataFrame(test_catset.toarray())


# In[ ]:


X = pd.concat([test_numset, test_catset], axis=1)


# In[ ]:


X = X.fillna(0)


# In[ ]:


X.columns = range(X.columns.shape[0])


# In[ ]:


y = final_model.predict(X)


# In[ ]:


y.shape, test_dataset['PassengerId'].shape


# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test_dataset['PassengerId'],
    'Survived': y
})
submission.head()


# In[ ]:


submission.to_csv('./submission.csv', index=False)


# In[ ]:




