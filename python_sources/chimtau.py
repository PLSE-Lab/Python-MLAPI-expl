#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
train_df = pd.read_csv("/kaggle/input/train.csv")
test_df = pd.read_csv("/kaggle/input/test.csv")


# ## Iteration 0: Constant prediction
# 
# A very simple model which returns a constant prediction. From the bellow statistics, we pick `Survived=0`.
# 

# In[ ]:


train_df["Survived_cat"] = train_df["Survived"].astype('category')
train_df.describe(include="all")


# In[ ]:


prediction = pd.DataFrame()
prediction['PassengerId'] = test_df['PassengerId'].copy()
prediction['Survived'] = 0
prediction.to_csv("submission.csv", index=False, sep=",", header=True)


# ## Iteration 1: Logistic regression

# In[ ]:


from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.utils import shuffle

def dataset(dataframe):
    try:
        Y = dataframe['Survived'].copy()
    except:
        Y = None
    X = dataframe[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].copy()
    X['Age'] = X['Age'].fillna(value=29.7)
    X = X.fillna(method='ffill')
    return X,Y

train_df = pd.get_dummies(train_df, columns=["Pclass", "Embarked"])
test_df = pd.get_dummies(test_df, columns=["Pclass", "Embarked"])
train_df = shuffle(train_df).reset_index(drop=True)
train_X, train_Y = dataset(train_df[0:600])
val_X  , val_Y   = dataset(train_df[600:])
test_X, _ = dataset(test_df)

normalizer = preprocessing.StandardScaler().fit(train_X.append(val_X).append(test_X) )
train_X = normalizer.transform(train_X)
val_X = normalizer.transform(val_X)
test_X = normalizer.transform(test_X)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(random_state=2, solver='lbfgs', max_iter=300, C=0.1).fit(train_X, train_Y)
val_pred = clf.predict(val_X) 
accuracy_score(val_Y, val_pred)


# In[ ]:


final_train_X, final_train_Y = dataset(train_df)
clf = LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000, C=0.01).fit(final_train_X, final_train_Y)
pred = clf.predict(test_X)
prediction = pd.DataFrame()
prediction['PassengerId'] = test_df['PassengerId'].copy()
prediction['Survived'] = pred
prediction.to_csv("submission.csv", index=False, sep=",", header=True)


# ## Iteration 2: Decision tree

# In[ ]:


from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.utils import shuffle

def fill_data(X):
    X['Age'] = X['Age'].fillna(value=29.7)
    X = X.fillna(method='ffill')
    return X

def dataset(dataframe, enc):
    try:
        Y = dataframe['Survived'].copy()
    except:
        Y = None
    
    X = dataframe[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']:
        X[col] = enc[col].transform(X[col].copy())
    
    return X,Y

train_df = fill_data(train_df)
test_df = fill_data(test_df)

full_df = train_df.append(test_df)

enc = dict()
for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']: 
    enc_col = preprocessing.LabelEncoder()
    enc_col.fit(full_df[col].copy())
    enc[col] = enc_col

train_df = shuffle(train_df).reset_index(drop=True)

train_X, train_Y = dataset(train_df[0:600], enc)
val_X  , val_Y   = dataset(train_df[600:], enc)
test_X, _ = dataset(test_df, enc)


# In[ ]:


normalizer = preprocessing.StandardScaler().fit(train_X.append(val_X).append(test_X) )
train_X = normalizer.transform(train_X)
val_X = normalizer.transform(val_X)
test_X = normalizer.transform(test_X)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)
val_pred = clf.predict(val_X)
accuracy_score(val_Y, val_pred)


# In[ ]:


pred = clf.predict(test_X)
prediction = pd.DataFrame()
prediction['PassengerId'] = test_df['PassengerId'].copy()
prediction['Survived'] = pred
prediction.to_csv("submission.csv", index=False, sep=",", header=True)

