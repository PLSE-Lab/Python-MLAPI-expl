#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning, Feature Engineering & Model Stacking

# Wanting to play around with a small dataset that focused on classification, I decided to tackle the Titanic dataset. This notebook focuses on data cleaning--particularly addressing missing values, feature engineering (to a lesser degree), and model stacking.

# First, of course, import the necessary packages. I also like to preempitively change my display so that columns are never truncated (such an issue doesn't really apply to a dataset this small; but I think it's a good habit).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

import gc; gc.enable()

pd.set_option('display.max_columns', 100)


# Next, I read in the train and test data sets, and briefly examine the train.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
final_test = pd.read_csv('../input/test.csv')

print(len(df_train))
print(df_train.shape)
print(df_train.columns)
df_train.head()


# I do the same with the test data.

# In[ ]:


print(len(df_test))
print(df_test.shape)
print(df_test.columns)
df_test.head()


# Stating the obvious, only the train data includes the target variable, 'Survived'.

# ### Data Cleaning

# Checking each data set for NaNs or missing values, I find the following:

# In[ ]:


print('Missing Values in Train Data')
print(df_train.isnull().sum())
print('')
print('Missing Values in Test Data')
print(df_test.isnull().sum())


# Tackling the 'Age' column first, I borrowed some code from a far more experienced Kaggler [https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic] and addressed the matter in both the train and test sets. The strategy of filling in these missing 'Age' values entails taking the median value of other passengers based on their 'Sex' and 'Pclass'.

# In[ ]:


# Filling the missing values in Age with the medians of Sex and Pclass groups
df_train['Age'] = df_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_test['Age'] = df_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# Regarding the two passengers with missing data in the 'Embarked' column of the train data set, others have researched the matter and concluded that these passengers left from Southhampton; hence, their values should be 'S'.

# In[ ]:


# Filling the missing values in Embarked with S
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_test['Embarked'] = df_test['Embarked'].fillna('S')


# There is a single value in the 'Fare' column of the test data that needs to be addressed. Once again, this idea and the code are indebted to the experienced Kaggler mentioned above.

# In[ ]:


test_med_fare = df_test.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

# Filling the missing value in Fare with the median Fare of a 3rd class alone passenger
df_test['Fare'] = df_test['Fare'].fillna(test_med_fare)


# With the 'Age', 'Embarked' & 'Fare' missing values dealt with, the missing 'Cabin' values need to be addressed. Following the lead of the aforementioned Kaggler, 'Cabin' is simplified and translated into a 'Deck' column.

# In[ ]:


# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
df_train['Deck'] = df_train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df_test['Deck'] = df_test['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')


# In[ ]:


print('Deck Column for Train Set')
print(df_train['Deck'].value_counts())
print('')
print('Deck Column for Test Set')
print(df_test['Deck'].value_counts())


# Due to the proximity of the T deck to the A deck, the single passenger in the T deck can be changed to A.

# In[ ]:


# Passenger in the T deck is changed to A
train_idx = df_train[df_train['Deck'] == 'T'].index
df_train.loc[train_idx, 'Deck'] = 'A'


# The general measure of proximity also justifies the combining of certain decks in order to simplify the data.

# In[ ]:


df_train['Deck'] = df_train['Deck'].replace(['A', 'B', 'C'], 'AB')
df_train['Deck'] = df_train['Deck'].replace(['D', 'E'], 'DE')
df_train['Deck'] = df_train['Deck'].replace(['F', 'G'], 'FG')

df_test['Deck'] = df_test['Deck'].replace(['A', 'B', 'C'], 'AB')
df_test['Deck'] = df_test['Deck'].replace(['D', 'E'], 'DE')
df_test['Deck'] = df_test['Deck'].replace(['F', 'G'], 'FG')


# With the 'Deck' column complete and created from the now obsolete 'Cabin' column, the latter can be removed from the test and train data sets.

# In[ ]:


df_train.drop(columns = ['Cabin'], inplace=True, axis=1)
df_test.drop(columns = ['Cabin'], inplace=True, axis=1)


# All missing data should have been dealt with; it's worth double-checking to make sure.

# In[ ]:


print('Missing values in Train data')
print(df_train.isnull().sum())
print('')
print('Missing values in Test data')
df_train.isnull().sum()


# Aiming to transform the data set into a machine readable format, it's worth changing the 'Sex' column to 0s & 1s. 

# In[ ]:


df_train['Sex'] = df_train.Sex.map({'female': 0, 'male': 1})
df_test['Sex'] = df_test.Sex.map({'female': 0, 'male': 1})


# ### Feature Engineering

# The 'Advanced Feature Engineering Tutorial with Titanic' suggests that a decent amount of information gain is attained through the binning of both the 'Fare' column and the 'Age' column. The number of bins for 'Fare' and 'Age' were most likely arrived at through trial and error.

# In[ ]:


fare_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

df_train['Fare'] = pd.qcut(df_train['Fare'], 13, labels=fare_labels)
df_test['Fare'] = pd.qcut(df_test['Fare'], 13, labels=fare_labels)

df_train['Age'] = pd.qcut(df_train['Age'], 10, labels=age_labels)
df_test['Age'] = pd.qcut(df_test['Age'], 10, labels=age_labels)


# The tutorial also suggests creating a 'Family Size' column by adding 1 (for the current passenger) to the 'SibSp' & 'Parch' columns. This makes good intuitive sense.

# In[ ]:


df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['Family_Size'] = df_test['SibSp'] + df_test['Parch'] + 1


# Another suggestion from the tutorial that makes sense is the creation of a 'Ticket Frequency' column (what I've decided to call 'Group Size') from the 'Ticket' column. This feature will help capture the approximate size of the group each respective passenger was traveling with. With a measure of helpful information extracted from the 'Ticket' column; it can now be dropped.

# In[ ]:


df_train['Group_Size'] = df_train.groupby('Ticket')['Ticket'].transform('count')
df_test['Group_Size'] = df_test.groupby('Ticket')['Ticket'].transform('count')

df_train.drop(columns = ['Ticket'], inplace=True)
df_test.drop(columns = ['Ticket'], inplace=True)


# A final feature engineering suggestion from the advanced tutorial was to squeeze out useful information embedded in the 'Name' column using the .split() method. First, the presence of the title 'Mrs.' (or lack thereof) can be used to create a 'Married' column. Though imperfect, especially given the lack of an analogous title for men that indicates their marital status, at least some information is captured with this effort. Second, the isolated remaining titles can be grouped together in regards to estimations of social position.

# In[ ]:


df_train['Title'] = df_train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]


# In[ ]:


# Apply the .split() method to isolate the words in each 'Name' entry: first on the comma, then on the period
df_train['Title'] = df_train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_train['Married'] = 0
df_train['Married'].loc[df_train['Title'] == 'Mrs'] = 1

df_test['Title'] = df_test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_test['Married'] = 0
df_test['Married'].loc[df_test['Title'] == 'Mrs'] = 1


# In[ ]:


df_train['Title'] = df_train['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_train['Title'] = df_train['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

df_test['Title'] = df_test['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_test['Title'] = df_test['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# In[ ]:


print(df_train['Title'].value_counts())
df_test['Title'].value_counts()


# In[ ]:


df_train.drop(columns = ['Name'], inplace=True)
df_test.drop(columns = ['Name'], inplace=True)


# Preemptively addressing a mismatch in columns between the train and test set, I used .value_counts() to discover discrepancies in both the 'Parch' values and the 'Group_Size' values. I replace the unique categories with the highest alternative in each column.

# In[ ]:


df_test['Parch'] = df_test['Parch'].replace(9, 6)

df_train['Group_Size'] = df_train['Group_Size'].replace([6, 7], 5)


# The train and test sets are now ready to 'dummify'; dummy variables will help create a lot of machine readable columns. Some of the columns that I turn into dummy variables are ordinal rather than categorical; that is, they don't have to be turned into dummy variables, but given that I plan on using a few tree-based models, I've decided to dummy-up as many of my features as possible.

# In[ ]:


df_train = pd.get_dummies(data=df_train, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Deck', 'Family_Size', 'Group_Size', 'Title'])
df_test = pd.get_dummies(data=df_test, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Deck', 'Family_Size', 'Group_Size', 'Title'])


# In[ ]:


df_train.head()


# ### Model Stacking

# Readying the data for modeling, I separate the target variable from the predictors and go ahead and drop 'PassengerId' from both the train and test sets.

# In[ ]:


y = df_train['Survived']

df_train = df_train.drop(columns = ['Survived', 'PassengerId'], inplace=False)
df_test = df_test.drop(columns = ['PassengerId'], inplace=False)


# Importing the necessary libraries for modeling...

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier


# I split the train set data into train and test to get a sense of how my models are performing.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.30, random_state=42)


# Using the StackingCV Classifier, I use K-Nearest-Neighbors (KNN) (since we're dealing with a small dataset), Logistic Regression, Random Forest, and ExtraTrees for 'lower-level' models and XGB Classifier for my 'meta-level' model.

# In[ ]:


import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import xgboost as xgb
import lightgbm as lgb

clf1 = KNeighborsClassifier()
clf2 = LogisticRegressionCV(cv=7)
clf3 = RandomForestClassifier(max_depth=3, n_estimators=150, random_state=42)

xgb = xgb.XGBClassifier(silent=False,
                        scale_pos_weight=1,
                        learning_rate=0.02,  
                        colsample_bytree = 0.4,
                        subsample = 0.8,
                        objective='binary:logistic', 
                        n_estimators=1000, 
                        reg_alpha = 0.3,
                        max_depth=3, 
                        gamma=1)

# [Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.]
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            use_probas=False,
                            meta_classifier=xgb,
                            random_state=42)

print('7-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'LogisticRegression',
                       'Random Forest',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_train, y_train, 
                                              cv=7, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# In[ ]:


print('7-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN',
                       'LogisticRegression', 
                       'Random Forest',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_test, y_test, 
                                              cv=7, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# In[ ]:


sclf.fit(X_train, y_train)
training_preds = sclf.predict(X_train)
val_preds = sclf.predict(X_test)
training_accuracy = accuracy_score(y_train, training_preds)
val_accuracy = accuracy_score(y_test, val_preds)

print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
print("Validation accuracy: {:.4}%".format(val_accuracy * 100))


# In[ ]:


# Predicting on the titanic test set with my stacked model:
y_pred = sclf.predict_proba(df_test)


# In[ ]:


# Grabbing the predictive probabilities that I'll need for my submission and converting them to integers:
y_pred = y_pred[ : , 1]

y_pred = pd.Series(y_pred)
y_pred = (y_pred.round()).astype(int)


# In[ ]:


submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = final_test['PassengerId']
submission_df['Survived'] = y_pred

submission_df.to_csv('submissions.csv', header=True, index=False)


# Thanks so much for perusing my notebook! I hope it was helpful.
