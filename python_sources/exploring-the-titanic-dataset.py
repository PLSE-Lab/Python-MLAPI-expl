#!/usr/bin/env python
# coding: utf-8

# # Introduction
# For this dataset, we try to explore the possibility of one surviving the Titanic. This activity is practically just to get our hands dirty and get an idea on how Kaggle works, and how to join competitions.

# In[ ]:


# Importing basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the Titanic dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_file_path = '/kaggle/input/titanic/train.csv'
train_data = pd.read_csv(train_file_path)
test_file_path = '/kaggle/input/titanic/test.csv'
test_data = pd.read_csv(test_file_path)


# # Dataset Analysis
# To get an idea on what preprocessing techniques can be done, it is important to see first the information that the dataset contains. 
# Some of the things that one usually looks for here are (but not limited to): 
# * Possibilities for Feature Engineering
# * Features are numerical and categorical
# * Unique values in categorical columns
# * Columns that have missing values
# * Correlation of each feature with the survival prediction

# In[ ]:


train_data.head()


# From the first few rows of the dataset, it could be possible to create a separate column on the number of family members that the passenger was on board with from the SibSp and Parch columns, as family members tend to stick together (whether surviving or not).
# 
# Depending on the values of the Ticket Number and Title of the Name, there could also be a possibility to use these as another categorical column.

# In[ ]:


train_data.dtypes


# Categorical Columns are the Name, Sex, Ticket Number, Cabin Type, and Place of Embarkment.
# 
# Numerical Columns are the Class, Age, 

# In[ ]:


train_data[train_data.columns[1:]].corr()['Survived'][:]


# In[ ]:


train_data.info()
print("-"*100)
test_data.info()


# With 891 entries, the Age, Cabin Type, and Place of Embarkment have missing values. 
# 
# Since the place of embarkment has only two missing values, these rows can be dropped. However, the Age column can be filled in by Feature Engineering. 
# 
# Moreover, the Titanic was said to hold passengers in rooms that had three different classes of cabins. Otherwise, the passenger was placed in a First Class Suite. The empty values in the Cabin column could probably infer that the passenger was placed in a Suite.

# In[ ]:


print(train_data['Age'].describe())
print("-"*100)
print(train_data['Fare'].describe())


# # Data Visualization
# Section is currently in progress

# In[ ]:


import seaborn as sns


# In[ ]:


#Percentage of factor of each feature on the chances of survival


# A general sense of the data is acquired. From the information above, the following can be inferred.
# * The categorical values of Sex, the type of Room (Cabin or Suite), and place of Embarkment can be encoded using the One Hot Encoder. 
# * Due to the huge difference in range, numerical values such as the passenger's age and ticket fare can be normalized converted to categorical values. 
# * Other numerical values such as the passenger's class and the number of family members can be left as is since it can act as Label Encoded values.
# * Features such as the Name and the Ticket Number can be dropped from the data set as these have unique values.

# # Data Preprocessing
# #Age: Simple Imputer
# #Sex, Embarked, Room Type: One Hot Encode
# #Remove two rows where embarked is empty
# #Cabin, or suite: If null, suite, else cabin

# In[ ]:


datasets = [train_data, test_data]


# In[ ]:


train_data.dropna(subset = ["Embarked"], inplace=True)


# In[ ]:


for dataset in datasets:
    dataset['Family Members'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[ ]:


for dataset in datasets:
    dataset['Boarded Alone'] = 0
    dataset.loc[dataset['Family Members'] == 1, 'Boarded Alone'] = 1


# In[ ]:


for dataset in datasets:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
    dataset['Categorical Fare'] = pd.qcut(dataset['Fare'], 4)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Categorical Age'] = pd.qcut(dataset['Age'], 4)


# In[ ]:


def edit_cabin(dataset):
    dataset["Cabin"] = dataset["Cabin"].fillna("Suite")
    dataset.loc[~dataset["Cabin"].isin(["Suite"]), "Cabin"] = "Regular"
    return dataset

train_data = edit_cabin(train_data)
test_data = edit_cabin(test_data)


# In[ ]:


train_data.info()


# In[ ]:


import re as re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)
    return ""

for dataset in datasets:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[ ]:


train_data.info()


# In[ ]:


y_train = train_data['Survived']
X_train = train_data.drop(['Survived', 'Name', 'Ticket', 'PassengerId', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1)
X_valid = test_data.drop(['Name', 'Ticket', 'PassengerId', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1)


# In[ ]:


X_train.head()


# In[ ]:


categorical_cols = ["Sex", "Embarked", "Cabin", "Categorical Fare", "Categorical Age", "Title"]


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ])


# # Prediction with Ensemble Learning and Stacking

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0)

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

rf_pipeline.fit(X_train, y_train)


# # Competition Submission

# In[ ]:


submission = rf_pipeline.predict(X_valid)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': submission})
output.to_csv('lester_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




