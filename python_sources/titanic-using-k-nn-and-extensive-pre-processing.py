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


# # Importing All Relevant and Used Packages

# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# # Importing The Dataset
# Import the Dataset and get a basic understanding of the data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
train_df.head()


# # Missing Values
# Despite the fact that the k-NN algorithm can handle missing values, all missing values will be imputed for optimal performance.
# Imputation rules followed will be:
# * Features with more then 30% missing values removed
# * Categoric Features: Mode Imputed
# * Numeric Features with Outliers: Median
# * Numeric Features without Outliers: Mean

# In[ ]:


null_values = pd.isnull(train_df).sum()
null_values


# **Age Missing Values**
# As this is numeric, visualise the boxplot to see if outliers

# In[ ]:


train_df.boxplot(column=["Age"], grid=True)
plt.show()


# Boxplot indicates outliers, therefore impute median:

# In[ ]:


not_null_index = pd.notnull(train_df["Age"])
train_df = train_df[not_null_index]
age_median = train_df["Age"].median()
null_index = pd.isnull(train_df["Age"])
train_df.loc[null_index, "Age"] = age_median


# In[ ]:


null_values = pd.isnull(train_df).sum()
null_values


# Remove Cabin feature as has more then 30% of it's observations as missing values:

# In[ ]:


train_df = train_df.drop(columns="Cabin")


# Impute Mode for Embarked Feature

# In[ ]:


print(train_df["Embarked"].value_counts())  # S is the mode


# In[ ]:


null_index = pd.isnull(train_df["Embarked"])
train_df.loc[null_index, "Embarked"] = "S"


# In[ ]:


null_values = pd.isnull(train_df).sum()
null_values


# No missing values remain.

# # Irrelevant Features
# Drop all irrelevant Features

# In[ ]:


train_df = train_df.drop(columns="PassengerId")
train_df = train_df.drop(columns="Name")
train_df = train_df.drop(columns="Ticket")


# In[ ]:


train_df.head()


# # Encoding Categoric
# All nominal features encoded using One-Hot-Encoding and Ordinal Features encoded to preserve ordered relationship

# In[ ]:


train_encoded = pd.get_dummies(train_df, prefix= ["SibSp", "Parch", "Embarked"], columns= ["SibSp", "Parch", "Embarked"])
train_encoded["Sex"] = LabelEncoder().fit_transform(train_encoded.Sex)
train_encoded.head()


# # Splitting into Training and Testing

# In[ ]:


train_df = train_encoded
training_X = train_df.iloc[:, 1:]
training_y = train_df.iloc[:, 0]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(training_X, training_y, test_size=0.3, random_state=40)


# # Define a normalisation function to be used on numerical features

# In[ ]:


def normalize(x, col_max, col_min):
    if x == -1:
        return np.nan
    else:
        return ((x - col_min) / (col_max - col_min))


# In[ ]:


train_X = train_X.loc[:]
train_X["Age"] = train_X["Age"].apply(lambda x: normalize(x, train_X[('Age')].max(), train_X[('Age')].min()))
train_X["Fare"] = train_X["Fare"].apply(lambda x: normalize(x, train_X[('Fare')].max(), train_X[('Fare')].min()))

test_X = test_X.loc[:]
test_X["Age"] = test_X["Age"].apply(lambda x: normalize(x, test_X[('Age')].max(), test_X[('Age')].min()))
test_X["Fare"] = test_X["Fare"].apply(lambda x: normalize(x, test_X[('Fare')].max(), test_X[('Fare')].min()))


# In[ ]:


test_X.head()


# In[ ]:


train_X.head()


# # Fitting the Nearest Neighbor Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9, metric= "euclidean")
knn.fit(train_X, train_y)
knn_pred = knn.predict(test_X)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(test_y, knn_pred)
accuracy = accuracy_score(test_y, knn_pred)
cm


# In[ ]:


accuracy


# # Applying to Test Data File

# In[ ]:


testing_file = pd.read_csv("/kaggle/input/titanic/test.csv")
testing_df = testing_file
testing_df.head()


# In[ ]:


null_values = pd.isnull(testing_df).sum()
null_values


# In[ ]:


testing_df.boxplot(column=["Age"], grid=True)
plt.show()


# In[ ]:


not_null_index = pd.notnull(testing_df["Age"])
testing_nn = testing_df[not_null_index]
age_median = testing_df["Age"].median()
null_index = pd.isnull(testing_df["Age"])
testing_df.loc[null_index, "Age"] = age_median


# In[ ]:


null_values = pd.isnull(testing_df).sum()
null_values


# In[ ]:


testing_df = testing_df.drop(columns="Cabin")


# In[ ]:


testing_df.boxplot(column=["Fare"], grid=True)
plt.show()


# In[ ]:


not_null_index = pd.notnull(testing_df["Fare"])
testing_nn = testing_df[not_null_index]
fare_median = testing_nn["Fare"].median()
null_index = pd.isnull(testing_df["Fare"])
testing_df.loc[null_index, "Fare"] = fare_median


# In[ ]:


null_values = pd.isnull(testing_df).sum()
null_values


# In[ ]:


testing_df = testing_df.drop(columns="PassengerId")
testing_df = testing_df.drop(columns="Name")
testing_df = testing_df.drop(columns="Ticket")


# In[ ]:


testing_df.head()


# In[ ]:


testing_encoded = pd.get_dummies(testing_df, prefix= ["SibSp", "Parch", "Embarked"], columns= ["SibSp", "Parch", "Embarked"])
testing_encoded["Sex"] = LabelEncoder().fit_transform(testing_encoded.Sex)
testing_encoded.head()


# In[ ]:


testing_df = testing_encoded
testing_df = testing_df.loc[:]
testing_df["Age"] = testing_df["Age"].apply(lambda x: normalize(x, testing_df[('Age')].max(), testing_df[('Age')].min()))
testing_df["Fare"] = testing_df["Fare"].apply(lambda x: normalize(x, testing_df[('Fare')].max(), testing_df[('Fare')].min()))


# In[ ]:


testing_df.head()


# # Using the same model as previously trained

# In[ ]:


new_list = [train_X.columns, testing_df.columns]
new_list


# Evident that whilst "SibSp_8" appears in the testing set, it does not appear in the training set.
# Theoretically by dropping this column, the algorithm should still have the ability to make an accurate prediction.

# In[ ]:


testing_df = testing_df.drop(columns='SibSp_8')
testing_df = testing_df.drop(columns='Parch_9')


# In[ ]:


knn_predictions = knn.predict(testing_df)


# In[ ]:


testing_file["Survived"] = pd.Series(knn_predictions, index = testing_file.index)
testing_file


# In[ ]:


final_df = testing_file.filter(['PassengerId', 'Survived'], axis = 1)
final_df.set_index('PassengerId')


# In[ ]:


final_df.to_csv("survival_predictions2.csv", encoding='utf-8', index = False, header = True)

