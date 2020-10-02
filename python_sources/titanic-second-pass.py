#!/usr/bin/env python
# coding: utf-8

# This is my second go at the Titanic dataset. This time there are several features that I would like to try and test such as extracting a passngers title from the name column and parsing the ticket column into section and room number.

# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Read in Data

# In[ ]:


train = pd.read_csv("../input/train.csv", index_col='PassengerId')
test = pd.read_csv("../input/test.csv", index_col='PassengerId')
test.head()


# In[ ]:


train.tail()


# ## Set Up For Feature Engineering
# Here I'm going to try a trick I've seen in another notebook. Here we just concatenate the the training and the test data into the same dataframe in order to transform it. We keep track of the index where the test data starts so that we can split it back into two different pieces once we are done with the feature engineering.

# In[ ]:


train_results = train["Survived"].copy()
train.drop("Survived", axis=1, inplace=True, errors="ignore")
full_df = pd.concat([train, test])
traindex = train.index
testdex = test.index


# In[ ]:


full_df.drop("Ticket", axis=1, inplace=True, errors="ignore")
full_df["Fare"] = full_df["Fare"].fillna(full_df["Fare"].mean())
full_df["Age"] = full_df["Age"].fillna(full_df["Age"].mean())
full_df["Embarked"] = full_df["Embarked"].fillna(full_df["Embarked"].mode().iloc[0])
full_df["Cabin_Data"] = full_df["Cabin"].isnull().apply(lambda x: not x)


# In[ ]:





# In[ ]:


full_df["Deck"] = full_df["Cabin"].str.slice(0,1)
full_df["Room"] = full_df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
full_df[full_df["Cabin_Data"]]


# In[ ]:



full_df["Deck"] = full_df["Deck"].fillna("N")
full_df["Room"] = full_df["Room"].fillna(full_df["Room"].mean())


# In[ ]:


full_df.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")


# In[ ]:


def one_hot_column(df, label, drop_col=False):
    '''
    This function will one hot encode the chosen column.
    Args:
        df: Pandas dataframe
        label: Label of the column to encode
        drop_col: boolean to decide if the chosen column should be dropped
    Returns:
        pandas dataframe with the given encoding
    '''
    one_hot = pd.get_dummies(df[label], prefix=label)
    if drop_col:
        df = df.drop(label, axis=1)
    df = df.join(one_hot)
    return df


def one_hot(df, labels, drop_col=False):
    '''
    This function will one hot encode a list of columns.
    Args:
        df: Pandas dataframe
        labels: list of the columns to encode
        drop_col: boolean to decide if the chosen column should be dropped
    Returns:
        pandas dataframe with the given encoding
    '''
    for label in labels:
        df = one_hot_column(df, label, drop_col)
    return df


# In[ ]:


one_hot_df = one_hot(full_df, ["Embarked","Deck"])


# In[ ]:


one_hot_df.info()


# In[ ]:


one_hot_df.drop(["Embarked", "Deck"], axis=1, inplace=True, errors="ignore")


# In[ ]:


one_hot_df.head()


# In[ ]:


full_df["Title"] = full_df["Name"].str.extract("([A-Za-z]+\.)", expand=False)


# In[ ]:


full_df["Title"] = full_df["Title"].fillna("None")


# In[ ]:


full_df["Title"].value_counts()


# In[ ]:


one_hot_df["Title"] = full_df["Title"]
one_hot_df = one_hot_column(one_hot_df, "Title")


# In[ ]:


one_hot_df.drop("Name", axis=1, inplace=True, errors="ignore")


# In[ ]:


one_hot_df["Sex"] = one_hot_df["Sex"].map({"male": 0, "female":1}).astype(int)


# In[ ]:


one_hot_df.drop("Title_Dona.", axis=1, inplace=True, errors="ignore")
one_hot_df.drop("Title", axis=1, inplace=True, errors="ignore")


# In[ ]:


# Train
train_df = one_hot_df.loc[traindex, :]
train_df['Survived'] = train_results

# Test
test_df = one_hot_df.loc[testdex, :]


# In[ ]:


corr = train_df.corr()


# In[ ]:


corr["Survived"].sort_values(ascending=False)


# ## Start Training Models
# Ok so I think that I've dug as much info out of our given data as possible. now it's time to try training some possible algorithms on it.

# ### Imports Again

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics
from sklearn.model_selection import cross_val_score

import scipy.stats as st


# In[ ]:


rfc = RandomForestClassifier()


# In[ ]:


rfc.get_params().keys()


# In[ ]:


X = train_df.drop("Survived", axis=1).copy()
y = train_df["Survived"]


# In[ ]:


param_grid ={'max_depth': st.randint(6, 11),
             'n_estimators':st.randint(300, 500),
             'max_features':np.arange(0.5,.81, 0.05),
            'max_leaf_nodes':st.randint(6, 10)}

grid = RandomizedSearchCV(rfc,
                    param_grid, cv=10,
                    scoring='accuracy',
                    verbose=1,n_iter=80)

grid.fit(X, y)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.cv_results_


# In[ ]:


predictions = grid.best_estimator_.predict(test_df)


# In[ ]:


predictions


# In[ ]:


results_df =pd. DataFrame()
results_df["PassngerId"] = test_df.index
results_df["Predictions"] = predictions


# In[ ]:


results_df


# In[ ]:


results_df.to_csv("Predictions", index=False)


# In[ ]:




