#!/usr/bin/env python
# coding: utf-8

# <h1 align=center><font size = 4>Tackle the Titanic Dataset</font></h1>
# <h1 align=center><font size = 5>Model Binary Classifier</font></h1>

# # Table of Contents
# * [Setup](#setup)
# * [Get the Data](#get_data)
# * [Take a Quick Look at the Data Structure](#data_structure)
# * [Prepare Data for Machine Learning](#preparation)
# * [Train and Select Models](#model_selection)
# * [Fine-tune the best Models](#model_tuning)
# * [Make Predictions](#predictions)

# <a id="setup"></a>
# # Setup

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="get_data"></a>
# # Get the data

# In[ ]:


def load_titanic_data(filename, titanic_path):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


# In[ ]:


train_data = load_titanic_data('train.csv',"../input")
test_data = load_titanic_data('test.csv','../input')


# In[ ]:


test_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# <a id="data_structure"></a>
# # Take a Quick Look at the Data Structure

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
train_data[['Age','SibSp','Parch','Fare']].hist(figsize=(8,8))


# Check that the output is indeed 0 or 1

# In[ ]:


train_data['Survived'].value_counts()


# In[ ]:


train_data['Pclass'].value_counts()


# In[ ]:


train_data['Sex'].value_counts()


# In[ ]:


train_data['Embarked'].value_counts()


# <a id="preparation"></a>
# # Prepare Data for Machine Learning

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('select_numeric', DataFrameSelector(['Age','SibSp','Parch','Fare'])),
    ('imputer', SimpleImputer(strategy='median'))
])


# In[ ]:


num_pipeline.fit_transform(train_data)


# In[ ]:


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


cat_pipeline = Pipeline([
    ('select_cat', DataFrameSelector(['Pclass','Sex','Embarked'])),
    ('imputer', MostFrequentImputer()),
    ('cat_encoder', OneHotEncoder(sparse=False))
])


# In[ ]:


cat_pipeline.fit_transform(train_data)


# In[ ]:


from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


# In[ ]:


X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train.shape)
X_train


# In[ ]:


y_train = train_data['Survived']


# <a id="model_selection"></a>
# # Train and Select Models

# In[ ]:


from sklearn.svm import SVC

svm_clf = SVC(gamma='auto')
svm_clf.fit(X_train, y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()


# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=100, random_state=42, tol=0.001)
sgd_scores = cross_val_score(sgd_clf, X_train, y_train, cv=10)
sgd_scores.mean()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=10)
knn_scores.mean()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.plot([3]*10, sgd_scores, ".")
plt.plot([4]*10, knn_scores, ".")
plt.boxplot([svm_scores, forest_scores, sgd_scores, knn_scores], labels=("SVM","Random Forest","SGD",'KNN'))
plt.ylabel("Accuracy", fontsize=14)
plt.show()


# In[ ]:


train_data['AgeBucket'] = train_data['Age'] // 15 * 15
train_data[['AgeBucket','Survived']].groupby(['AgeBucket']).mean()


# In[ ]:


train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
pd.pivot_table(train_data, index="RelativesOnboard", values="Survived", aggfunc="mean")


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

age_ix, sibsp_ix, parch_ix = 1, 2, 3

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes = True):
        self.add_attributes = add_attributes
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        if self.add_attributes:
            age_bucket = X[:, age_ix] // 15 * 15
            relatives_onboard = X[:, sibsp_ix] + X[:, parch_ix]
            return np.c_[X, age_bucket, relatives_onboard]
        else:
            return np.c_[X]


# In[ ]:


from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


# In[ ]:


num_pipeline.fit_transform(train_data)


# In[ ]:


from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[ ]:


X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train.shape)
X_train


# <a id="model_tuning"></a>
# # Fine-tune the best Models

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [50, 100, 150],
     'max_depth':[5, 10]}
]

forest_clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(forest_clf, param_grid, cv=10, return_train_score=True)

grid_search.fit(X_train, y_train)


# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[ ]:


grid_search.best_params_


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=20, high=150),
    'max_features': randint(low=5, high=12),
    'max_depth': randint(low=5, high=12)
}

forest_clf = RandomForestClassifier(random_state=42)
forest_rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs, n_iter=50, cv=10, return_train_score=True,
                                      random_state=42, verbose=2, n_jobs=-1)
forest_rnd_search.fit(X_train, y_train)


# In[ ]:


forest_rnd_search.best_params_


# In[ ]:


cvres = forest_rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# <a id="predictions"></a>
# # Make Predictions

# In[ ]:


final_model = forest_rnd_search.best_estimator_


# In[ ]:


test_data.head()


# In[ ]:


X_test = preprocess_pipeline.transform(test_data)


# In[ ]:


y_pred = final_model.predict(X_test)
y_pred


# In[ ]:


test_data["Survived"] = y_pred
test_data[["PassengerId","Survived"]].to_csv('submission.csv', index=False)

