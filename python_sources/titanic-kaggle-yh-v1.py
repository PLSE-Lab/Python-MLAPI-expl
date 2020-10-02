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


# In[ ]:


data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()


# In[ ]:


len(data)


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# In[ ]:


data.isna().sum()


# In[ ]:


data.info()


# In[ ]:


data["Survived"].value_counts()


# In[ ]:


data.corr()


# In[ ]:


# Split into X and y
X = data.drop("Survived", axis=1)
y = data["Survived"]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


# Split data into train and validation sets
from sklearn.model_selection import train_test_split
np.random.seed(17)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[ ]:


len(X_train), len(X_val), len(y_train), len(y_val)


# ### Prepare our data
# 
# We need to:
# 1. handle missing values
# 2. make all data numerical
# 
# Only these features have missing:
# 1. age
# 2. cabin - to be removed
# 3. embarked
# 4. fare (in test data)
# 
# Our pipeline function will:
# 1. remove "PassengerId", "Name", "Ticket", "Cabin"
# 2. change "Sex" into numerical
# 3. change "Embarked" into numerical
# 4. fill embarked missing with mode?
# 5. fill age with mean?
# 6. fill fare with mean

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def transform_data(data):
    # remove "PassengerId", "Name", "Ticket", "Cabin"
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Name", axis=1)
    data = data.drop("Ticket", axis=1)
    data = data.drop("Cabin", axis=1)
    
    # fill na with pandas
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    data["Fare"].fillna(data["Fare"].mean(), inplace=True)
    
    # change "Sex" and "Embark" into numerical
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                      one_hot,
                                      ["Sex", "Embarked"])],
                                    remainder="passthrough")
    data = transformer.fit_transform(data)
    
    return data


# In[ ]:


X_train_tf = transform_data(X_train)
X_train_tf
# note this converts to numpy array, and not pd


# In[ ]:


X_val_tf = transform_data(X_val)
X_val_tf


# In[ ]:


pd.DataFrame(X_val_tf).isna().sum()


# ### Modeling
# We're going to try 3 different machine learning models:
# 
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train_tf, X_val_tf, y_train, y_val):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    """
    # Set random seed
    np.random.seed(17)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train_tf, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_val_tf, y_val)
    return model_scores


# In[ ]:


model_scores = fit_and_score(models=models,
                             X_train_tf=X_train_tf,
                             X_val_tf=X_val_tf,
                             y_train=y_train,
                             y_val=y_val)

model_scores


# ### Further exploring model
# on RandomForest, since it currently gives us the best result

# In[ ]:


# creating a evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_preds(y_true, y_preds):
    """
    perform evaluation comparison on y_true labels vs y_preds labels
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                          "precision": round(precision, 2),
                          "recall": round(recall, 2),
                          "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    return metric_dict


# In[ ]:


np.random.seed(17)
clf = RandomForestClassifier()
clf.fit(X_train_tf, y_train)
y_preds = clf.predict(X_val_tf)  # prediction using X_val_tf

# evaluate using our function on validation set
baseline_metrics = evaluate_preds(y_val, y_preds)
baseline_metrics


# In[ ]:


# tuning hyperparameters by RandomSearchCV

from sklearn.model_selection import RandomizedSearchCV
grid = {"n_estimators": [10,100,200,500,1000,1200],
        "max_depth": [None,5,10,20,30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2,4,6],
        "min_samples_leaf": [1,2,4]}

np.random.seed(17)

clf = RandomForestClassifier()
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,  # what we defined above
                            n_iter=10, # number of combinations to try
                            cv=5,   # number of cross-validation split
                            verbose=2)
rs_clf.fit(X_train_tf, y_train);


# In[ ]:


rs_clf.best_params_


# In[ ]:


rs_y_preds = rs_clf.predict(X_val_tf)

# evaluate predictions
rs_metrics = evaluate_preds(y_val, rs_y_preds)


# In[ ]:


# tuning hyperparameters by GridSearchCV
from sklearn.model_selection import GridSearchCV

grid_2 = {'n_estimators': [500, 1000, 2000],
         'max_depth': [None, 10],
         'max_features': ['sqrt'],
         'min_samples_split': [4, 6],
         'min_samples_leaf': [1, 2]}

np.random.seed(17)

clf = RandomForestClassifier()

# Setup GridSearchCV
gs_clf = GridSearchCV(estimator=clf,
                            param_grid=grid_2,
                            cv=5,
                            verbose=2)

# Fit the GSCV version of clf
gs_clf.fit(X_train_tf, y_train);


# In[ ]:


gs_clf.best_params_


# In[ ]:


gs_y_preds = gs_clf.predict(X_val_tf)
gs_y_preds


# In[ ]:


gs_metrics = evaluate_preds(y_val, gs_y_preds)


# In[ ]:


# Lets compare our different model metrics

compare_metrics = pd.DataFrame({"baseline": baseline_metrics,
                                "random search": rs_metrics,
                                "grid search": gs_metrics})
compare_metrics.plot.bar(figsize=(10,8))


# Seems like RandomSearchCV yields the best result for our model

# Testing
# 1. transform test data
# 2. run test
# 3. export into desired test result

# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


test_data_tf = transform_data(test_data)
test_data_tf 


# In[ ]:


pd.DataFrame(test_data_tf ).isna().sum()


# In[ ]:


test_data.isna().sum()


# In[ ]:


test_preds = gs_clf.predict(test_data_tf)
pd.DataFrame(test_preds)


# In[ ]:


submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], 
                          "Survived": test_preds})


# In[ ]:


submission


# In[ ]:


len(submission)


# In[ ]:


len(test_data)


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




