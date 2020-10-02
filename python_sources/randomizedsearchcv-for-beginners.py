#!/usr/bin/env python
# coding: utf-8

# ### The goal of this kernel is to showcase the functionality of RandomizedSearchCV
# 
# RandomizedSearchCV is preferred over GridSearchCV for its speed, particularly when tuning many hyperparameters.
# 
# Comments are welcome :) Please share your expertise - I would love to learn better ways of tuning models.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint
import matplotlib.pyplot as plt
import time
import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


# ### Read data and split out labels

# In[ ]:


train_df = pd.read_csv("../input/learn-together/train.csv")
test_df = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


X_train = train_df.copy()
X_test = test_df.copy()

y_train = X_train["Cover_Type"]
X_train.drop(columns=["Id", "Cover_Type"], inplace=True, axis=1)
test_ids = X_test["Id"]
X_test.drop(columns=["Id"], inplace=True, axis=1)


# ### Set the starting state for random number generation to ensure reproducibility

# In[ ]:


SEED = 42


# ### Create a baseline random forest model

# In[ ]:


rf = RandomForestClassifier(n_jobs=-1,
                            random_state=SEED)
rf.fit(X_train, y_train)


# In[ ]:


scores = cross_val_score(rf, X_train, y_train, cv=10, scoring="accuracy")
scores.mean(), scores.std()


# ### Observe the impact of scaling features in our base model

# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)


# In[ ]:


rf_scaled = RandomForestClassifier(n_jobs=-1,
                            random_state=SEED)
rf_scaled.fit(X_train, y_train)

scores = cross_val_score(rf_scaled, X_train, y_train, cv=10, scoring="accuracy")
scores.mean(), scores.std()


# ### Create a reusable function to tune hyperparameters

# In[ ]:


def hyperparameter_tune(base_model, parameters, n_iter, kfold, X=X_train, y=y_train):
    start_time = time.time()
    
    # Arrange data into folds with approx equal proportion of classes within each fold
    k = StratifiedKFold(n_splits=kfold, shuffle=False)
    
    optimal_model = RandomizedSearchCV(base_model,
                            param_distributions=parameters,
                            n_iter=n_iter,
                            cv=k,
                            n_jobs=-1,
                            random_state=SEED)
    
    optimal_model.fit(X, y)
    
    stop_time = time.time()

    scores = cross_val_score(optimal_model, X, y, cv=k, scoring="accuracy")
    
    print("Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print("====================")
    print("Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(scores.mean(), scores.std()))
    print("Best Score: {:.3f}".format(optimal_model.best_score_))
    print("Best Parameters: {}".format(optimal_model.best_params_))
    
    return optimal_model.best_params_, optimal_model.best_score_


# ### Tune parameters (a smaller set of parameters for demo purposes)

# In[ ]:


base_model = RandomForestClassifier(n_jobs=-1,
                                   random_state=SEED)

lots_of_parameters = {
    "max_depth": [3, 5, 10, None],
    "n_estimators": [100, 200, 300, 400, 500],
    "max_features": randint(1, 3),
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
    "min_samples_leaf": randint(1, 4)
}

parameters = {
    "max_depth": [3, 5, 10, None],
    "n_estimators": [100, 200, 300, 400, 500]
}

best_params, best_score = hyperparameter_tune(base_model, parameters, 10, 5, X_train, y_train)


# ### As an interesting aside, observe how changing the number of folds impacts the optimal paramaters...

# In[ ]:


scores = []
folds = range(2, 8)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score = hyperparameter_tune(base_model, parameters, 10, i, X_train, y_train)
    scores.append(best_score)


# In[ ]:


plt.plot([x for x in folds], scores)
plt.xlabel("# of Folds")
plt.ylabel("Best Score")
plt.title("The Impact of # of Folds on Randomized Search CV Score")
plt.show()

