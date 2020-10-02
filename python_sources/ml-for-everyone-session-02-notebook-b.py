#!/usr/bin/env python
# coding: utf-8

# # Session 02: Train/Test, Cross-Validation, and Regularization

# In this notebook, we'll see how to create train/test splits in python.  We'll also see the `GridSearchCV` function, which is a very simple way to do cross-validation to select tuneable parameters.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV


# Let's print what versions of the libraries we're using
print(f"python\t\tv {sys.version.split(' ')[0]}\n===")
for lib_ in [np, pd, sns, sklearn, ]:
    sep_ = '\t' if len(lib_.__name__) > 8 else '\t\t'
    print(f"{lib_.__name__}{sep_}v {lib_.__version__}"); del sep_, lib_


# Let's read in the "Hitters" dataset from ISLR that has information on baseball players, their stats, and their salaries.  Also, we'll drop any rows with missing values.

# In[ ]:


import os
os.getcwd()


# In[ ]:


get_ipython().system('ls')


# In[ ]:


hitters = pd.read_csv("../input/hitters/Hitters.csv")
hitters = hitters.dropna(inplace=False)
hitters.head()


# We'll get rid of a few categorical columns rather than deal with converting them.  Then we'll create a binary variable for whether a player makes more than the median salary.

# In[ ]:


X = np.array(hitters.drop(["Salary", "League", "Division", "NewLeague"], axis=1))
y = (hitters["Salary"] >= np.median(hitters["Salary"])).astype("int")


# Creating a training/testing split is extremely simple:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=10)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# Next, we'll fit a logistic regression model to the training data and score the test data:

# In[ ]:


logit = LogisticRegression(penalty="l2", C=1e5, n_jobs=-1, max_iter=4000)
logit.fit(X_train, y_train)

test_preds = logit.predict_proba(X_test)[:, 1]


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, test_preds)


# In[ ]:


# we want to draw the random baseline ROC line too
fpr_rand = tpr_rand = np.linspace(0, 1, 10)

plt.plot(fpr, tpr)
plt.plot(fpr_rand, tpr_rand, linestyle='--')
plt.show()


# In[ ]:


roc_auc_score(y_test, test_preds)


# If we re-run the train/test split, we'll see the variability in this estimate.

# We can use the test set (which, in this case, should really be called a validation set) to choose the best value of the tuneable parameter `C` of the logisitc regression, which is the inverse of $\lambda$, the regularization strength.

# In[ ]:


# create equally space values beteen 10^-10 and 10^10
c_vals = np.logspace(-10, 10, 20)

aucs = []
for c_val in c_vals:
    logit = LogisticRegression(C=c_val)
    logit.fit(X_train, y_train)

    test_preds = logit.predict_proba(X_test)[:, 1]
    aucs.append(roc_auc_score(y_test, test_preds))


# In[ ]:


aucs


# In[ ]:


plt.plot(np.log10(c_vals), aucs)
plt.xlabel("C")
plt.ylabel("Test AUC")
plt.show()


# Instead of using a train/test split, scikit-learn has a really nice way to use cross-validation to choose the tuneable parameters of a model.  First, we make a dictionary, where the key is the name of the parameter we want to tune (it has to match the name of the parameter in the model), and the values are the values we want to try:

# In[ ]:


param_grid = {"C": np.logspace(2, 8, 50)}


# Then, we pass in the model we want to fit and the grid.  The option 'n_jobs' allows us to split the cross-validation over multiple cores of your computer, and `refit` tells it to fit the best performing model on the full dataset once it's done.

# In[ ]:


cv = GridSearchCV(logit, param_grid, cv=10, n_jobs=-1, refit=True, verbose=True)
cv.fit(X_train, y_train)


# We can see the best values and the grid scores:

# In[ ]:


cv.best_estimator_


# In[ ]:


cv.best_params_


# Let's see what value of $\lambda$ corresponds to the best C:

# In[ ]:


np.log10(1.0/cv.best_params_['C'])


# In[ ]:


cv.best_score_


# In[ ]:


cv.cv_results_


# In[ ]:


test_preds = cv.best_estimator_.predict_proba(X_test)[:, 1]
test_preds


# In[ ]:


roc_auc_score(y_test, test_preds)


# In[ ]:




