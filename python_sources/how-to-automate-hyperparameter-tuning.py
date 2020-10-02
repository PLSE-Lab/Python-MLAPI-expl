#!/usr/bin/env python
# coding: utf-8

# # Auto Model Selection and Hyperparameter Tuning - A Comprehensive Guide on Hyperopt
# 
# This is to demonstrate how hyperopt-sklearn works using Iris dataset.
# 
# The small size of Iris means that hyperparameter optimization takes just a few seconds. On the other hand, Iris is so easy that we'll typically see numerous near-perfect models within the first few random guesses; hyperparameter optimization algorithms are hardly necessary at all.
# 
# Nevertheless, here is how to use hyperopt-sklearn (hpsklearn) to find a good model of the Iris data set. 
# 
# We will talk about the following staff:
# 
# 1. [ A motivation to show what is under the hood for hyperopt](#Motivation)
# 2. [Find the best number of neighbors of a KNN model]( #KNN Model)
# 3. [Choose the best model and parameter among many ](# Complicated SKLean model)
# 4. [Use the result to train a best model](#Train)

# ## Motivation
# The hyperopt is like solver in excel, it finds parameters that minimize your function. 
# 
# In the following example, we have a function $y = x$. Hyperopt will search in the $(0,1)$ space to find the x that minimize y. TPE is the algorithm that suggests the next x to try.

# In[3]:


from hyperopt import fmin, tpe, hp
best = fmin(
    fn = lambda x: x,
    space = hp.uniform('x', 0, 1),
    algo = tpe.suggest,
    max_evals = 10
)
print(best)


# ## Iris Example
# Now move to examples. For iris, we are doing something similar.  We search the hyperparameter space to find the hyperparameters that minimize the cv score. 

# ### Import and prepare data

# In[4]:


from sklearn.model_selection import train_test_split
import pandas as pd
SEED = 98105

df = pd.read_csv("../input/Iris.csv")
y = pd.factorize(df["Species"])[0]
X = df.drop(["Species", "Id"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = SEED)


# ### KNN Model
# 
# The following is a very simple model. Here we set the model to KNN, and try to adjust `n_neighbors` to get the owest loss. As the following chart shows,  when `n=7`, the loss is the lowest.

# In[5]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK, Trials

# Define cross validation scoring method
def cv_score(params):
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X_train, y_train).mean()
    return {'loss': -score, 'status': STATUS_OK}

# space for searching
space = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 50))
}
trials = Trials() # Recorder for trails

# Train
best = fmin(cv_score, space, algo=tpe.suggest, max_evals=100, trials=trials)


# In[6]:


# Reporting
import matplotlib.pyplot as plt
n_neighbors = [t['misc']['vals']['n_neighbors'][0] for t in trials.trials]
n_neighbors_df = pd.DataFrame({'n_neighbors': n_neighbors, 'loss': trials.losses()})
plt.scatter(n_neighbors_df.n_neighbors, n_neighbors_df.loss)


# ## Complicated SKLean model
# 
# In the following exercise, we will think about how to optimize between several models. Here, we search through the space and find that TPE algorithm make slightly better suggestions. However, the problem is too easy, so there are lots of choices that end up with very good score.
# 
# The best is a svm model, with the following hyperparameters: <br />
# {'C': 1.4017025742525835, 'classifier_type': 1, 'gamma': 0.023448008774836993, 'kernel': 0}
# 

# In[7]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

space = hp.choice('classifier_type', [ # Here we are dealing with a classification problem
    {
        'type': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0),
    },
    {
        'type': 'svm',
        'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0),
    },
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1]),
        'n_jobs': -1
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,50)),
        'n_jobs': -1
    }
])

def cv(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X, y).mean()

def cv_score(params):
    score = cv(params)
    return {'loss': -score, 'status': STATUS_OK}
trials = Trials()
best = fmin(cv_score, space, algo=tpe.suggest, max_evals=1500, trials=trials)
print(best)


# In[27]:


arr = trials.losses()
plt.scatter(range(len(arr)), arr, alpha = 0.3)
plt.ylim(-1, -0.8)
plt.show()


# In[36]:


print("Lowest score:" ,trials.average_best_error())


# ## Train
# The above is only cross-validation. 
# 
# We will need to use all the data to retrain a model and make prediction. Here, we happily achieved 100% accuracy.

# In[42]:


import hyperopt
from sklearn.metrics import accuracy_score

def train(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    model = clf.fit(X, y)
    return model

model = train(hyperopt.space_eval(space, best))

preds = model.predict(X_test)

print("On test data, we acccheived an accuracy of: {:.2f}%".format(100 * accuracy_score(y_test, preds)))

