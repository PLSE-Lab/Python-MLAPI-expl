#!/usr/bin/env python
# coding: utf-8

# # Bagging, random forests, and boosting with animal shelter outcomes
# 
# ## Discussion of bagging
# 
# In a previous notebook ([this one](https://www.kaggle.com/residentmario/decision-trees-with-animal-shelter-outcomes)) I talked about the decision tree algorithm. Decision trees provide great, highly interpretable machine learning models which are usually quite generalizable.
# 
# They also have the property that they are easy to overfit, generating models with little bias but high variance. It turns out that this property makes them ideal fodder for an important machine learning concept: bagging!
# 
# Bagging stands for "bootstrap aggregation". It's an ensemble learning technique: a way of combining several "weaker" machine learners into a single "stronger" one. That's the aggregation part; the bootstrapping part is taking samples (with replacement) of the original dataset, and training each learner on the designated samples.  I covered bootstrapping in [this earlier notebook](https://www.kaggle.com/residentmario/bootstrapping-and-cis-with-veteran-suicides).
# 
# In bagging we build $n$ different purposefuly overfitted models, each one using a different sample of the dataset for training. Then, when predicting, we take the average of the results given by the models as the prediction. That translates to:
# * The average of the prediction values, in a value prediction scenario.
# * The dominant class "voted for" by the models, in a class selection scenario.
# * The average of the model prediction probabilities, in the class probability prediction scenario.
# 
# The sampling with replacement part is what makes this bagging, but there are analogous algorithms (pasting, patching, etcetera) for different selection variable and/or feature selection schemes (described in the `sklearn` documentation: [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)). These are less used however. `sklearn` calls its algorithm the `BaggingClassifier` generically, even though it admittedly encompases more than bagging alone...
# 
# ## Bagging demo
# 
# What follows is an application of the bagging technique to predicting animal shelter outcomes. Note that since we're training bags-of-10 across 5 folds, this takes a while to run!

# In[4]:


import pandas as pd
outcomes = pd.read_csv("../input/aac_shelter_outcomes.csv")

df = (outcomes
      .assign(
         age=(pd.to_datetime(outcomes['datetime']) - pd.to_datetime(outcomes['date_of_birth'])).map(lambda v: v.days)
      )
      .rename(columns={'sex_upon_outcome': 'sex', 'animal_type': 'type'})
      .loc[:, ['type', 'breed', 'color', 'sex', 'age']]
)
df = pd.get_dummies(df[['type', 'breed', 'color', 'sex']]).assign(age=df['age'] / 365)
X = df
y = outcomes.outcome_type.map(lambda v: v == "Adoption")


# In[5]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# In[6]:


from tqdm import tqdm

clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10)

kf = KFold(n_splits=4)
scores = []

for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    scores.append(
        accuracy_score(clf.fit(X_train, y_train).predict(X_test), y_test)
    )


# In[7]:


scores


# In[8]:


sum(scores) / len(scores)


# Simple enough. There's also a `BaggingRegressor` for performing regression, obviously!
# 
# Quick note from [here](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/):
# 
# > The only parameters when bagging decision trees is the number of samples and hence the number of trees to include. This can be chosen by increasing the number of trees on run after run until the accuracy begins to stop showing improvement (e.g. on a cross validation test harness). Very large numbers of models may take a long time to prepare, but will not overfit the training data.

# ## Discussion of random forests
# 
# An improvement on bagged decision trees in practice is the random forest. The utility of bagging stems from the fact that we are injecting randomness into the decision tree algorithms by bootstrapping samples over the dataset, with the average of the randomly perturbed trees being a better classifier overall than any single tree.
# 
# We can extend this advantage even further by being even more random. Random forests are an elaboration on bagged decision trees where the decision trees are trained on a random (bootstrapped) sample of records of the dataset, _and also_ on a random subset of columns in the dataset.
# 
# The resulting decision trees have higher bias and variance, and are more orthogonal to one another than bagged decision trees. The higher bias carries through to the forest output, but the averaging effect negates the higher variance down and ultimately results in lower variance than you would expect with bagged decision trees. If the decreased variance overwhelms the increased bias, then this algorithm will perform better than bagged trees will. When this happens in practice is a matter of art.
# 
# The number of features to select for each of the trees is a user choice (hyperparameter). Good defaults apparently are `sqrt(p)` for classification and `p/3` for regression.
# 
# ## Random forests implementation
# 
# The `sklearn` implementation follows. The only differences from the bagged classifier API is the addition of the `max_features` field, which defaults to the square root rule anyway, so it's not a required parameter; and the fact that `RandomForestClassifier` only works for decision trees, so there's no `base_estimator` parameter.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_features=5)

kf = KFold(n_splits=4)
scores = []

for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    scores.append(
        accuracy_score(clf.fit(X_train, y_train).predict(X_test), y_test)
    )


# ## Extremely randomized trees discussion
# 
# Taking this concept of making ever more random trees one step further, the extremely randomized trees algorithm randomizes the samples, randomizes the features, and then also pseudo-randomizes the splits, by choosing the best-performing of a set of randomly generated decision boundaries. This technically further increases the bias and reduces the post-voting variance.
# 
# This is a computationally expensive algorithm because getting a useful result with trees this random requires on the order of thousands of trees. I don't think I see this algorithm used that often in practice.
# 
# An implementation is omitted for the sake of time. It won't look that different from the implementations we have so far anyway, just one more variable. Here's the import statement:

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# ## Boosting discussion
# 
# The final step in the development of bagging is boosting.
# 
# Boosting adds weights on each of the inputs into the machine learning model you are building in the random tree algorithm. Initially these weights are all set to the same amount, so all of the points are equally important to the weak learners that are being trained. As much trees are trained, the weights are modified: entries that are predicted correctly are decreased in value, while entries that are predicted incorrectly are increased in their weighted value. Thus the later trees in the sequence are especially adapted to fitting the outlying, more-difficult-to-predict points in the dataset.
# 
# This further adapts the voting approach to producing the best overall score (again, at the cost of additional processing time and black boxy-ness).
# 
# `sklearn` has two boosted gradient tree algorithms. The first one is AdaBoost, which dates back to 1995. The second is gradient boosting trees, which is a more recent generalization of the AdaBoost algorithm. Gradient boosting trees as performed by the `XGBoost` and `LightGBM` are the current state-of-the-art for classical (e.g. feature-based) ML problems not amenable to neural networks.

# In[11]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[ ]:




