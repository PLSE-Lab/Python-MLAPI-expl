#!/usr/bin/env python
# coding: utf-8

# # This kernel is a basic introduction to Ensemble learning using sklearn for newbies
# # importing neccessary libaries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # Voting classifier
suppose you have trained a few classifier, each one achieving about 80% accuracy.
You may have a logistic regression classifier, an SVM classifier, a random forest classifier, a K-nearest neighbor
classifier and perhaps a few more.

A very simple way to create a even better classifier is to aggregate the predictions of each classifier
and predict the class that gets the most votes.

Somewhat, the voting classifier often achieves a higher accuracy than the best classifier in the ensemble.

In fact, even if each classifier is a weak learner (meaning it does only slightly better than random guessing), 
the ensemble can still be a strong learner (achieving a higher accuracy), provided there are a sufficient number
of weak learner and they sufficiently diverse.
# In[ ]:


# The following code creates and train a voting classifier in sklearn, composed of 3 divers classfier on the moon dataset


# In[ ]:


import sklearn


# In[ ]:


from sklearn.datasets import make_moons
x, y = make_moons()


# In[ ]:


x.shape, y.shape


# In[ ]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 12)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators = [('lr',log_clf), ('rf',rnd_clf),('svc',svm_clf)], voting = 'hard')

voting_clf.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))


# # soft voting
if all classifiers are able to estimate class probablities (i.e, they have a predict_proba method), they you can tell sklearn to predict the class with the 
highest class probability, averaged over all the individual classifiers.

we have soft voting by replacing voting = 'hard' with 'soft'
# In[ ]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability = True)

voting_clf = VotingClassifier(estimators = [('lr',log_clf), ('rf',rnd_clf),('svc',svm_clf)], voting = 'soft')

voting_clf.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))


# # Bagging and pasting
It involves using the same training algorithms for every predictor, but to train
them on different random subsets of the training set.

Bagging involves sampling with replacement
pasting involves sampling without replacement
# # bagging and pasting in sklearn

# In[ ]:


# the following code trains an ensemble of 500 decision tree classifier, each
# training instances randomly sampled from the training set with replacement

# to use pasting set bootstrap =false


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500,
                          max_samples = 70, bootstrap = True, n_jobs = -1)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf, bag_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))


# # out of bag evaluation
with bagging, some instances may be sampled several times for any given predictor, while others may not be sampled at all. The precentage of unsampled
data are called out-of-bag(oob).

Since a predictor never sees the oobs instances during training, it can be evaluated on these instances, without the need for a separate validation set or cross-validation.
# In[ ]:


# Setting oob_score = True while creating a bagging classifier to request an 
# automatic oob evaluation after training.

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500,
                           bootstrap = True, n_jobs = -1, oob_score = True)
bag_clf.fit(x_train, y_train)
bag_clf.oob_score_


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(x_test)
accuracy_score(y_test, y_pred)


# In[ ]:


bag_clf.oob_decision_function_

The bagging classifier class support sampling the features as well. This is controlled by two hyperparameters: max_features and bootstrap_features. They work the same way as max_samples and bootstrap, but for feature sampling instead of instance sampling.
# # random forest classifier
 random forest are optimized for decisiontrees
 with few exceptions, a randomforestclassifier has all the hyperparameters of a decisiontreeclassifier(to control how trees are grown), plus all the hyperparameters of a bagging classifier to control the ensemble itself.
 
 Random forest algorithm introduces extra randomness when growing trees; instead of searching for the best feature when spliting a node. It searches for the best feature among a random subset of features.
# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1)
rnd_clf.fit(x_train, y_train)

y_pred_rf = rnd_clf.predict(x_test)


# In[ ]:


# the following bagging classifier is roughly equivalent to the previous randomfores classifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter = 'random', max_leaf_nodes = 16),
                           n_estimators = 500, max_samples = 1.0, bootstrap = True ,n_jobs = -1)


# In[ ]:


bag_clf.fit(x_train, y_train)
rand_bag_clf = bag_clf.predict(x_test)
accuracy_score(y_test, y_pred)


# # EXtra Tree classifier

# A forest of extremely random trees is simple called an extremely randomized tree ensemble (or extra trees for short).

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

extr_cls = ExtraTreesClassifier()


# # feature importance with randomforest

# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print (name, score)


# # Boosting

# Boosting (originally called hypothesis boosting) refers to any ensemble method that can combine several weak learners into a strong learner.
# The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.

# # Adaboost

# One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. This results in new predictors focusing more and more on the hard cases. this is the technique used by Adaboost.

# In[ ]:


# weighted error rate of the Jth predictor

def weighted(x):
    return x


def weighted_error_rate(data,target, predictor):
    no_of_samples = data.shape[0]
    predict_value = predictor.predict(data)
    weighted_value = [weighted(i) if i != j else int(1) for i,j in zip(predict_value, target)]
    normal_weight = [weighted(i) for i in predict_value]
    solution = [a/b for a,b in zip(weighted_value, normal_weight)]
    return solution
    


# In[ ]:


def predicators_weight(data,target,predictor,learning_param = 0.3):
    compute_1 = weighted_error_rate(data,target,predictor)
    compute_2 = np.log((1-compute_1)/compute_1)
    return learning_param * compute_2


# In[ ]:


def alpha(val):
    return val

def updated_weights(predicted_value, value):
    if predicted_value == value:
        val = weighted(predicted_value)
    else:
        val = weighted(predicted_value) * np.exp(alpha(value))
    return val


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 200, 
                            algorithm = 'SAMME.R', learning_rate = 0.5)
ada_clf.fit(x_train, y_train)


# # Gradient boosting

# It works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictors.

# from sklearn.tree import DecisionTreeRegressor
# x, y = None, None
# 
# 
# 
# tree_reg1 = DecisionTreeRegressor(max_depth = 2)
# 
# tree_reg1.fit(x,y)
# 
# y2 = y - tree_reg1.predict(x)
# 
# tree_reg2 = DecisionTreeRegressor(max_depth = 2)
# 
# tree_reg2.fit(x, y2)
# 
# y3 = y2 - tree_reg2.predict(x)
# 
# tree_reg3 = DecisionTreeRegressor(max_depth = 2)
# 
# tree_reg3.fit(x,y3)
# 
# 
# y_pred = sum(tree.predict(x_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

# from sklearn.ensemble import GradientBoostRegressor
# 
# gbrt = GradientBoostRegressor(max_depth = 2, n_estimators = 3, learning_rate = 1.0)
# gbrt.fit(x,y)
# the following doe trains a GBRT ensemble with 120 trees, then measures the validation
# error at each stage of training to find the optimal number of trees, and finally trains
# another gbrt ensemble using the optimal number of trees.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_val, y_train, y_val = train_test_split(X,y)

gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators = 120)
gbrt.fit(x_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(x_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostRegressor(max_depth = 2, n_estimators =bst_n_estimators)
gbrt_best.fit(x_train, y_train)
#  it is also possible to implement early stopping training early (instead of training
#  a large number of trees first and then looking back to find the optimal number).
#  we can do by setting warm_start = True
# from sklearn.ensemble import GradientBoostingRegressor
# 
# gbrt = GradientBoostingRegressor(max_depth = 2, warm_start = True)
# 
# min_val_error = float('inf')
# 
# error_going_up = 0
# 
# for n_estimators in range(1,120):
# 
#     gbrt.n_estimators = n_estimators
#     
#     gbrt.fit(x_train, y_train)
#     
#     y_pred = gbrt.predict(x_val)
#     
#     val_error = mean_squared_error(y_val, y_pred)
#     
#     if val_error < min_val_error:
#     
#         min_val_error = val_error
#         
#         error_going_up = 0
#         
#     else:
#         error_going_up += 1
#         
#         if error_going_up == 5:
#         
#             break # early stopping

# In[ ]:




