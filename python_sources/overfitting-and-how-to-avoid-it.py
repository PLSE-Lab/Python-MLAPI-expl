#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# We'll briefly go over, how to set up a validation system for approaching a machine learning problem. If you want an introduction to Kaggle's online kernels, you can check the public "EDA"-kernel.

# In[ ]:


import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")


# ## Train and test split
# Since machine learning models are built to make predictions of future events, it's critical to evaluate their performance on **unseen** data. Unseen data just refers to data that the model hasn't been trained on (it hasn't seen it before). This is commonly done by splitting your original training dataset into two smaller datasets: a train and test set. Then we train the model on the train set, and test the model on the test set. This is easily done with the module "train_test_split" from SKLearn.
# 
# Let's start by separating our independent (input) and dependent (target) variables.

# In[ ]:


X_train_original = train[["0", "1", "2", "3", "4"]]
y_train_original = train["target"]


# Now we use the module from SKLearn to further split our entire training dataset into two smaller ones. Note that we can change the size of our test set.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_original, y_train_original, test_size=0.2)
print(X_train.shape, X_test.shape)


# ## Overfitting

# Let's now make a simple k-nearest neighbour classifier and train it on the training set.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)


# Let's see how it performs on our train set using the competition evaluation metric (log loss).

# In[ ]:


from sklearn.metrics import log_loss

probas = clf.predict_proba(X_train)
ll = log_loss(y_train, probas)

print("Log loss: {0}".format(ll))


# That's a much better log loss than the logistic regression baseline - but let's see how it performs on unseen data; our test set.

# In[ ]:


probas = clf.predict_proba(X_test)
ll = log_loss(y_test, probas)

print("Log loss: {0}".format(ll))


# Well, that's not what we saw on the training set. This is what is called "overfitting"; our model has learned how to predict our training set *too* well, so it's not able to generalize.
# 
# To address this problem, we can change the hyperparameters in our model, until we're satisfied with the performance on the test set.

# In[ ]:


for n in range(5):
    print("="*40)
    print("Neighbours: {0}".format(n+1))
    
    clf = KNeighborsClassifier(n_neighbors=n+1)
    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_train)
    ll = log_loss(y_train, probas)
    print("Train log loss: {0}".format(ll))
    
    probas = clf.predict_proba(X_test)
    ll = log_loss(y_test, probas)
    print("Test log loss: {0}".format(ll))


# By adjusting the hyperparameters in our model, we can get the train and test set to *agree* more.
# 
# Once we're happy with the result, we can use the final model to train on our entire (original) training set and make our predictions on the test set.

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_original, y_train_original)
test_probas = clf.predict_proba(test[["0", "1", "2", "3", "4"]])


# Then we take our predicted probabilities for class 1 and use as our submission.

# In[ ]:


sample_submission["target"] = test_probas[:,1]
sample_submission.to_csv("knn_baseline.csv",index=False)


# ## Further notes
# We just skimmed the surface of validation in machine learning. Validation is used for many different things and there are much more complicated validation systems. A big part of building a machine learning model is figuring out exactly how you can properly validate your model; making sure it performs as well as you think. This varies from problem to problem, but a commonly used (thorough) validation method is called **k-fold cross-validation**.
# 
# If you want to know a bit more about validation, a good place to start is just wikipedia: https://en.wikipedia.org/wiki/Cross-validation_(statistics).
