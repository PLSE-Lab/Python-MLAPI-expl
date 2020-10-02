#!/usr/bin/env python
# coding: utf-8

# # Using keras models with scikit-learn pipelines
# 
# ## Discussion
# 
# `sklearn` is Python's general purpose machine learning library, and it features a lot of utilities not just for building learners but for pipelining and structuring them as well. `keras` models don't work with `sklearn` out of the box, but they can be made compatible quite easily. To be compatible with `sklearn` utilities on a basic level a learner need only be a class object with `fit`, `predict`, and `score` methods (and optionally a `predict_proba`), so you can write a quick object wrapper that delegates these methods on a `keras` object.
# 
# However this is unnecessary because `keras` comes with a wrapper built in. This is described in the `keras` docs [here](https://keras.io/scikit-learn-api/).
# 
# There are two wrappers, one for classifiers and one for regressors. The signatures are `keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)` and `keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)` respectively.
# 
# The `build_fn` parameter should be given a factory function (or a functional object, e.g. an object that defines a `__call__()` method) that returns the model.
# 
# The `sk_params`parameter can be used to pass parameters to the `fit`, `predict`, `predict_proba`, and `score` methods, which are the aforementioned "standard interface" methods for a scikit-learn compatible predictor. These methods by default will take on the values that you set for them inside of the factory function, but this parameter does is it allows you to change them manually at call time. To change parameters unambiguously, use a dictionary whose first-level keys is the methods whose parameters are being modified. To change parameters all at once, pass a top-level key-value pair; this will be passed down to all of these methods.

# ## Demonstration
# 
# A quick demonstration of this wrapper follows. In this code, we will perform cross validation on the Keras model accuracy using the `StatifiedKFold` method in the `sklearn` library.

# In[ ]:


#
# Generate dummy data.
#

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
import pandas as pd

X_train = np.random.random((1000, 3))
y_train = pd.get_dummies(np.argmax(X_train[:, :3], axis=1)).values
X_test = np.random.random((100, 3))
y_test = pd.get_dummies(np.argmax(X_test[:, :3], axis=1)).values


# In[ ]:


#
# Build a KerasClassifier wrapper object.
# I had trouble getting the callable class approach to work. The method approach seems to be pretty universial anyway.
#

from keras.wrappers.scikit_learn import KerasClassifier

# Doesn't work?
# class TwoLayerFeedForward:
#     def __call__():
#         clf = Sequential()
#         clf.add(Dense(9, activation='relu', input_dim=3))
#         clf.add(Dense(9, activation='relu'))
#         clf.add(Dense(3, activation='softmax'))
#         clf.compile(loss='categorical_crossentropy', optimizer=SGD())
#         return clf

def twoLayerFeedForward():
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=3))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf


# clf = KerasClassifier(TwoLayerFeedForward(), epochs=100, batch_size=500, verbose=0)
clf = KerasClassifier(twoLayerFeedForward, epochs=100, batch_size=500, verbose=0)


# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

trans = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

import pandas as pd

# Keras classifiers work with one hot encoded categorical columns (e.g. [[1 0 0], [0 1 0], ...]).
# StratifiedKFold works with categorical encoded columns (e.g. [1 2 3 1 ...]).
# This requires juggling the representation at shuffle time versus at runtime.
scores = []
for train_idx, test_idx in trans.split(X_train, y_train.argmax(axis=1)):
    X_cv, y_cv = X_train[train_idx], pd.get_dummies(y_train.argmax(axis=1)[train_idx]).values
    clf.fit(X_cv, y_cv)
    scores.append(clf.score(X_cv, y_cv))


# In[ ]:


scores


# ## Conclusion
# 
# As you can see the accuracy scores that we achieve vary widely between consequetive runs, indicating that our model has not yet found enough signal in the dataset (we should increase the number of epochs, increase the learning rate, or decrease the batch size, or all of the above).
# 
# I couldn't get the callable class approach to work, unfortunately. I didn't want to get bogged down digging too deep, but in poking around online I noticed that every example I see uses a factory function to build the Keras classifier...
# 
# There are still some awkward edges around the interaction of `keras` and `sklearn`. In this example we see that we have to perform representational transformations on the target columns from one-hot to a categorical encoding to one-hot encoding again. So it seems that whilst having this pipeline code helps a lot, there's still some glue code that you have to write yourself!
# 
# [Wrapping Keras learners in scikit-learn pipelines](https://www.kaggle.com/residentmario/pipelines-with-linux-gamers) seems like a good way to go for production development environments.
# 
# You can use Keras from within a `scikit-learn` pipeline as a grid search target: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/.
