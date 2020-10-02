#!/usr/bin/env python
# coding: utf-8

# # Notes on tensorflow native estimators
# 
# Estimators are Tensorflow's native high-level API, comparable to `sklearn` in look and feel. They sit on top of the library's lower layers-based API (demonstrated [here](https://www.kaggle.com/residentmario/tensorflow-high-level-eager-interface/)), which itself sits on top of the lowest level raw computational graph API (discussion [here](https://www.kaggle.com/residentmario/tensorflow-system-design-notes)).
# 
# Estimators handle sessionization (building and tearing down the `Session`) and graph construction (likewise with the `Graph`) for you. They are designed to pipeline easily and to be extensible and buildable.
# 
# Using estimators comes down to four steps. First, define an input function, whose job will be to return a tuple whose first item is a feature-tensor key value pair and whose second item is a tensor of labels. Next, define the feature types, using e.g. `tf.feature_column.numeric_column`. Next instantiate the estimator. Finally, run the estimator on the data.
# 
# The one additional step over `sklearn` is the need to pass the features to objects ahead of computation. This is because it needs you to associate type information with your data.
# 
# You can use the Tensorflow datasets API to handle pre-processing. But you can also use any other data processing tool to do the job...like, say, `pandas`.
# 
# The three functions on an Estimator are `train`, `evaluate`, and `predict`. `train`-`predict` is equivalent to `fit`-`predict` in `sklearn`. `evaluate` returns some information on classifier performance, though you can obviously also evaluate outside of the loop.

# ## Data
# 
# For this demo I'm going to use a synthetic classification dataset with some overlap. There'll be just two numeric features between the classes, and a plot of what it all looks like follows.

# In[ ]:


import numpy as np
from sklearn.datasets import make_classification

np.random.seed(42)
X, y = make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0)
n_train_samples = 1000

X_train, y_train = X[:n_train_samples], y[:n_train_samples]
X_test, y_test = X[n_train_samples:], y[n_train_samples:]


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.axis('off')


# ## Estimators
# 
# Let's tour through the estimators now.
# 
# The code that follows is a skeleton for training with the Tensorflow estimators API. Notice the need to do data transforms out of `numpy` and into Tensorflow's format, using the `tf.data` API to do so. It uses the `BaselineClassifier`, which just predicts the average (in this classifier example that works out to the dominant class).
# 
# Note that all of the estimators have a classifier and a regressor flavor, but this section only looks at recipes for classifiers.

# In[ ]:


import tensorflow as tf


def input_fn(X, y): # returns x, y (where y represents label's class index).
    dataset = tf.data.Dataset.from_tensor_slices(({'X': X[:, 0], 'Y': X[:, 1]}, y))
    dataset = dataset.shuffle(1000).batch(1000)
    return dataset


from tensorflow.estimator import BaselineClassifier
clf = BaselineClassifier(n_classes=2)


clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))

# Convert object-wrapped prediction iterator to a list.
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Now the linear classifier. This is just a linear regression model trained using a stochastic gradient descent process (as opposed to the well-known matrix math solution). The regression companion is the `LinearRegressor`, of course.

# In[ ]:


from tensorflow.estimator import LinearClassifier
feature_columns = [
    tf.feature_column.numeric_column(key='X', dtype=tf.float32),
    tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
]
clf = LinearClassifier(n_classes=2, feature_columns=feature_columns)


clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Now the `DNNClassifier`. Although DNN stands for "deep neural network", this architecture is not fundamentally deep. Instead `DNNClassifier` implements an old-school multilayer perceptron, but MLPClassifier doesn't "sound as cool".  More comments [here](https://stackoverflow.com/questions/48431870/what-does-dnn-mean-in-a-tensorflow-estimator-dnnclassifier). If you use enough nodes it becomes "deep", but you don't have to use so many nodes to have the desired effect.

# In[ ]:


from tensorflow.estimator import DNNClassifier
feature_columns = [
    tf.feature_column.numeric_column(key='X', dtype=tf.float32),
    tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
]
clf = DNNClassifier(n_classes=2, feature_columns=feature_columns, hidden_units=[32, 32])


clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10000)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# `BoostedTreesClassifier` implements gradient boosted trees. It only allows bucketized or identity columns (see [here](https://www.kaggle.com/residentmario/tensorflow-feature-columns/) for details on feature types), which is strange and indicates a rough and incomplete edge in the library.

# In[ ]:


# from tensorflow.estimator import BoostedTreesClassifier
# feature_columns = [
#     tf.feature_column.numeric_column(key='X', dtype=tf.float32),
#     tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
# ]
# clf = BoostedTreesClassifier(n_classes=2, feature_columns=feature_columns, n_trees=100, n_batches_per_layer=1)


# clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10000)
# y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
# y_pred = np.array([p['class_ids'][0] for p in y_pred])

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))


# One interesting final option is `DNNLinearCombinedClassifier`, which combines a linear classifier with a DNN. The linear classifier is "wide", and deals well with having lots of features, whilst the NN is "deep", and deals with complex relationships within particularly important features. By combining the two models and averaging their outputs, in some way, we may arrive at a more accurate overall prediction. An interesting idea, to be sure.

# In[ ]:


from tensorflow.estimator import DNNLinearCombinedClassifier
feature_columns = [
    tf.feature_column.numeric_column(key='X', dtype=tf.float32),
    tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
]
clf = DNNLinearCombinedClassifier(n_classes=2, dnn_feature_columns=feature_columns, dnn_hidden_units=[32, 32], linear_feature_columns=feature_columns)

clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10000)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# That's actually the entirely of the `tf.estimator` API! This module seems to exist mainly as a primer and an example showcase of what kinds of machine learning -focused computational graphs you can build, and what they might look like, moreso than a practical machine learning library in and of itself.
# 
# Digging deeper, we find that TensorFlow in general seems to have a history of building and then deprecating different attempts at high level APIs. There used to be a high-level API named `tf.slim`, which many old code examples still use, but which was deprecated a while ago.  The `tf.estimators` API is not an attempt at a replacement so much as it is a methodology for defining custom estimators within Tensorflow code. The module segment used in the current tutorials, including the Machine Learning Crash Course, is `tf.layers`...except that `tf.layers` is [also going to be deprecated soon](https://github.com/tensorflow/tensorflow/issues/14703#issuecomment-376654034). This last fact is particularly funny to me, as it shows two different teams within Google totally failing to coordinate; I don't think introducing a to-be-deprecated module as The Way To Do Things is exactly a wise idea, and I'm impressed I had to go as far as GitHub issue to dig this up!
# 
# For neural machine learning algorithms you could use the `tf.keras` module, which is Tensorflow's [Keras](https://keras.io/) integration. Keras is a spec for implementing a machine learning model which provides a well-known and well-liked machine learning API . Keras is Tensorflow-independent as a project, and is also the API implemented in CNTK and Theano, amongst other libraries. These libraries simply provide the backing service to Keras, whilst Keras provides the APIs and the abstractions.
# 
# What's interesting is the fact that `tf.keras` is a Keras project independent implementation of the Keras spec. It's arguably easier to use if you are coding with Tensorflow abstractions as your first-class objects, but it's also probably outdated and doesn't see much commit volume. You're better off using Keras directly, with the Tensorflow backend enabled, unless you are really married to Tensorflow (more deets in [this StackOverflow answer](https://stackoverflow.com/questions/44068899/what-is-the-difference-between-keras-and-tf-contrib-keras-in-tensorflow-1-1)). Why would you be married to the Tensorflow implementation? Because it has deeper integration with the library, featuring better support for things like Tensorboard and the like.

# ## Custom estimator
# 
# The `tf.estimator` API can be used for implementing a custom estimator. Coming soon?
