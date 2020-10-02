#!/usr/bin/env python
# coding: utf-8

# # Lecture 1: k-Nearest Neighbors and Handwritten Digit Classification

# In this example, we'll use k-NN to classify 8x8 pixel images of hand-written digits.  The k-NN classifier is park of scikit-learn:
# * [http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
# 
# ---
# Forked from: https://github.com/cmmalone/UChicago_ML

# In[ ]:


# This command makes sure that we can see plots we create as part of the notebook & without having to save & read files
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sys
from logging import info

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import datasets, neighbors, preprocessing

# Let's print what versions of the libraries we're using
print(f"python\t\tv {sys.version.split(' ')[0]}\n===")
for lib_ in [np, pd, sns, sklearn, ]:
    sep_ = '\t' if len(lib_.__name__) > 8 else '\t\t'
    print(f"{lib_.__name__}{sep_}v {lib_.__version__}"); del sep_


# The dataset consists of 1,797 images, each 8 pixels by 8 pixels.  The "target" field has the label, telling us the true digit the image represents.

# In[ ]:


# The digits dataset
digits = datasets.load_digits()


# In[ ]:


digits.images.shape


# In[ ]:


digits.target


# Here, we define a function that takes an image and the true label and plots it for us:

# In[ ]:


def plot_handwritten_digit(the_image, label): # plot_handwritten_digit<-function(the_image, label)
    plt.figure(figsize=(2, 2))
    plt.axis('off')
    plt.imshow(the_image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# In[ ]:


# this will show us the pixel values
image_num = 1000
digits.images[image_num]


# In[ ]:


# and then we can plot them
plot_handwritten_digit(digits.images[image_num], digits.target[image_num])


# In[ ]:


fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


# Instead of each image being 8x8 pixels, we flatten it to just be a single row of 64 numbers:

# In[ ]:


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, 64))
labels = digits.target


# In[ ]:


data.shape


# If we need to standardize the features (make them all have mean zero and standard deviation one), this is how we do it:

# In[ ]:


data_scaled = preprocessing.scale(data)
data_scaled


# In[ ]:


data.mean(axis=0)


# In[ ]:


data_scaled.mean(axis=0)


# Make a training set and a test set.  We'll use the nearest neighbors from the training set to classify each image from the test set.

# In[ ]:


n_train = int(0.9*n_samples)

X_train = data[:n_train]
y_train = labels[:n_train]
X_test = data[n_train:]
# re-shape this back so we can plot it again as an image
test_images = X_test.reshape((len(X_test), 8, 8))
y_test = labels[n_train:]


# In[ ]:


X_train.shape


# Scikit-learn classifiers generally have a standard programming interface.  You construct the class:

# In[ ]:


knn = neighbors.KNeighborsClassifier(n_neighbors=5)


# You fit it to your data:

# In[ ]:


knn.fit(X_train, y_train)


# And you predict on new data:

# In[ ]:


pred_labels = knn.predict(X_test)
pred_labels


# In[ ]:


pred_probs = knn.predict_proba(X_test)
pred_probs


# In[ ]:


test_num = 11
plot_handwritten_digit(test_images[test_num], y_test[test_num])
print("true label is %s" % y_test[test_num])
print("predicted label is %s" % pred_labels[test_num])
print("predicted probabilities are %s" % pred_probs[test_num])


# Let's find examples where the predicted label is wrong:

# In[ ]:


np.where(pred_labels != y_test)


# In[ ]:


test_num = 41
plot_handwritten_digit(test_images[test_num], y_test[test_num])
print("true label is %s" % y_test[test_num])
print("predicted label is %s" % pred_labels[test_num])
print("predicted probabilities are %s" % pred_probs[test_num])


# In[ ]:


test_num = 43
plot_handwritten_digit(test_images[test_num], y_test[test_num])
print("true label is %s" % y_test[test_num])
print("predicted label is %s" % pred_labels[test_num])
print("predicted probabilities are %s" % pred_probs[test_num])


# In[ ]:


for test_num in np.where(pred_labels != y_test)[0]:
    print(f"true label is {y_test[test_num]}"
          f"\npredicted label is {pred_labels[test_num]}"
         )
    print("predicted probabilities are %s" % pred_probs[test_num])
    plot_handwritten_digit(test_images[test_num], y_test[test_num])
    plt.show()


# In[ ]:




