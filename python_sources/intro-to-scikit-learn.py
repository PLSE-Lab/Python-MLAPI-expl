#!/usr/bin/env python
# coding: utf-8

# # Intro to scikit-learn
# Jake Lee, TA for COMS4701 Fall 2019
# 
# ## Introduction
# I'm putting together this guide for those who have never used a machine learning package before to ease the learning curve. I don't want to give away too much of the homework, so I will be using functions not used for HW7 as well as a different dataset.
# 
# ## Useful Links and Documentation for HW7 P5
# 
# - scikit-learn's main website
#   - https://scikit-learn.org/stable/
# - tic-tac-toe dataset on OpenML
#   - https://www.openml.org/d/50
#   - https://scikit-learn.org/stable/datasets/index.html#downloading-datasets-from-the-openml-org-repository
# - Multinomial Naive Bayes classifier reference
#   - Doc: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
#   - User Guide: https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
# - Decision Tree classifier reference
#   - Doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
#   - User Guide: https://scikit-learn.org/stable/modules/tree.html#tree
# - Perceptron reference
#   - Doc: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
#   - User Guide: https://scikit-learn.org/stable/modules/linear_model.html#perceptron
# - kNN reference
#   - Doc: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#   - User Guide: https://scikit-learn.org/stable/modules/neighbors.html#classification

# # Data Setup
# This section of the homework may not be worth any points, but it is the most important. Data saving and data processing are often the most time-consuming, and if done incorrectly, will result in incorrect results for all classifiers.
# 
# Instead of tic-tac-toe (which is your assignment), we're going to look at the Glass Identification Data Set [1], originally distributed through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/glass+identification) but also accessible through [OpenML](https://www.openml.org/d/41).
# 
# Before we jump into importing, we should first understand what this dataset actually is. In the description:
# > The study of classification of types of glass was motivated by criminological investigation. At the scene of the crime, the glass left can be used as evidence...if it is correctly identified!
# 
# Cool! We also see that each data point has 9 *numeric* features. RI, the refractive index, and 8 elements (unit: weight percent in corresponding oxide, presumably measured with X-ray fluorescence). *hint: for tic-tac-toe, notice that all features are nominal, not numeric. Some classifiers don't like this; you probably want to look for a [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).*
# 
# We see that there are 214 items total. Finally, we see that we're trying to classify each item into one of six classes: 
# 1. build wind float
# 2. build wind non-float
# 3. vehic wind float
# 4. vehic wind non-float (NO SAMPLES PROVIDED)
# 5. containers
# 6. tableware
# 7. headlamps
# 
# Great, now we can jump in to actually importing the data. I recommend you do this for the tic-tac-toe dataset also before jumping in.
# 
# [1] Evett, Ian W., and Ernest J. Spiehler. "Rule induction in forensic science." *KBS in Goverment* (1987): 107-118.

# In[ ]:


import numpy as np
from sklearn.datasets import fetch_openml
# by the way, you can't just do "import sklearn"! Each subpackage must be directly imported.
# you could do "import sklearn.datasets" and then call "sklearn.datasets.fetch_openml()", 
# or "from sklearn import datasets" then call "datasets.fetch_openml()",
# but that's so long!

glass_x, glass_y = fetch_openml(data_id=41, return_X_y=True)
print(glass_x.shape)
print(glass_y.shape)


# I have my data, and I have my labels. Now I want to split this into a train/test set so I can both train my model and see how well it does. We'll use a 75/25 split, which is default for the function we'll use. We also want to shuffle the dataset in this case before splitting because the data is in order - `train_test_split` already does this for us.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(glass_x, glass_y, random_state=4701)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Classification
# 
# Now that we have our train and test sets, we can train a classification model. We'll be using the Linear Support Vector Classification model because that's not in the homework. It's not important for you to know how LinearSVC works for this walkthrough.
# 
# - LinearSVC
#   - Doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
#   - User Guide: https://scikit-learn.org/stable/modules/svm.html#svm-classification
# 
# Due to the classifier and the dataset I'm using, I have to normalize my feature vectors so that each feature ranges between -1 to 1. You do not have to do this for the tic-tac-toe dataset and associated classifiers, but you may have to do this for the Kaggle competition.

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# Let's train and classify:

# In[ ]:


# import the classifier
from sklearn.svm import LinearSVC

# instantiate the linearSVC object
clf = LinearSVC(random_state=4701, max_iter=10000)
# train it on the training data and label
clf.fit(X_train_s, y_train)
# now predict the test set
pred = clf.predict(X_test)
print(pred)


# Uh oh! That doesn't look right! What happened?
# 
# Looks like I trained on the scaled training set but I predicted on the unscaled test set. I'll fix that and...

# In[ ]:


# now predict the test set
pred = clf.predict(X_test_s)
print(pred)


# That looks much better, but we still don't know how well the classifier is really doing. We do have the ground truth for the test set, so now we can calculate the *classification accuracy*, that is, the percentage of items that were correctly classified. Of course, sklearn has a function for that.

# In[ ]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)


# That's... okay, considering there are six classes total, making the random accuracy score 16.67%. Still, I wonder if another classifier would perform better... :)
# 
# This concludes our introduction to scikit-learn. This should be more than enough to get you started on HW7 P5, even if you've never used this package before.

# # Kaggle-Specific Tips
# 
# If you're planning on using scikit-learn for your Kaggle submission, Consider the following.
# 
# - Images are provided in a 2D array - you'll need to reshape this into a 1D array before doing any classification.
# - Pixel values range from 0-255 - double check that your classifier does not require normalization of some type.
# - At the end, you need to write out the results of `predict()` on the provided test set to a CSV file as specified. Consider the `csv` package.
# - I don't recommend using sklearn if you're going for the extra credit - deep learning is much more powerful.

# As always, if you enjoyed the writeup, click below to upvote!
