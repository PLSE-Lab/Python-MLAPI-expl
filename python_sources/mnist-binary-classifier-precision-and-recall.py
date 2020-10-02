#!/usr/bin/env python
# coding: utf-8

# # MNIST Binary Classifier - precision and recall
# In this kernel we will detect one number from MNIST data set using binary classifiers (classifiers that detect only one class). 
# 
# We'll look at** precision and recall** charts to find the best classifier.
# 
# First - import MNIST data from CSV

# In[ ]:


# import all common modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# get data
train = pd.read_csv("../input/train.csv")
y = train['label']
X = train.drop(['label'], axis=1)

#X = X.values.reshape(-1,28,28)
X = X.values
y = y.values

# delete train to gain some space
del train

print("Shape of X:{0}".format(X.shape))
print("Shape of y:{0}".format(y.shape))


# Display single image to be sure it's loaded properly

# In[ ]:


# get single digit graphical data
# to display it, we need to convert single line of 784 values to 28x28 square
digit_image = X[3].reshape(28,28)

plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# Split sets to train and test

# In[ ]:


# split data for test and for training
# data before split_index will go for training and
#  data after split_index will go for testing
split_index = int(X.shape[0]*0.8)

X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

shuffle_index = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# # Training a binary classifier
# Using Stochastic Gradiend Descend classifier

# In[ ]:


y_train_5 = y_train == 5 # True for all 5s, False for all other digits
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, max_iter=10)
sgd_clf.fit(X_train, y_train_5)


# # Model validation
# Classifier performance evaluation is often more trickier than regressor evaluation.
# Here's a simple cross validation
# 
# I'm using here StratifiedKFold object to run split method for X_train and y_train_5 arrays.
# 
# This method will split the data into 3 (n_splits=3) sets, making sure that each class taken from labels array (y_train_5) shows up same amount of times in each set.
# For example if we have 10 digits, each fold should contain around 10% of each digit. That's what stratified sampling does - splits sets equally for given attribute, which in our case is digit category.
# 
# Each fold contains 1/3 of the data because we have 3 splits. Data is not repeated in any fold.

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

i=0;
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("Correct ratio for fold {1}: {0}".format(n_correct / len(y_pred), i))
    i += 1


# Scikit-Learn has a handy method `cross_val_score` that can do the same as method above, returning the result in an array:

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# ## Comparing with "dumb" classifier
# Even if ratio above 0.9 looks great at first glance, we need to be aware of 1 thing. 
# 
# **There is only around 10% of "5"s in our data.**
# 
# If we would guess all the time that our digit is not 5 we would be 90% correct. We can check this by creating a dumb "Never5Classifier" by extending Scikit-Learn's BaseEstimator:

# In[ ]:


from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# # Confusion Matrix
# A much better way to evalueate the performance of a classifier is to look at the confusion matrix.
# It will tell us how many times (in our case) 
# * 5 was classified as 5 (it's called True Positive)
# * 5 was incorrectly classified as not 5 (it's called False Negative)
# * not 5 was classified as not 5 (it's called True Negative)
# * not 5 was incorrectly classified as 5 (it's called False Positive)
# 
# To compute the confusion matrix, we need to predict all the training set to be evaluated with their corresponding labels.
# We can use cross_val_predict() function for this

# In[ ]:


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print("First 10 predictions of number 5: {0}".format(y_train_pred[:10]))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# What this array tells us is that 
# * 30269 examples were classified as not 5
# * 1950 examples were classified as 5
# * 283 examples were classified as 5 but actually thy were not 5
# * 1098 examples were classified as not 5 but they were 5
# 
# We read this array in a way that: **Columns** are corresponding to our class predictions,
# **Rows** are corresponding to actual classes
# 
# We can calculate from this 2 important metrics **precision** and **recall**.
# 
# **Precision** is the amount of True Positives divided by the sum of True Positives and False **Positives** (TP/(TP+FP)) Precision is the accuracy of positive predictions. If we would have a classifier with 100% precision, it would correctly detect all 5. It would never detect non 5 as 5, but it could incorrectly detect some 5 as non 5.
# 
# **Recall** is the amount of True Positives divided by the sum of True Positives and False **Negatives** (TP/(TP+FN)) Recall is also called *sensitivity* or *true positive rate (TPR)*. Classifiers with higher recall will detect more instances of 5 but also indorrectly detect some other digits as 5.
# 
# Scikit-Learn has handy methods for this purpose: precision_score and recall_score:

# In[ ]:


from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))


# Now our classifier does not look that good as it seemed at the beginning.
# # The F1 score
# F1 score is precision and recall combined into single metric. It's the harmonic mean of precision and recall

# In[ ]:


from sklearn.metrics import f1_score
score = f1_score(y_train_5, y_train_pred)
print(score)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print(y_scores)


# **F1 score favors classifier that have similar precission and recall.** This may be not always what we want. Sometimes we want to have a model with greater precision, but this will as a consequence lower the recall rate. We can visualize the precision / recall ratio in a line chart using precision_recall_curve function from sklearn.metrics:

# In[ ]:


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# Scikit-Learn won't let us set threshold directly, but it will give us access to decision scores it uses to make predictions. Using decision_function() we can get score values and decide whether it should be classified as 5 or not 5.

# In[ ]:


y_scores = sgd_clf.decision_function([X_train[0]])
print("Score for 1st digit: {0}".format(y_scores[0]))
print("Was this digit a real 5? {0}".format(y_train_5[0]))

digit_image = X_train[0].reshape(28,28)
plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.title("Digit image")
plt.show()


# We can use custom threshold value to fine-tune our classifier in precision / recall space.
# In this extreme example, by setting thresold to very low value -250000, we can even predict image that displays 1 as 5.

# In[ ]:


threshold = -250000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# You can use precision / recall chart to select best threshold value for your specific task. 
# 
# There is also another way to select best threshold value - from chart where precision and recall are displayed against each other:

# In[ ]:


def print_recalls_precision(recalls, precisions, title):
    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision vs Recall plot - {0}".format(title), fontsize=16)
    plt.axis([0,1,0,1])
    plt.show()
print_recalls_precision(recalls, precisions, "stochastic gradient descend")


# # Using different classifier
# Let's use RandomForestClassifier and compare it with SGDClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_probas_forest contains 2 columns, one per class. Each row's sum of probabilities is equal to 1
y_scores_forest = y_probas_forest[:,1]

precisions_forest, recalls_forest, thresholds = precision_recall_curve(y_train_5, y_scores_forest)

print_recalls_precision(recalls_forest, precisions_forest, "random forest classifier")


# RandomForestClassifier performs clearly better.
# 
# Now let's see all 3 classifiers in one chart - dumb classifier, sgd and random forest. For that, we need to compute precision and recall for "never 5" dumb classifier.

# In[ ]:


never_5_predictions = cross_val_predict(never_5_clf, X_train, y_train_5, cv=3)
precisions_dumb, recalls_dumb, thresholds = precision_recall_curve(y_train_5, never_5_predictions)
print_recalls_precision(recalls_dumb, precisions_dumb, "dumb classifier")


# In[ ]:


plt.figure(figsize=(8,6))
plt.plot(precisions_forest, recalls_forest, "r-", label="Random Forest")
plt.plot(precisions, recalls, "g-", label="SGD classifier")
plt.plot(recalls_dumb, precisions_dumb, "b-", label="Dumb classifier")
plt.plot([0, 1], [1,0], "k--", label="Random guess")
plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.title("Precision vs Recall - model comparison", fontsize=16)
plt.axis([0,1,0,1])
plt.legend(loc="center left")
plt.ylim([0, 1])


# In[ ]:


print("F1 score for dumb classifier: {0}".format(f1_score(y_train_5, never_5_predictions)))
print("F1 score for SGD classifier: {0}".format(f1_score(y_train_5, y_train_pred)))
print("F1 score for Random Forest: {0}".format(f1_score(y_train_5, y_scores_forest > 0.5)))


# # Conclusions
# * Random forest worked the best out of 3 tested binary classifiers.
# * Dumb classifier had F1 score of 0 due to 0 predicted samples (0 samples were classified as "5")
# * Classifier that works best has the largest **area under the curve** on Precision / Recall chart
# * Straight line on precision vs recall chart indicates a random classifier that has 50/50% chanse of choosing the right class.
# * F1 score gives more usefull  information about classifier accuracy than calculating the ratio of good / bad predictions

# In[ ]:




