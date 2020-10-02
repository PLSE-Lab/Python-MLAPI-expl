#!/usr/bin/env python
# coding: utf-8

# # Classification Chapter

# # MNIST dataset

# In[ ]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()


# In[ ]:


X, y = mnist['data'], mnist['target']
X.shape, y.shape


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show


# In[ ]:


import numpy as np

y[0] ## is a string must turn this into number
y = y.astype(np.uint8)
y[0]


# In[ ]:


## creating the train and the test set as always, the MNIST dataset is 
## already split into a training set(the first 60,000 images) and a test
## set (the last 10,000) images.

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# # Training a Binary Classifier

# In[ ]:


## first of all i'll try to identify if the number is a 5
y_train_5 = (y_train == 5) ## labels of the train set, 60,000 in size
y_test_5 = (y_test == 5) ## labels of test set, 10,000 in size
y_train_5, y_test_5


# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[ ]:


sgd_clf.predict([X_train[0]])


# measuring performance of classifiers is trickier than that of regressors

# # Performance Measures
# # Many pages are dedicated to Performance Measures...

# # Measuring acuracy using cross-validation:
# The folowing code does roughly the same thing as the Scikit-Learn's `cross_val_score()` function, and it prints the same result

# In[ ]:


from sklearn.model_selection import StratifiedKFold # performs stratified sampling to produce
from sklearn.base import clone                      # folds that contain a representative ratio
                                                    # of each class
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5): # Generate indices to split 
    clone_clf = clone(sgd_clf) ## using clone()                   # data into training and test set.
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    print('train_index:', train_index)
    print('test_index:', test_index)
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('score: ', n_correct / len(y_pred), '\n-----------') ## outputs the ratio of correct predictions


# In[ ]:


#help(skfolds.split)


# Now using the proper `cross_val_score()` function to evaluate `SGDClassifier`

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# Now, demonstrating why 'accuracy' is not a got performance measure:

# In[ ]:


from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')
## it will output about 90% accuracy, as there are around 90% non fives in the data set


# # Confusion Matrix

# In[ ]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#with this, i can get a prediction for each instance in the training set
y_train_pred


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# A perfect classifier would have only true positives and true negatives:

# In[ ]:


y_train_5_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_5_perfect_predictions)


# ### Precision and Recall:

# In[ ]:


from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_train_5, y_train_pred) ## when it claims it is a 5, the sgd_clf is correct in (precision*100)% of the time
recall = recall_score(y_train_5, y_train_pred) ## (1 - recall) says % how many true 5s were not spot by the classifier
precision, recall                              ## that is it only detects (recall*100)% of the 5s    


# Tying Precision and Recall together, we have the `f1_score`, which is the harmonic mean of both:

# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# In[ ]:


## instead of calling the classifier's predict() method, you can call its decision_function() method,
## which returns a score for each instance, and then use any threshold you want to make predictions
## based on those scores:
y_scores = sgd_clf.decision_function([some_digit])
y_scores


# ### The `SGDClassifier` uses a threshold equal to 0:

# In[ ]:


threshold = 0  #threshold from the tradeoff between Precision and Recall
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# Raising the threshold:

# In[ ]:


threshold = 8000  #threshold from the tradeoff between Precision and Recall
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# This confirms that raising the threshold reduces recall. That is what were previously a True positive is now a false negative. Lowering the threshold would reduce the precision.

# In[ ]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')
#wtih 'decision_function' as the method, i'll get the scores of all instances in the training set
y_scores


# Now i'll use `precision_recall_curve()` function to compute precision and recall for all possible thresholds:

# In[ ]:


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# Precision may sometimes go down when you raise the threshold.

# In[ ]:


plt.plot(recalls, precisions)
plt.xlabel('Recalls')
plt.ylabel('Precisions')
plt.show()


# Now, to show that i can create a classifier that can give me virtually any precision i want, i'll find the lowest threshold that gives me at least 90% **precision**

# In[ ]:


threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
# to make predictions (on the training set for now), instead of calling the classifier's predict() method,
# i'll run this code:
y_train_pred_90 = (y_scores >= threshold_90_precision)


# Now checking these predictions' `precision` and `recall`

# In[ ]:


prec_s = precision_score(y_train_5, y_train_pred_90)
rec_s = recall_score(y_train_5, y_train_pred_90)
prec_s, rec_s


# 90% of precision, exactly what i forced it to be, but the recall is too low. A high precision classifier is not very usefull if its recall is too low.

# # The ROC curve

# In[ ]:


from sklearn.metrics import roc_curve

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_5, y_scores)
fpr, tpr = false_positive_rate, true_positive_rate

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') #Dashed diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')

plot_roc_curve(fpr, tpr)
plt.show()


# A good classifier will be far from the dotted line, towards the top left corner. A good way to measure the performance is to value the area under the curve (AUC). A perfect classifier will have AUC = 1, and a random one would have AUC = 0.5. We can test it with `roc_auc_score`:

# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# ## Now I'll trian a `RandomForestClassifier` and compare its ROC curve and ROC AUC to those of the `SGDClassifier`

# The `RandomForestClassifier` class does not have a `decision_function()` method. Insteat it has a `predict_proba()` method.

# In[ ]:


## The predict_proba() method returns an array containing a row per instance
## and a column per class, each containing the probability that the given 
## instance belongs to the given class

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method='predict_proba')


# The `roc_curve()` function expects labels and scores, but instead of scores I can give it class probabilities:

# In[ ]:


y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# Now ploting the ROC curve

# In[ ]:


plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc='lower right')
plt.show()


# In[ ]:


roc_auc_score(y_train_5, y_scores_forest)


# The Random Forest classifier is superior to the SGD classifier because its ROC curve is much closer to the top-left corner, and it has a greater AUC.

# **LATER I SHOULD TRY SEEING THAT THE FOREST HAS A 99% PRECISION AND 86.6% RECAL !!!**

# Message from the author of the book: "You now know how to train binary classifiers, choose the appropriate metric for your task, evaluate your classifiers using cross-validation, select the precision/recall tradeoff that fits your needs, and use ROC curves and ROC AUC scores to compare various models."

# # Multiclass Classification

# ### Trying a Support Vector Machine classifier:

# In[ ]:


from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
## under the hood scikit-learn is using the OvO strategy: it trained 45 binary classifiers,
## when we call .predicti() it gets their decision scores for the image, and selected the class 
## that won the most duels. 


# In[ ]:


svm_clf.predict([some_digit])


# In[ ]:


some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores


# In[ ]:


svm_clf.classes_


# I can use`OneVsOneClassifier` or `OneVsRestClassifier` classes if I wanna choose which method to use. This code creates a multiclass classifier using the OvR strategy, based on SVC:

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)


# Training a `SGDClassifier` (or a `RandomForestClassifier`) is just as easy:

# In[ ]:


sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# In the cell above Scikit-Learn did not have to run OvR or OvO because SGD classifiers can directly classify instances into multiple classes (as `RandomForestClassifier` can)

# In[ ]:


sgd_clf.decision_function([some_digit])


# ### Evaluating:

# In[ ]:


cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')


# Scaling inputs increases accuracy...

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')


# # Error Analysis
# Analysing the type of errors my model does

# In[ ]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# Image representation of the confusion matrix:

# In[ ]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[ ]:





# In[ ]:




