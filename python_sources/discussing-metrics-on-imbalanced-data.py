#!/usr/bin/env python
# coding: utf-8

# # Intro

# The notebook contains some simple experiment to evaluate what metric to use on inbalanced classification. The question came up when trying Adversarial Validation on an imbalanced data set.
# Adversiaral Validation can be used to measure how similar a test and train set are and can furthermore be used to build a test-set-like validation set from train.
# 
# This notebooks does some basic tests on some metrics that can be used. Help my figuere out which one is the best (or the best in certain situations).
# 
# Here are some references where I first stumbled upon Adverserial Validation (AV):
# * http://fastml.com/adversarial-validation-part-one/
# * https://www.kaggle.com/tunguz/adversarial-santander
# 
# Here are public kernels relevant to this competition applying AV:
# * https://www.kaggle.com/joatom/a-test-like-validation-set
# * https://www.kaggle.com/lukeimurfather/adversarial-validation-train-vs-test-an-update

# # Setup

# The true data set contains six entries. Two entries of class 1 and four entries of class 2:
# 
# [0, 0, 1, 1, 1, 1]
# 
# We run two test rounds with two false entries each: 
# 1. Simulating a total overfitting to class 1: 
# [1, 1, 1, 1, 1, 1]
# 2. Simulating one missmatch per class:
# [0, 1, 1, 1, 1, 0]
# 

# In[ ]:


from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, classification_report
y_true = [0, 0, 1, 1, 1, 1]
y_pred1 = [1, 1, 1, 1, 1, 1]
y_pred2 = [0, 1, 1, 1, 1, 0]

def eval_preds(y_true, y_pred):
    print('True:',y_true)
    print('Pred:',y_pred)
    print('--------------------------')
    print('accuracy:', accuracy_score(y_true, y_pred))
    print('balcanced_accuracy:', balanced_accuracy_score(y_true, y_pred))
    print('roc_auc:', roc_auc_score(y_true, y_pred))
    print('f1:', f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


# # Test

# ## 1. Simulating overfitting to majority class

# In[ ]:


eval_preds(y_true, y_pred1)


# ## 2. Simulating one missmatch per class

# In[ ]:


eval_preds(y_true, y_pred2)


# # Conclussion

# Both predictions contain two mismatches. 
# In the first test the minority class contains only mismatches (0% match). The predictions of the majority classes are all right (100 % match).
# In the second test each class contains one missmatch (50 % match for Class 0 and 75 % match for Class 1).
# 
# Although both predictions have different characteristics **accuracy** has the same value (0.666). It ignors that class 0 is predicted totaly wrong in the first test.
# **balanced_accuracy** averages the accuracy per class. It figuers out that on the second prediction both classes are predicted partialy right. **roc_auc** shows the same results. **f1** states that the first test with the correct classified class 1 is more accurate then the second test.

# What is our opinion of metrics on imbalanced data? Which one do you prefere? Please leave a comment if I got things wrong.
