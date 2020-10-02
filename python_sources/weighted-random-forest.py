#!/usr/bin/env python
# coding: utf-8

# Introduction
# ==
# In an earlier pipeline we implemented a multivariate Gaussian approach to anomaly detection [1]. While the data resulting from the principal component analysis is fairly normal distributed this looked promising. 
# 
# While easily reaching an optimal recall, the precision was very low, resulting in many false positives.
# 
# The reason for this approach to anomaly detection is obvious: As the model is only trained on the normal/valid data, the majority class, the skewdness of the data is not the problem. For other supervised learning algorithm this imbalance has to be taken into account. 
# 
# One possible way would be to sample the data to get a more balanced dataset another is the introduction of weighted cost for the classes [2]. We will focus on the second approach as it would be a pity to down-sample the data set. The cost sensitive approach also feels natural in the present case, as the detection of a fraud is of high value to the customer.
# 
# We use the standard sklearn Random Forest algorithm as it has the weighting of classes already bulit-in.
# 
# [1] https://www.kaggle.com/clemensmzr/simple-multivariate-gaussian-anomaly-detection/
# 
# [2] http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

# Data handling
# ==

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')
X = data.drop('Class', 1).values
y = data['Class'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Weighted Random Forest
# ==

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


w = 50 # The weight for the positive class

RF = RandomForestClassifier(class_weight={0: 1, 1: w})


# In[ ]:


RF.fit(X_train, y_train)


# Evaluation
# ==

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


y_pred = RF.predict(X_test)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
recall = tp / (tp + fn)
prec = tp / (tp + fp)
F1 = 2 * recall * prec / (recall + prec)
print(recall, prec, F1)


# In[ ]:


# Some results for different weights (bad implementation, 
# these weights should be chosen agains a validation set)

#w=1 : 0.735632183908 0.888888888889 0.805031446541
#w=10 : 0.701149425287 0.938461538462 0.802631578947
#w=100 : 0.724137931034 0.940298507463 0.818181818182
#w=1000 : 0.701149425287 0.953125 0.807947019868

