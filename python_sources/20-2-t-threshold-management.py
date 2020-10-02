#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Load packages and observe dataset

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


data = pd.read_csv('../input/handson-pima/Hands on Exercise Feature Engineering_ pima-indians-diabetes (1).csv')
data.head()


# In[ ]:


#Define certain columns under features X

feature_cols = ['Preg','Plas','Pres','skin','test','mass','age']
X = data[feature_cols]
y = data['class']


# # 2. Train Logistic Regression

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 1)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred_class  = logreg.predict(X_test)


# In[ ]:


#How accurate is this predicted y_test
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


# In[ ]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logreg = confusion_matrix(y_test, y_pred_class)
cm_logreg


# In[ ]:


TP = 40
TN = 110
FN = 29
FP = 13


# In[ ]:


#Lets calculate Classification Accuracy, Misclassification rate, Sensitivity, Specificity, False Positive Rate, Precision

#Classification Accuracy: Overall, how often is the classifier correct?

print((TP + TN)/ float(TP +TN+FP+FN))
print(metrics.accuracy_score(y_test, y_pred_class))


# In[ ]:


#Classification Error: Overall, how often is the classifier incorrect?
#Also known as "Misclassification Rate"

print((FP +FN)/ float(TP +TN+FP+FN))
print(1 - metrics.accuracy_score(y_test, y_pred_class))


# In[ ]:


#Sensitivity: When the actual value is positive, how often is the prediction correct?

#- How "sensitive" is the classifier to detecting positive instances?
#- Also known as "True Positive Rate" or "Recall"

print(TP/ float(TP +FN))
print(metrics.recall_score(y_test, y_pred_class))


# In[ ]:


#**Specificity:** When the actual value is negative, how often is the prediction correct?
#- How "specific" (or "selective") is the classifier in predicting positive instances?

print(TN/ float(TN+FP))


# In[ ]:


#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
print(FP/ float(TN +FP))


# In[ ]:


#Precision: When a positive value is predicted, how often is the prediction correct?

#How "precise" is the classifier when predicting positive instances?


print(TP/ float(TP +FP))
print(metrics.precision_score(y_test, y_pred_class))


# # 3. Adjusting classification threshold

# In[ ]:


#print the first 10 predicted class with default threshold of .5

logreg.predict(X_test)[0:10]


# In[ ]:


# print the first 10 predicted probabilities of class membership

logreg.predict_proba(X_test)[0:10,:]


# In[ ]:


# print the first 10 predicted probabilities for class 1  (diabetics)
logreg.predict_proba(X_test)[0:10,1]


# In[ ]:


# store the predicted probabilities for diabetic class for all records... 
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


# ### Reduce the threshold from .5 to .3 to predict the diabetics class. This will make the model sensitive to diabetic class

# In[ ]:


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]  # deciding the class of the 1st 10 records based on new threshold


# In[ ]:


# print the first 10 predicted probabilities
y_pred_prob[0:10]


# In[ ]:


# print the first 10 predicted classes with the lower threshold. Note the change in class...
# with threshold of .5 (default) , the first data point would belong to 0 class i.e. non-diabetic
y_pred_class[0:10]


# In[ ]:


# previous confusion matrix (default threshold of 0.5)
print(metrics.confusion_matrix(y_test, y_pred_class))


# In[ ]:


# sensitivity has increased (used to be 0.57)
print(98/float(98 + 48))


# In[ ]:


# specificity has increased (used to be 0.91)
print(98/ float(98+72))


# # Observations:
# 
# - Default threshold of .5 is not sensitive towards diabetic class. Lowering the threshold increases the sensitivity to
# - diabetic class

# # 4. ROC Curves and Area Under the Curve (AUC)
# 
# **Question:** Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?
# 
# **Answer:** Plot the ROC curve!

# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# - ROC curve can help you to **choose a threshold** that balances sensitivity and specificity in a way that makes sense for your particular context
# - You can't actually **see the thresholds** used to generate the curve on the ROC curve itself

# In[ ]:


# define a function that accepts a threshold and prints sensitivity and specificity

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[ ]:


evaluate_threshold(0.5)


# In[ ]:


evaluate_threshold(0.3)


# AUC is the **percentage** of the ROC plot that is **underneath the curve**:

# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))


# - AUC is useful as a **single number summary** of classifier performance.
# - If you randomly chose one positive and one negative observation, AUC represents the likelihood that your classifier will assign a **higher predicted probability** to the positive observation.
# - AUC is useful even when there is **high class imbalance** (unlike classification accuracy).

# In[ ]:


#calculate cross validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(logreg,X,y, cv = 10, scoring = 'roc_auc').mean()


# **Confusion matrix advantages:**
# 
# - Allows you to calculate a **variety of metrics**
# - Useful for **multi-class problems** (more than two response classes)
# 
# **ROC/AUC advantages:**
# 
# - Does not require you to **set a classification threshold**
# - Still useful when there is **high class imbalance**
