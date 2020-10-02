#!/usr/bin/env python
# coding: utf-8

# **Breast Cancer Classification**
# 
# In this project we will try to diagnose breast cancer using **Breast Cancer Wisconsin** data set. the project will include the following:
# 
# 1. reading the data 
# 2. splitting the data in to training and validation set
# 3. choosing the best model to use
# 4. training the model
# 5. making prediction with the model
# 6. calculating the model accuracy and it's confusion matrix
# 7. making type **M** cancer as our concern 
# 8. calculating type **M** cancer accuracy, racall, precision and ROC curve
# 9. making type **B** cancer as our concern 
# 10. calculating type **B** cancer accuracy, racall, precision and ROC curve
# 
# let's get stated
# 
# 
# 

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


# **Importing the libraries to be use**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict


# **Reading the Data**
# 
# The code below read the brest cancer data Into **data** Variable, all the features are in **X** variable and the target in the **y** variable. lastly the **X** and **y** are split in to trainning and testing set.

# In[ ]:


Data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data = Data.drop('Unnamed: 32', axis = 1)
X = data.drop('diagnosis', axis = 1)
y = data.diagnosis

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 123)


# the below code checks if we have any missing value in our training set

# In[ ]:


X_train.isnull().any()


# lets check how our training data is, by displaying its first  rows 10

# In[ ]:


X_train.head(10)


# the code below give us some inside about our training data by providing us with some statistics like **min**, **max**, **count** and so on of each feature in our training set

# In[ ]:


X_train.describe()


# **Building the model**
# 
# it is time to buid our model now, RandomForestClassifier is choosen as the model, the model is fit using  training set (X_train, y_train), make prediction with validation set (X_valid, y_valid) and then compute the prediction accuracy and confusion matrix of the classifier to see how accurate is the classifier on the test data. You can also uncomment the other classifiers to see how they ferform too.

# In[ ]:


model = RandomForestClassifier(random_state = 123, n_estimators = 100)
# model = XGBClassifier(random_state = 123, n_estimators = 100, learning_rate = 0.3)
# model = SGDClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_valid)

acc = accuracy_score(y_valid, pred)
confusion_mat = confusion_matrix(y_valid, pred)
print('Accuracy: ', acc)
print('Confusion_mat:\n ', confusion_mat)

plt.matshow(confusion_mat, cmap=plt.cm.gray)
plt.xlabel('True Values')
plt.ylabel('predicted Values')
plt.title('Confusion Matrix')
plt.show()


# wow! it seems like our classifier did a good job, we got 99% accuracy. And if you look at the confusion matrix you can see that it only miss classify only one patient from its class. 

# **1. Let say Type M cancer is our concern now**
# 
# let's now see how our classifier will classify type M cancer from all our sample data. we will try to find the following:
# 1. Accuracy of the classifier
# 2. Confusion matrix
# 3. precision score
# 4. recall score
# 5. ROC curve of cancer type M

# In[ ]:


y_train_M = (y_train == 'M') # making only type M as our training target
y_valid_M = (y_valid == 'M') # making only type M as our validation target


# In[ ]:


model.fit(X_train, y_train_M)
pred = model.predict(X_valid)

confusion_mat = confusion_matrix(y_valid_M, pred)
precision = precision_score(y_valid_M, pred)
recall = recall_score(y_valid_M, pred)
print('Accuracy: ', acc)
print('Confusion_mat:\n ', confusion_mat)
print('Precision: ', precision)
print('recall: ', recall)

plt.matshow(confusion_mat, cmap=plt.cm.gray)
plt.xlabel('True Values')
plt.ylabel('predicted Values')
plt.title('Confusion Matrix')
plt.show()


# Yess! as we want it, our classifier find all the predictate patient with type **M** cancer (**100%** precision) with ** 98%** recall which shows that our classifier is a good one, which is what we are after in this project. but of course it depend on your project you may say that you want to detect all the patient with the type **M** cancer in that case you may have to makes your recall **100%**.

# In[ ]:


pred_scores_of_M = cross_val_predict(model, X_train, y_train_M, cv= 3, method="predict_proba")
pred_scores_of_M = pred_scores_of_M[:, 1] # score = proba of positive class
# pred_scores


# **cross_val_predict()** function is use to predict the model that will be use to make **ROC** curve graph. Note that we use **predict_proba()** method, it may not be the case in some other classifiers like **SGDClassifier()** which use **decision_function()** method instead of what we now use.

# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_M, pred_scores_of_M)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    
plot_roc_curve(fpr, tpr)
plt.show()


# yess! as we can see from the graph the True **positive rate** is far more than the **false positive rate** which also indicate our classification is good
# 

# **2. Let say Type B cancer is our concern now**
# 
# let's now see how our classifier will classify type B cancer from all our sample data as we did with type M. we will try to find the following:
# 1. Accuracy of the classifier
# 2. Confusion matrix
# 3. precision score
# 4. recall score
# 5. ROC curve of cancer type B

# In[ ]:


y_train_B = (y_train == 'B') # making only type B as our training target
y_valid_B = (y_valid == 'B') # making only type B as our validation target


# In[ ]:


model.fit(X_train, y_train_B)
pred = model.predict(X_valid)

confusion_mat = confusion_matrix(y_valid_B, pred)
precision = precision_score(y_valid_B, pred)
recall = recall_score(y_valid_B, pred)
print('Accuracy: ', acc)
print('Confusion_mat:\n ', confusion_mat)
print('Precision: ', precision)
print('recall: ', recall)

plt.matshow(confusion_mat, cmap=plt.cm.gray)
plt.xlabel('True Values')
plt.ylabel('predicted Values')
plt.title('Confusion Matrix')
plt.show()


# Yess! as we want it, our classifier find almost all the predictate patient with type **B** cancer (**99%** precision) with **100 %** recall which shows that our classifier is a good one, which is what we are after in this project. but of course it depend on your project you may say that you want to detect all all the predicted patient with the type **B** cancer in that case you may have to makes your precision **100%**.

# In[ ]:


pred_scores_of_B = cross_val_predict(model, X_train, y_train_B, cv= 3, method="predict_proba")
pred_scores_of_B = pred_scores_of_B[:, 1] # score = proba of positive class
# pred_scores


# same reason as we did previously in type **M** cancer

# In[ ]:


fpr, tpr, thresholds = roc_curve(y_train_B, pred_scores_of_B)
plot_roc_curve(fpr, tpr)
plt.show()


# This also gives a promissing result as the curve for patient with type **M** cancer that we plot previously.

# **Conclusion**
# 
# cancer dataset is use to buid a model with reletively high accuracy of **99%** using RandomForestClassifier. And also try to predict one class from the other (Binary classification) to see the classifier recall and precisioon scores with their ROC curve graph when one class of the cancer is our priority.
# 
