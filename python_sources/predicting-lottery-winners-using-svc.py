#!/usr/bin/env python
# coding: utf-8

# In[97]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[98]:


#Loading the data into Notebook
data = pd.read_csv("../input/POWERBALL.csv")


# In[99]:


"""Two Methods presented
   METHOD 1 IS A SIMPLE SVM CLASSIFIER: This method seeks only to predict if Division 1 winnter or not
   Columns used to predict are Prize Payable, Rollover, Rollover count and Next Estimated Jackpot"""
y = data.iloc[:, 16:17].values
X = data.iloc[:, 24:28].values
y1 = np.where(y>=1, 1, 0)#Converting the y data to binary. 0 represents no winner and 1 represents a winner on division 1.


# In[100]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.25, random_state = 0)


# In[102]:


#Suppressing warnings on use of Standard Scaler: DataConversionWarning. Changing data with input dtype int64 to floar64 by StandardScaler
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[103]:


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', C = 10, random_state = 0)
classifier.fit(X_train, y_train.ravel())


# In[104]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[105]:


#Comparing the Test Data to the predicted data. This is a 100% Match.
#Turn both arrays to pandas DataFrames and concantenate
y_test = pd.DataFrame(y_test)
y_pred = pd.DataFrame(y_pred)
result = pd.concat([y_test, y_pred], axis = 1, sort = False)
result


# In[106]:


# Making the Confusion Matrix for Visualization of the data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score = classifier.score(X_test, y_test)
score


# In[107]:


#Plotting the Confusion Matrix
import matplotlib.pyplot as plt
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[108]:


# Applying k-Fold Cross Validation to check for mean and Variance
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train.ravel(), cv = 10)
Mean = accuracies.mean()#Mean close to 1
Variance = accuracies.std()#Low Variance
print(Variance, Mean)


# In[109]:


"""METHOD 2. Using ROC to calculate the area under the curve.
This method is also used to predict the number of winners in division 1 if indeed
there are winners """

# Binarize the output
"""The Binerizer output helps identify the number of winners"""
from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
y


# In[110]:


#Splitting the Dataset to Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[111]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[112]:


#This is a multi output classifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(SVC(kernel = 'linear', probability = True,
                                 random_state = 0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
y_score


# In[113]:


#Plotting the ROC Curve
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[114]:


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

