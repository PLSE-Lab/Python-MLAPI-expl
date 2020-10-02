#!/usr/bin/env python
# coding: utf-8

# **Load the data into a pandas dataframe**
# 
# I like to load all the required libraries at the start, no real reason, just what I am used to from other programming environments. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn import metrics

import scikitplot as skplt




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
HDdata=pd.read_csv('../input/heart.csv')


# Let's have a look at the data
# 

# In[ ]:


HDdata.head()


# Create the target and features dataframes.  Then split into the train and test sets.

# In[ ]:


y = HDdata['target'].as_matrix()

del HDdata['target']

# We will need this later
featurenames = HDdata.columns

X = HDdata.as_matrix()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# As this is a medical dataset and we would like to know how the classification decisions are being made; ie some "explainability", we will start with a decision tree model.  The aim being to produce a decision tree that can be quickly consulted to provide assistance in determining the risk of heart disease.    At this stage we will leave the data as is, not normalizing or doing any other feature engineering that will make the decision tree less friendly for the end user. 
# 
# Decision trees will overfit the data if we don't set some parameters to keep to the tree complexity under control.   
# 
# We will use grid search to iterate through a bunch of hyperparemeter combinations and output the most effective one.   
# This will take some time if you don't have parralell processing...

# In[ ]:


# instantiate the decision tree classifier
clf = tree.DecisionTreeClassifier()

# Parameters we want to try
param_grid = {
    'max_depth': [4, 3, 5, 6, 7, 8, 9],
    'min_samples_leaf': [3, 5, 7, 9, 2],
    'min_samples_split':[2, 4, 5, 7, 10, 12, 15],
    'max_features': [0.1, 0.01, 0.05, 0.001, 1],
    'criterion': ['gini', 'entropy']
}

GS_clf = GridSearchCV(clf, param_grid, n_jobs=4)

GS_clf.fit(X_train, y_train)

# The best score and the best Params
print('best score: ', GS_clf.best_score_)
print("best params -", GS_clf.best_params_)


# best params - {'criterion': 'entropy', 'max_depth': 6, 'max_features': 0.001, 'min_samples_leaf': 9, 'min_samples_split': 10}
# best score:  0.8349056603773585
# 
# Ok we have the best Params, let's see how the model performs.   It is important compare the test and train accuracy to asses for underfitting and overfitting. 

# In[ ]:


#You could just use GS_clf.fit(X_train, y_train), but i wanted to make sure the params are shown. 
 
clfBestParams = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 6, max_features= 0.001, min_samples_leaf=9, min_samples_split= 10)

clfBestParams.fit(X_train, y_train)

decisionTreePredictionsTrain = clfBestParams.predict(X_train)

decisionTreePredictionsTest = clfBestParams.predict(X_test)

print("Dec Tree accuracy on the training set:", accuracy_score(y_train, decisionTreePredictionsTrain))

print("Dec Tree accuracy on the test set:", accuracy_score(y_test, decisionTreePredictionsTest))


# ~80% for both training
# ~77% for  testing.
# 
# But remeber this is a medical application, we are more interested in precision/specificity and recall/sensitivity than  accuracy. 
# 
# Let's look at the ROC area and create a confusion matrix to see how useful the model is. 

# In[ ]:



print(pd.DataFrame(
    confusion_matrix(y_test, decisionTreePredictionsTest),
    columns=['Predicted No HD', 'Predicted HD'],
    index=['True Not HD', 'True HD']
))




# Ok from a quick glance we can see there are a lot of false negatives; this test is not very sensitive, but seems to be quite precise, ie if the patient does not have the disease it is unlikley they will test positive. 
# 
# Let's work out these scores. 

# In[ ]:


specificity = 36/(36+5)
sensitivity = 34/(34+16)

print("sensitivity", sensitivity)
print("Specificity", specificity)


# sensitivity 0.68
# Specificity 0.8780487804878049
# 
# As expected.  Let's have a look at the ROC curve. 

# In[ ]:


# calculate the fpr and tpr for all thresholds of the classification
probs = clfBestParams.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic Testing')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('(1 - Specificity) False Positive Rate')
plt.show()


# Ok we have a nice model, now let's have a look at how it comes to it's predictions. We shall show the decision tree. 

# In[ ]:


import graphviz
dot_data = tree.export_graphviz(clfBestParams, out_file=None, feature_names= featurenames,
                                class_names= ['0', '1'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("HeartDisease")


# (https://postimg.cc/w3ktn7kC)
# I hope this can be seen. 

# 

# 
