#!/usr/bin/env python
# coding: utf-8

# # TUTORIAL 5: FEATURE SELECTION
# by [Nikola S. Nikolov](http://bdarg.org/niknikolov)
# 
# -----

# This notebook builds on Tutoial 4 by introducing feature selection into the process of selecting the best classifier for a binary classification problem. It also demonstrates how to split a dataset into a training and test sets and use the test set for evaluation.
# 
# The feature selection method applied here is Recursive Feature Elimination (RFE) as demonstrated in the tutorial at https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/.
# 
# In this demonstration we use a modified version of the seeds dataset (see https://archive.ics.uci.edu/ml/datasets/seeds), which is the same dataset used in Tutorial 4.

# # A. Preparation

# ## Import Python Modules

# In[ ]:


import pandas as pd
import numpy as np

from sklearn import preprocessing #needed for scaling attributes to the nterval [0,1]

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split


# ## Load and Prepare the Dataset for Training and Evaluation

# In[ ]:


df = pd.read_csv('../input/seeds-dataset-binary/seeds_dataset_binary.csv')
df.describe()


# In[ ]:


# target attribute
target_attribute_name = 'type'
target = df[target_attribute_name]

# predictor attributes
predictors = df.drop(target_attribute_name, axis=1).values


# Split the dataset into a training (80%) and test (20%) datasets.

# In[ ]:


# pepare independent stratified data sets for training and test of the final model
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.20, shuffle=True, stratify=target)


# Scale all predictor values to the range [0, 1]. Note the target attribute is already binary. This is a useful pre-processing technique to ensure that all attributes are treated equally during training. Applying a scaler (MinMaxScaler) can be seen as another parameter of the ML to be applied. It may or may not improve the accuracy of the trained model, which can be evaluated with a test dataset. 
# 
# Note that the MinMaxScaler is applied separately to the training and the testing datasets. 
# This is to ensure that this transformation when performed on the testing dataset is not influnced by the training dataset.

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
predictors_train = min_max_scaler.fit_transform(predictors_train)
predictors_test = min_max_scaler.fit_transform(predictors_test)


# # B. Feature Selection

# ## Apply RFE with SVM for Selecting the Features

# In[ ]:


# create a base classifier used to evaluate a subset of attributes
estimatorSVM = svm.SVR(kernel="linear")
selectorSVM = RFE(estimatorSVM, 3)
selectorSVM = selectorSVM.fit(predictors_train, target_train)
# summarize the selection of the attributes
print(selectorSVM.support_)
print(selectorSVM.ranking_)


# ## Apply RFE with Logistic Regression for Selecting Features

# In[ ]:


# create a base classifier used to evaluate a subset of attributes
estimatorLR = LogisticRegression(solver='lbfgs')
# create the RFE model and select 3 attributes
selectorLR = RFE(estimatorLR, 3)
selectorLR = selectorLR.fit(predictors_train, target_train)
# summarize the selection of the attributes
print(selectorLR.support_)
print(selectorLR.ranking_)


# ## Evaluate on the Test Dataset

# ### Apply the selectors to prepare a training dataset only with the selected features.
# 
# __Note:__ The same selectors are applied to the test dataset. However, it is important that the test dataset was not used by (it's invisible to) the selectors. 

# In[ ]:


predictors_train_SVMselected = selectorSVM.transform(predictors_train)
predictors_test_SVMselected = selectorSVM.transform(predictors_test)


# In[ ]:


predictors_train_LRselected = selectorLR.transform(predictors_train)
predictors_test_LRselected = selectorLR.transform(predictors_test)


# ### Train and evaluate SVM classifiers with both the selected features and all features 
# 
# Here we train three models:
# * model1 - with the features selected by SVM
# * model2 - with the features selected by Logistic Regression
# * model3 - with all features (i.e. without feature selection)

# In[ ]:


classifier = svm.SVC(gamma='auto')


# In[ ]:


model1 = classifier.fit(predictors_train_SVMselected, target_train)
model1.score(predictors_test_SVMselected, target_test)


# In[ ]:


model2 = classifier.fit(predictors_train_LRselected, target_train)
model2.score(predictors_test_LRselected, target_test)


# In[ ]:


model3 = classifier.fit(predictors_train, target_train)
model3.score(predictors_test, target_test)


# # C. Conclusion and Further Work
# 
# When you execute this code again, it is very likely to get different results.
# 
# To get more accurate results, accounting for the variance in the results, it is better to run the whole experiment multiple times and measure the variance in the results. Then pick the model that gives better results.
# 
# The process outlined in this tutorial can be further authomated with the use of scikit-learn pipelines. As an exercise build at least two pipelines for training classifiers for the seeds dataset. Each pipeline should include a feature-selection method, and the feature-selection method in pipeline 1 should be different from the feature-selection method in pipeline 2.
# 
# To do this follow the examples at:
# 
# * https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/
# * https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976
# * https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/

# [Continue with Tutorial 6: Clustering and Manifold Learning](https://www.kaggle.com/nikniko101v/tutorial-6-clustering-and-manifold-learning)
