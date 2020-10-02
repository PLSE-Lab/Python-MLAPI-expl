#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# Machine Learning Classifiers
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
# Machine Learning Resamplers
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


from collections import Counter
# measure error
from sklearn.metrics import mean_squared_error


# In[ ]:


#set random state
rand_state=42


# # Summary

# In this kernel I want to use different machine learning methods to build a model by:
# 
# 1. Modeling the training data with identity labels to predict identity labels and Predict identity labels that are found in the test set on the rest of the training data as well as the testing data.
# 2. Train on the training data with its new labels and Predict the Testing data

# Before we start, I will import the training and testing set.

# In[ ]:


# obtain training and testing dataframes
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("number of training rows:%i" % len(train_df))
print("number of testing rows:%i "% len(test_df))


# # 1. Train and Predict Identity Labels

# I want to subset the training set to train for each of the identity labels that are found in the testing set.

# In[ ]:


# subset the training dataframe to only include rows with identity labels.
identityAnn_train_df = train_df.loc[train_df["identity_annotator_count"]>0,:]
print(len(identityAnn_train_df))


# In[ ]:


# subset the identityAnn_train_df to only include the id, target, and comment column
# as well as the columns that contain identities that are used in the 
# testing data
identitiesInTestSet=["male","female","homosexual_gay_or_lesbian","christian","jewish","muslim","black","white","psychiatric_or_mental_illness"]
identityAnn_train_df = identityAnn_train_df.loc[:,["id","comment_text", "target"]+identitiesInTestSet]


# Next, I am going to import TFIDF Vectorizers which is used to first convert the comments_text column into a matrix of word counts and then transforms these counts by normalizing them based on the term frequency. This matrix can then be used as by the machine learning algorithm unlike the comments themselves.
# 
# For more information please visit
# 
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# I am now going to loop through the identities that will be found in the testing set, train them on rows that have identities and predict them on rows that do not as well as predict them on the test set.

# First we need to get all the training rows that are not yet labeled for identities

# In[ ]:


# subset the training dataframe to only include rows with identity labels.
notIdentityAnn_train_df = train_df.loc[train_df["identity_annotator_count"]==0,:].copy()
print(len(notIdentityAnn_train_df))


# I also just want to make a copy of the test dataframe for safe measures

# In[ ]:


MultiNB_test_df = test_df.copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for identity_x in identitiesInTestSet:\n  print("Predicting %s..." % identity_x)\n  X_identity_train1 = identityAnn_train_df["comment_text"].copy()\n  X_identity_train2 = notIdentityAnn_train_df["comment_text"].copy()\n  X_identity_test = MultiNB_test_df["comment_text"].copy()\n  y_identity_train = identityAnn_train_df[identity_x].copy()\n\n  # In order to convert the coninuous values of the identity value to binary, as\n  #  naive bayes can accept only binary values (0 or 1) as the target values\n  # Here we choose above 0 as a cutoff as we want to classify identities even if\n  # only one of the people thought it matched that identity\n  y_identity_train_binary = np.array(y_identity_train > 0, dtype=np.float)\n\n  # Fit the comments into a count matrix \n  #  and then into a normalized term-frequency representation\n  identity_tfvect = TfidfVectorizer().fit(X_identity_train1)\n  # Then transform the comments based on the fit\n  X_identity_train_tf1 = identity_tfvect.transform(X_identity_train1)\n  X_identity_train_tf2 = identity_tfvect.transform(X_identity_train2)\n  X_identity_test_tf = identity_tfvect.transform(X_identity_test)\n  \n  # over-sample the toxic comments using SMOTE\n  sm = SMOTE(random_state=rand_state)\n  X_identity_train_tf1_sm, y_identity_train_binary_sm = \\\n      sm.fit_resample(X_identity_train_tf1, y_identity_train_binary)\n  \n  # Fit a Naive Base classifier to the training set\n  identity_clf = MultinomialNB().fit(X_identity_train_tf1_sm, y_identity_train_binary_sm)\n\n  # get predicted values\n  train2_identity_predicted = identity_clf.predict(X_identity_train_tf2)\n  notIdentityAnn_train_df.loc[:,identity_x] = train2_identity_predicted\n  test_identity_predicted = identity_clf.predict(X_identity_test_tf)\n  MultiNB_test_df.loc[:,identity_x] = test_identity_predicted\nMultiNB_train_df = pd.concat([identityAnn_train_df, notIdentityAnn_train_df], ignore_index=False)\nprint("DONE!")')


# # 2. Train and Predict Toxic Labels

# Okay so now that everything has labels lets train on each label individually. I first need to categorize each of the rows into their respective bins. Note that I am fine with putting the same row into multiple bins. Then I will predict based on the comments of each bin individually. Next, I will average the predictions of the different models to get a final target score.

# In[ ]:


# create a list of dataframe where the first dataframe
# contains all the rows with no labels and then the rest
# contain rows  with a specific label from the list
# `identitiesInTestSet`
def binByIdentitiesinTestSet(dfWithAllLabels, identitiesInTestSet, verbose = True):
    # calculate how many labels are given to each row
    dfWithAllLabels.loc[:,"numTestSetIdentLabels"] =         dfWithAllLabels[identitiesInTestSet].sum(axis=1)
    
    # rows with no label
    noTestSetIdentLabel_df =         dfWithAllLabels.loc[dfWithAllLabels["numTestSetIdentLabels"]==0, :].copy()

    # rows with labels
    binnedTrainingDfs=[noTestSetIdentLabel_df]
    for ident in identitiesInTestSet:
        identInTestSet = dfWithAllLabels.loc[dfWithAllLabels[ident]>0,:]
        binnedTrainingDfs.append(identInTestSet)

    if verbose:
        for i in range(0,len(binnedTrainingDfs)):
            if i==0:
                print("no label:%i" % len(binnedTrainingDfs[i]))
            else:
                print("%s:%i" % (identitiesInTestSet[i-1],len(binnedTrainingDfs[i])))
    return(binnedTrainingDfs)


# In[ ]:


trainingRowsWithLabel = MultiNB_train_df[identitiesInTestSet].sum(axis=1) > 0
trainingRowsWithNoLabel = MultiNB_train_df[identitiesInTestSet].sum(axis=1) == 0

print("Number of Rows with a Labels Found in Test Df: %i" %       sum(trainingRowsWithLabel))

print("Number of Rows with No Labels Found in Test Df: %i" %       sum(trainingRowsWithNoLabel))


# In[ ]:


final_train_df = MultiNB_train_df.copy()
final_test_df = MultiNB_test_df.copy()


# In[ ]:


print("Number of Rows in Each Training Bin")
binnedTraining_list = binByIdentitiesinTestSet(final_train_df,identitiesInTestSet)
print("\nNumber of Rows in Each Testing Bin")
binnedTesting_list = binByIdentitiesinTestSet(final_test_df,identitiesInTestSet)


# In[ ]:


for i in range(0,len(binnedTraining_list)):
    identity_x = "No"
    if i > 0:
        identity_x = identitiesInTestSet[i-1]
    print("Predicting Rows with %s Identity Labels..." % identity_x)
    cur_train_df = binnedTraining_list[i]
    cur_test_df = binnedTesting_list[i]
    X_train = binnedTraining_list[i].loc[:,"comment_text"]
    X_test = binnedTesting_list[i].loc[:,"comment_text"]
    y_train = binnedTraining_list[i].loc[:,"target"]
    print("Training Set:")
    print(Counter(y_train))
    # In order to convert the coninuous values of the identity value to binary, as
    #  naive bayes can accept only binary values (0 or 1) as the target values
    # Here we choose above 0 as a cutoff as we want to classify identities even if
    # only one of the people thought it matched that identity
    y_train_binary = np.array(y_train > 0, dtype=np.float)

    # Fit the comments into a count matrix 
    #  and then into a normalized term-frequency representation
    identity_tfvect = TfidfVectorizer().fit(X_train)
    # Then transform the comments based on the fit
    X_train_tf = identity_tfvect.transform(X_train)
    X_test_tf = identity_tfvect.transform(X_test)
    
    # over-sample the toxic comments using SMOTE
    sm = SMOTE(random_state=42)
    X_train_tf_sm, y_train_binary_sm = sm.fit_resample(X_train_tf, y_train_binary)
    
    print("targets in training set after SMOTE:")
    print(Counter(y_train_binary_sm))
    
    # Fit a Naive Base classifier to the training set
    target_clf = MultinomialNB().fit(X_train_tf_sm, y_train_binary_sm)

    # get values
    target_prediction = target_clf.predict(X_test_tf)
    print("Prediction Set:")
    print(Counter(target_prediction))
    prediction_row = ("%s_Ident_Pred" % identity_x)
    cur_test_df.loc[:,prediction_row] = target_prediction
    cur_test_df = cur_test_df.loc[:,["id",prediction_row]]
    final_test_df = final_test_df.merge(cur_test_df, on="id", how="outer")


# In[ ]:


predCols = [x + "_Ident_Pred" for x in ["No"] + identitiesInTestSet]
print(predCols)
final_test_df["prediction"] = final_test_df[predCols].mean(axis=1)

submission_df = final_test_df.loc[:,["id","prediction"]].copy()
submission_df.head()


# In[ ]:


submission_df.to_csv('submission.csv', index = False)

