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
from imblearn.over_sampling import SMOTE


# In[ ]:


from collections import Counter
# measure error
from sklearn.metrics import mean_squared_error


# # Summary

# In this kernel I want to use different machine learning methods to build a model by:
# 1. Modeling the training data with identity labels to predict identity labels.
# 2. Predict identity labels that are found in the test set on the rest of the training data as well as the testing data.
# 3. Train on the training data with its new labels. 
# 4. Predict the Testing data

# Before we start, I will import the training and testing set.

# In[ ]:


# obtain training and testing dataframes
train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
print("number of training rows:%i" % len(train_df))
print("number of testing rows:%i "% len(test_df))


# # 1. Modeling the training data with identity labels to predict identity labels.

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


# Next, I am going to import TFIDF Vectorizers which is used to first convert the `comments_text` column into a matrix of word counts and then transforms these counts by normalizing them based on the term frequency. This matrix can then be used as by the machine learning algorithm unlike the comments themselves.
# 
# For more information please visit 
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# I am now going to loop through the identities that will be found in the testing set to see which is the best model to predict these identities.
# 1. I will then split the `identityAnn_train_df` into a training and testing set for cross validation using the`train_test_split` module
# 2. Transform the comments into a count matrix to a normalized term-frequency representation
# 3. Run a classifier on the transformed comments matrix
# 4. Obtain the Training Accuract as well as the Testing Accuracy, Precision, and Recall

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
# This dataframe should contain all the info about each model
identity_cross_val_df = pd.DataFrame(columns=["Classifier","Identity","Train_Acc","Test_Acc","Test_Prec","Test_Recall"])
identity_cross_val_frames=[]


# ## Using Multinominal Naive Bayes to Model Identities
# I will test a Naive Bayes classifier for multinomial models with the `MulinomialNB` module

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif os.path.isfile(\'../input/beginner-modeling/identity_cross_val_df.csv\'):\n    identity_cross_val_df = pd.read_csv(\'../input/beginner-modeling/identity_cross_val_df.csv\')\nelse:\n    for identity_x in identitiesInTestSet:\n        print("Predicting %s..." % identity_x)\n        # Split DataFrame for Cross Validation\n        X_identity_train, X_identity_test, y_identity_train, y_identity_test = \\\n            train_test_split(identityAnn_train_df["comment_text"], \\\n                             identityAnn_train_df[identity_x], test_size = .10)\n\n        # In order to convert the coninuous values of the identity value to binary, as\n        #  naive bayes can accept only binary values (0 or 1) as the target values\n        # Here we choose above 0 as a cutoff as we want to classify identities even if\n        # only one of the people thought it matched that identity\n        y_identity_train_binary = np.array(y_identity_train > 0, dtype=np.float)\n        y_identity_test_binary = np.array(y_identity_test > 0, dtype=np.float)\n\n        # Fit the comments into a count matrix \n        #  and then into a normalized term-frequency representation\n        identity_tfvect = TfidfVectorizer().fit(X_identity_train)\n        # Then transform the comments based on the fit\n        X_identity_train_tf = identity_tfvect.transform(X_identity_train)\n        X_identity_test_tf = identity_tfvect.transform(X_identity_test)\n\n        # Fit a Naive Base classifier to the training set\n        identity_clf = MultinomialNB().fit(X_identity_train_tf, y_identity_train_binary)\n\n        # get values\n        train_acc = identity_clf.score(X_identity_train_tf, y_identity_train_binary)\n        test_acc = identity_clf.score(X_identity_test_tf, y_identity_test_binary)\n        identity_predicted = identity_clf.predict(X_identity_test_tf)\n        test_prec = precision_score(y_identity_test_binary, identity_predicted)\n        test_recall = recall_score(y_identity_test_binary, identity_predicted)\n\n        identity_result_df = pd.DataFrame({"Classifier":["MultinomialNB"],\n                                           "Identity":[identity_x],\n                                           "Train_Acc":[train_acc],\n                                           "Test_Acc":[test_acc],\n                                           "Test_Prec":[test_prec],\n                                           "Test_Recall":[test_recall]})\n        identity_cross_val_frames.append(identity_result_df)\n\n\n    identity_cross_val_df = pd.concat(identity_cross_val_frames,ignore_index=False)\n    # Now I am going to output the `cross_val_df` so I never have to run this code again.\n    identity_cross_val_df.to_csv(\'identity_cross_val_df.csv\', index = False)')


# In[ ]:


identity_cross_val_df


# Next I want to try logistic regression, naive bayes, and an svm classifier with a linear kernel...

# # 2. Predict identity labels that are found in the test set on the rest of the training data as well as the testing data.

# First we need to get all the training rows that are not yet labeled for identities

# In[ ]:


# subset the training dataframe to only include rows with identity labels.
notIdentityAnn_train_df = train_df.loc[train_df["identity_annotator_count"]==0,:].copy()
print(len(notIdentityAnn_train_df))


# I also just want to make a copy of the test dataframe for safe measures

# In[ ]:


MultiNB_test_df = test_df.copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predicted_allIdent_train_file = "../input/beginner-modeling/MultiNB_train.csv"\npredicted_allIdent_test_file = "../input/beginner-modeling/MultiNB_test.csv"\n\n\nif os.path.isfile(predicted_allIdent_train_file) and \\\n    os.path.isfile(predicted_allIdent_test_file) and 1==0:\n    MultiNB_train_df = pd.read_csv(predicted_allIdent_train_file)\n    MultiNB_test_df = pd.read_csv(predicted_allIdent_test_file)\n    MultiNB_train_df.loc[:,"target"].astype(np.float64)\nelse:\n    for identity_x in identitiesInTestSet:\n        print("Predicting %s..." % identity_x)\n        X_identity_train1 = identityAnn_train_df["comment_text"].copy()\n        X_identity_train2 = notIdentityAnn_train_df["comment_text"].copy()\n        X_identity_test = MultiNB_test_df["comment_text"].copy()\n        y_identity_train = identityAnn_train_df[identity_x].copy()\n\n        # In order to convert the coninuous values of the identity value to binary, as\n        #  naive bayes can accept only binary values (0 or 1) as the target values\n        # Here we choose above 0 as a cutoff as we want to classify identities even if\n        # only one of the people thought it matched that identity\n        y_identity_train_binary = np.array(y_identity_train > 0, dtype=np.float)\n\n        # Fit the comments into a count matrix \n        #  and then into a normalized term-frequency representation\n        identity_tfvect = TfidfVectorizer().fit(X_identity_train1)\n        # Then transform the comments based on the fit\n        X_identity_train_tf1 = identity_tfvect.transform(X_identity_train1)\n        X_identity_train_tf2 = identity_tfvect.transform(X_identity_train2)\n        X_identity_test_tf = identity_tfvect.transform(X_identity_test)\n\n        # Fit a Naive Base classifier to the training set\n        identity_clf = MultinomialNB().fit(X_identity_train_tf1, y_identity_train_binary)\n\n        # get predicted values\n        train2_identity_predicted = identity_clf.predict(X_identity_train_tf2)\n        notIdentityAnn_train_df.loc[:,identity_x] = train2_identity_predicted\n        test_identity_predicted = identity_clf.predict(X_identity_test_tf)\n        MultiNB_test_df.loc[:,identity_x] = test_identity_predicted\n    MultiNB_train_df = pd.concat([identityAnn_train_df, notIdentityAnn_train_df], ignore_index=False)\n    print("DONE!")')


# In[ ]:


# Now I am going to output these modified testing and training data frames so I never have to run this code again.
MultiNB_train_df.to_csv('MultiNB_train.csv', index = False)
MultiNB_test_df.to_csv('MultiNB_test.csv', index = False)


# Lets just take a peak at our new dataframes...

# In[ ]:


MultiNB_train_df.head()


# In[ ]:


MultiNB_test_df.loc[MultiNB_test_df["muslim"]>0,:].head()


# # 3. Train on the training data with its new labels. 

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


# Since there are a lot more rows that do not have a label found in the test set in them I am going to downsize their rows. This will just help run these tests a bit faster.

# In[ ]:


import random

trainingRowsWithLabel_df = MultiNB_train_df.loc[trainingRowsWithLabel,:]
trainingRowsWithNoLabel_df = MultiNB_train_df.loc[trainingRowsWithNoLabel,:]


downsample_n = 90000
random_rows = random.sample(range(0,len(trainingRowsWithNoLabel_df)), downsample_n)
trainingRowsWithNoLabel_df = trainingRowsWithNoLabel_df.iloc[random_rows,:]

train_downSampled = pd.concat([trainingRowsWithLabel_df, trainingRowsWithNoLabel_df],                               ignore_index=False)
print(len(train_downSampled))


# Next, I want to test a model on each of the different training sets now
# 

# In[ ]:


# This dataframe should contain all the info about each model
target_cross_val_df = pd.DataFrame(columns=["Classifier","Identity","Train_Acc","Test_Acc","Test_Prec","Test_Recall"])
target_cross_val_frames=[]


# ## MultinominalNB on Target

# In[ ]:


good=False
x=1
while good==False:
    print("Round "+ str(x))
    # Split DataFrame for Cross Validation
    X_fullcv_train, X_fullcv_test, y_fullcv_train, y_fullcv_test =             train_test_split(train_downSampled.loc[:,['id',"comment_text","target"]+identitiesInTestSet],                              train_downSampled["target"], test_size = .10)
    # Bin rows by labels
    binnedTrainingCV_list = binByIdentitiesinTestSet(X_fullcv_train, identitiesInTestSet, False)
    binnedTestingCV_list = binByIdentitiesinTestSet(X_fullcv_test, identitiesInTestSet, False)
    
    good=True
    for train_df_x in binnedTrainingCV_list:
        y_cv_train = train_df_x.loc[:,"target"].copy()
        y_cv_train_binary = np.array(y_cv_train > 0, dtype=np.float)
        toxicTrainCount = sum(y_cv_train_binary==1)
        if toxicTrainCount < 6:
            print("Count too low: "+ str(toxicTrainCount))
            good=False
    x+=1
# Bin rows by labels
print("test dataframe sizes:")
binnedTrainingCV_list = binByIdentitiesinTestSet(X_fullcv_train, identitiesInTestSet)
print("\ntraining dataframe sizes:")
binnedTestingCV_list = binByIdentitiesinTestSet(X_fullcv_test, identitiesInTestSet)
        


# In[ ]:


for i in range(0,len(binnedTrainingCV_list)):
    identity_x = "No"
    if i > 0:
        identity_x = identitiesInTestSet[i-1]
    print("Predicting Rows with %s Identity Labels..." % identity_x)
    cur_train_df = binnedTrainingCV_list[i]
    cur_test_df = binnedTestingCV_list[i]
    X_cv_train = binnedTrainingCV_list[i].loc[:,"comment_text"].copy()
    X_cv_test = binnedTestingCV_list[i].loc[:,"comment_text"].copy()
    y_cv_train = binnedTrainingCV_list[i].loc[:,"target"].copy()
    y_cv_test = binnedTestingCV_list[i].loc[:,"target"].copy()

    # In order to convert the coninuous values of the identity value to binary, as
    #  naive bayes can accept only binary values (0 or 1) as the target values
    # Here we choose above 0 as a cutoff as we want to classify identities even if
    # only one of the people thought it matched that identity
    y_cv_train_binary = np.array(y_cv_train > 0, dtype=np.float)
    y_cv_test_binary = np.array(y_cv_test > 0, dtype=np.float)
    print("targets in training set:")
    print(Counter(y_cv_train_binary))
    print("targets in testing set:")
    print(Counter(y_cv_test_binary))

    
    # Fit the comments into a count matrix 
    #  and then into a normalized term-frequency representation
    identity_tfvect = TfidfVectorizer().fit(X_cv_train)
    # Then transform the comments based on the fit
    X_cv_train_tf = identity_tfvect.transform(X_cv_train)
    X_cv_test_tf = identity_tfvect.transform(X_cv_test)

    # over-sample the toxic comments using SMOTE
    sm = SMOTE(random_state=46)
    X_cv_train_tf_sm, y_cv_train_binary_sm = sm.fit_resample(X_cv_train_tf, y_cv_train_binary)
    
    print("targets in training set after SMOTE:")
    print(Counter(y_cv_train_binary_sm))
    
    # Fit a Naive Base classifier to the training set
    identity_clf = MultinomialNB().fit(X_cv_train_tf_sm, y_cv_train_binary_sm)

    # get values
    train_acc = identity_clf.score(X_cv_train_tf_sm, y_cv_train_binary_sm)
    test_acc = identity_clf.score(X_cv_test_tf, y_cv_test_binary)
    identity_predicted = identity_clf.predict(X_cv_test_tf)
    print("prediction counter")
    print(Counter(identity_predicted))
    
    test_prec = precision_score(y_cv_test_binary, identity_predicted)
    test_recall = recall_score(y_cv_test_binary, identity_predicted)
    prediction_row = ("%s_Ident_Pred" % identity_x)
    predicted_df =  cur_test_df.copy()
    predicted_df.loc[:,prediction_row] = identity_predicted
    predicted_df = predicted_df.loc[:,["id",prediction_row]]
    X_fullcv_test = X_fullcv_test.merge(predicted_df, on="id", how="outer")

    cv_result_df = pd.DataFrame({"Classifier":["MultinomialNB"],
                                       "Identity":[identity_x],
                                       "Train_Acc":[train_acc],
                                       "Test_Acc":[test_acc],
                                       "Test_Prec":[test_prec],
                                       "Test_Recall":[test_recall]})
    target_cross_val_frames.append(cv_result_df)


target_cross_val_df = pd.concat(target_cross_val_frames,ignore_index=False)
# Now I am going to output the `target_cross_val_df` so I never have to run this code again.
target_cross_val_df.to_csv('target_cross_val_df.csv', index = False)


# Now I want to merge any rows that fell into multiple categories

# In[ ]:


x=len(train_downSampled.loc[train_downSampled["target"].notnull(),:])
y=len(train_downSampled)
print(x/y)


# In[ ]:


from sklearn.metrics import mean_squared_error

predCols = [x + "_Ident_Pred" for x in ["No"] + identitiesInTestSet]
print(predCols)
cv_predict = X_fullcv_test[predCols].mean(axis=1,skipna=True)
print("prediction")
print(Counter(cv_predict))
print("truth")
print(Counter(y_fullcv_test))
test_prec = mean_squared_error(y_fullcv_test, cv_predict)
test_recall = recall_score(y_fullcv_test, cv_predict)


# # 4. Predict the Testing data

# I first need to categorize each of the rows into their respective bins. Note that I am fine with putting the same row into multiple bins.

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

