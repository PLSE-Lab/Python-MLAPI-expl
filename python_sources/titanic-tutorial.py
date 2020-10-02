#!/usr/bin/env python
# coding: utf-8

# # Linear Regression on Titanic Dataset with sklearn

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import math
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Initial impressions
# Start by reading in the three supplied data files into Pandas DataFrames, and print out some basic info for the data in those files.

# In[ ]:


trainDf = pd.read_csv("../input/train.csv")
print(trainDf.info())


# In[ ]:


testDf = pd.read_csv("../input/test.csv")
print(testDf.info())


# In[ ]:


gender_submDf = pd.read_csv("../input/gender_submission.csv")
print(gender_submDf.info())

There are two columns that have missing data. The "cabin" column is missing most of its data, so I'll drop it later.
# In[ ]:


print("Missing data in 'Cabin': {:.2f}%".format(trainDf["Cabin"].isnull().sum()/len(trainDf.index)*100.0))
print("Missing data in 'Age': {:.2f}%".format(trainDf["Age"].isnull().sum()/len(trainDf.index)*100.0))


# ## Cleaning/preprocessing the data
# 
# The columns 'Sex', and 'Embarked' contain categorical values, convert to numerical values. Then drop the original columns.
# 
# Furthermore, the 'Name' and the 'Ticket' columns are (probably) not good predictors.

# In[ ]:


trainDf.drop(['Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


trainDf["IsFemale"] = trainDf["Sex"].astype("category").cat.codes
trainDf.drop("Sex", axis=1, inplace=True)
trainDf["EmbarkPort"] = trainDf["Embarked"].astype("category").cat.codes
trainDf.drop("Embarked", axis=1, inplace=True)
trainDf['CabinNo'] = trainDf["Cabin"].astype("category").cat.codes
trainDf.drop("Cabin", axis=1, inplace=True)
trainDf.head()


# Let's get an initial impression of the data. Separate out the survived and perished passengers into individual data sets, and plot histograms of the different categories.

# In[ ]:


cols = trainDf.columns.tolist()
cols.remove('Survived')
print(cols)
survived=trainDf[trainDf.Survived == 1]
survived.hist(figsize=(10,5), layout=(2,5))
died=trainDf[trainDf.Survived == 0]
died.hist(figsize=(10, 5), layout=(2,5))


# The histograms give me a first guess as to what might be important factors to consider in whether someone survived or not. It looks as though the columns 'EmbarkPort', 'isFemale' (Sex), 'Pclass' might be quite predictive, while the columns 'Parch', and 'SibSp' might have secondary predictive information. So I will proceed with more predictive attributes first.

# It is apparent that the 'Cabin' column is missing a lot of (i.e the majority) data. This is a bit of a bummer, since the cabin might be taken a proxy for describing the proximity to the life rafts, and thus could influence the chance of survival. Instead of trying to make up data, which is dangerous in general and stupid in this case. remove this column from the analysis.
# 
# Perhaps later I could use the subset of training/test data that has this information to give a better prediction for the subset only.
# 
# Also, the 'Name', and 'Ticket' columns are assumed to be unrelated to survival chance.

# In[ ]:


trainDf.drop("CabinNo", axis=1, inplace=True)


# At this point, I'm not sure what to do with the 'Age' column yet. ~20% is missing from the training data, which I find to be a lot. Other tutorials that I've consulted just fill the missing data with the median age without justification. I'm not comfortable with doing what everyone else does without fully understanding it (or knowing they know what they're doing). I'm just going to leave it here and decide later.
# 

# ## Alright, let's do some fitting/training!
# Randomly split the training data set into a training set (~60% of the samples) and a cross-validation set (~40% of the samples) so I can guage the training accuracy.

# In[ ]:


def split_training_test_set(trainDf):
    # calculate number of samples in training set
    m = len(trainDf.index)
    # create array of indices and shuffle those
    Idx = np.arange(0, m)
    np.random.shuffle(Idx) 
    # select the first ~60% for the training data, and the rest for the cross-validation data
    m_train = math.ceil(m * 0.6)
    m_cv = m - m_train
    #print(m, m_train, m_cv)
    training = trainDf.iloc[Idx[0:m_train]]
    cv_set = trainDf.iloc[Idx[m_train:]]
    
    return [training, cv_set]


# In[ ]:


[training, cv_set] = split_training_test_set(trainDf)


# Let's train on the data without the Age column, since some values are missing.

# In[ ]:


training1 = training.drop(['Age'], axis=1)
print(training1.columns)
cv_set1 = cv_set.drop(['Age'], axis=1)
print(cv_set1.columns)


# Set up a logistic regression model (since the prediction is either 0 (perished) or 1 (survived)) from the sklearn library.
# Be sure to only pass in the columns the model is supposed to be trained on (i.e. not the prediction column "Survived", not the "PassengerId").

# In[ ]:


from sklearn.linear_model import LogisticRegression
LogReg1 = LogisticRegression(penalty='l2')
## 'Age' contains missing values, skip for now and see how good we get without it.
LogReg1.fit(training1.drop(['Survived', 'PassengerId'], axis=1), training1.Survived)


# In[ ]:


## Use the built-in 'score' method to calculate the accuracy of this model on the training and the cross-validation set. 
## 'score' gives the same answer as calculating the percentage of correctly predicted samples out of the set.
train_score = LogReg1.score(training1.drop(['Survived', 'PassengerId'], axis=1), training1.Survived)
cv_score = LogReg1.score(cv_set1.drop(['Survived', 'PassengerId'], axis=1), cv_set1.Survived)


# In[ ]:


print("Correctness on the training samples: {:g}".format(train_score))
print("Correctly predicted CV samples: {:g}".format(cv_score))
print("Baseline prediction correctness (all females survived): {:g}".format(len(cv_set1[cv_set1.IsFemale == 1])/len(cv_set1)))


# I'm doing quite a bit better than the baseline prediction, which is the accuracy of assuming that all females survived. That's good!
# The accuracy of the model on the training set is naturally a bit higher than on the CV set, but not terribly so. So there's probably not a high 
# bias in this model - I'm not underfitting the data.
# Now try including the Age column in the fitting process, but only include the rows that contain an age.

# In[ ]:


## Remove the rows with missing values
trainDfAge = trainDf.dropna(axis=0, inplace=False)
[trainingAge, cv_setAge] = split_training_test_set(trainDfAge)
LogRegAge = LogisticRegression(penalty='l2')
LogRegAge.fit(trainingAge.drop(['Survived', 'PassengerId'], axis=1), trainingAge.Survived)
## It looks like this gives a notion on how well the learned logistic regression predicts the training data
train_scoreAge = LogRegAge.score(trainingAge.drop(['Survived', 'PassengerId'], axis=1), trainingAge.Survived)
cv_scoreAge = LogRegAge.score(cv_setAge.drop(['Survived', 'PassengerId'], axis=1), cv_setAge.Survived)
all_female_score = len(cv_setAge[cv_setAge.IsFemale == 1])/len(cv_setAge)
print("Accuracy on training set: {:g}".format(train_scoreAge))
print("Accuracy on cv set: {:g}".format(cv_scoreAge))
print("Accuracy of baseline prediction: {:g}".format(all_female_score))


# This is about the same as without the age (depending a bit on which randomly selected samples end up in each training and CV set), so the Age column doesn't seem to be a predictor for survival. Go on without it.
# 
# Apply the same manipulations/transformations as to the training data set, so that I can start making some predictions.

# In[ ]:


#testDf.drop("Cabin", axis=1, inplace=True)
testDf["IsFemale"] = testDf["Sex"].astype("category").cat.codes
testDf["EmbarkPort"] = testDf["Embarked"].astype("category").cat.codes
testDf.drop("Sex", axis=1, inplace=True)
testDf.drop("Embarked", axis=1, inplace=True)
testDf.drop(['Name', 'Ticket', 'Age', 'Cabin'], axis=1, inplace=True)
testDf.info()
testDf.head()


# There is one row in the 'Fare' column that's missing its value!
# I decided to try to fill in the missing value with an educated guess. The 'Fare' column shows a Poissonian- like distribution, so I take the mode of that 
# distribution as my educated guess.

# In[ ]:


modeFare = testDf.Fare.mode()
#print(modeFare.iloc[0])
isNull = pd.isnull(testDf["Fare"])
testDf[isNull]
testDf["Fare"].fillna(modeFare.iloc[0], inplace=True)
testDf.info()
testDf.iloc[152]


# ## Now I'm set to make a prediction!

# In[ ]:


testDf1 = testDf.drop('PassengerId', axis=1, inplace=False)
print(testDf1.columns)
Predict1 = LogReg1.predict(testDf1)


# And, stuff it all into a new DataFrame, so it's easy to submit later.

# In[ ]:


answer1 = pd.DataFrame(Predict1, columns=["Survived"])
answer1["PassengerId"] = 0
answer1.PassengerId = testDf.PassengerId
answer1.info()
answer1.head()


# Now I can select the two required columns for submission and write to a file to submit it.

# In[ ]:


answer1[['PassengerId', 'Survived']].astype('int32').to_csv("LogReg_output.csv", index=False)


# Let's try a decision tree classifier

# In[ ]:


from sklearn import tree

DecTree = tree.DecisionTreeClassifier()
DecTreeModel = DecTree.fit(training1.drop(['Survived', 'PassengerId'], axis=1), training1.Survived)
cv_score = DecTreeModel.score(cv_set1.drop(['Survived', 'PassengerId'], axis=1), cv_set1.Survived)
print(cv_score)

