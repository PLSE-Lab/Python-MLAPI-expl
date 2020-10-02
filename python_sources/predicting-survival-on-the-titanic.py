#!/usr/bin/env python
# coding: utf-8

# #Predicting Survival on the Titanic
# This is my code for the Starter Competition about survival on the Titanic. 
# Let's start with importing some libraries and removing warnings. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Now let's import the data that we're going to be using.

# In[ ]:


TestFilePath = '../input/test.csv'
TrainFilePath = '../input/train.csv'

testData = pd.read_csv(TestFilePath)
trainData = pd.read_csv(TrainFilePath)

trainData.head(10)


# We can already see a few things in the 10 columns we just got. 
# 
# First of all there seem to be some missing values in the Age & Cabin columns.
# Also the names seem to have not only the first & lastname of a person but also their title (Mr./Mrs./Master./etc.).
# 
# Let's look at some of the data in detail.

# In[ ]:


trainData["Age"] = trainData["Age"].fillna(-0.5)
testData["Age"] = testData["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
trainData['AgeGroup'] = pd.cut(trainData["Age"], bins, labels = labels)
testData['AgeGroup'] = pd.cut(testData["Age"], bins, labels = labels)

fig = plt.figure(figsize=(10,5))

sns.barplot(x="AgeGroup", y="Survived", data=trainData)


# What can we see in this histogram?:
# * Passengers under the age of 15 have a significantly higher chance of surviving though not between the age of 5 and 10.
# * About 3/4 of children under 5 years old survived.
# * While one of the biggest age groups the passengers that are between 20 and 30 also have about a 1/3 chance of surviving.

# In[ ]:


# Create plots about Survival for Sibsp & Parch, Fare, Embarked 

# Barplot for Passenger Class Survival Rate
sns.barplot(x="Pclass", y="Survived", data=trainData)


# In[ ]:


# Barplot for Sex Survival Rate
sns.barplot(x="Sex", y="Survived", data=trainData)


# In[ ]:


# Barplot for Sibling Survival Rate
sns.barplot(x="SibSp", y="Survived", data=trainData)


# In[ ]:


# Barplot for Parent/Children Survival Rate
sns.barplot(x="Parch", y="Survived", data=trainData)


# In[ ]:


# Barplot for Fare Survival Rate
fareHist = trainData[trainData.Fare <= 101]['Fare'].plot.hist(
    figsize=(12, 6),
    color = ['darkgrey'],
    bins = 10,
    fontsize = 16,
    label = 'Total'
)
fareHist.set_title("Fare Survival on the Titanic", fontsize=20)
fareHist = trainData[(trainData.Survived == 1 ) & (trainData.Fare <= 101)]['Fare'].plot.hist(
    figsize=(12, 6),
    color = ['green'],
    bins = 10,
    fontsize = 16,
    label = 'Survivors'
)

fareHist.legend()


# In[ ]:


# Barplot for Embarked Survival Rate
sns.barplot(x="Embarked", y="Survived", data=trainData)


# In[ ]:


trainData.describe()


# In[ ]:


print(pd.isnull(testData).sum())


# In[ ]:


# Fill in Data for NaN values of Age, Cabin (Create new value) and others if needed


# In[ ]:


trainData["Cabin"] = trainData["Cabin"].fillna("None")
testData["Cabin"] = testData["Cabin"].fillna("None")


# In[ ]:


from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


cols_to_use = ['Pclass', 'Sex', 'AgeGroup', 'SibSp', 'Parch', 'Fare', 'Cabin']
X = trainData[cols_to_use]
y = trainData.Survived
predictX = testData[cols_to_use]

# Create Usable Data with OneHotEncoding
one_hot_encoded_predict_X = pd.get_dummies(predictX)
one_hot_encoded_X = pd.get_dummies(X)
final_X, final_predict_X = one_hot_encoded_X.align(one_hot_encoded_predict_X, join='left', axis=1)

# Create pipelines for different models
RFtotalscore, GBtotalscore, LRtotalscore, KNtotalscore, DTtotalscore, XGBtotalscore, SVtotalscore, GPtotalscore, VCtotalscore = 0,0,0,0,0,0,0,0,0
for rs in range(1,10):
    RFclassifier = make_pipeline(SimpleImputer(), RandomForestClassifier(random_state = rs))
    GBclassifier = make_pipeline(SimpleImputer(), GradientBoostingClassifier(random_state = rs))
    LRclassifier = make_pipeline(SimpleImputer(), LogisticRegression(random_state = rs))
    KNclassifier = make_pipeline(SimpleImputer(), KNeighborsClassifier())
    DTclassifier = make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state = rs))
    XGBclassifier = make_pipeline(SimpleImputer(), XGBClassifier(random_state = rs))
    SVclassifier = make_pipeline(SimpleImputer(), SVC(random_state = rs))
    GPclassifier = make_pipeline(SimpleImputer(), GaussianProcessClassifier(random_state = rs))

    VC = make_pipeline(SimpleImputer(), VotingClassifier(estimators=[('rf', RFclassifier), ('gb', GBclassifier), ('XGB', XGBclassifier)]))
    # Calculate Cross Validation for pipelines
    scores = cross_val_score(RFclassifier, final_X, y, scoring='balanced_accuracy')
    RFscore = scores.mean()
    RFtotalscore += RFscore


    scores = cross_val_score(GBclassifier, final_X, y, scoring='balanced_accuracy')
    GBscore = scores.mean()
    GBtotalscore += GBscore


    scores = cross_val_score(LRclassifier, final_X, y, scoring='balanced_accuracy')
    LRscore = scores.mean()
    LRtotalscore += LRscore


    scores = cross_val_score(KNclassifier, final_X, y, scoring='balanced_accuracy')
    KNscore = scores.mean()
    KNtotalscore += KNscore


    scores = cross_val_score(DTclassifier, final_X, y, scoring='balanced_accuracy')
    DTscore = scores.mean()
    DTtotalscore += DTscore


    scores = cross_val_score(XGBclassifier, final_X, y, scoring='balanced_accuracy')
    XGBscore = scores.mean()
    XGBtotalscore += XGBscore


    scores = cross_val_score(SVclassifier, final_X, y, scoring='balanced_accuracy')
    SVscore = scores.mean()
    SVtotalscore += SVscore

    scores = cross_val_score(GPclassifier, final_X, y, scoring='balanced_accuracy')
    GPscore = scores.mean()
    GPtotalscore += GPscore

    scores = cross_val_score(VC, final_X, y, scoring='balanced_accuracy')
    VCscore = scores.mean()
    VCtotalscore += VCscore

# Print Accuracy
print('RF Accuracy:', round((RFtotalscore / rs * 100),2), '%')
print('GB Accuracy:', round((GBtotalscore / rs * 100),2), '%')
print('LR Accuracy:', round((LRtotalscore / rs * 100),2), '%')
print('KN Accuracy:', round((KNtotalscore / rs * 100),2), '%')
print('DT Accuracy:', round((DTtotalscore / rs * 100),2), '%')
print('XGB Accuracy:', round((XGBtotalscore / rs * 100),2), '%')
print('SV Accuracy:', round((SVtotalscore / rs * 100),2), '%')
print('GP Accuracy:', round((GPtotalscore / rs * 100),2), '%')
print('VC Accuracy:', round((VCtotalscore / rs * 100),2), '%')


# In[ ]:


VC.fit(final_X, y)
competitionPredictions = VC.predict(final_predict_X)
output = pd.DataFrame({'PassengerId': testData.PassengerId,
                       'Survived': competitionPredictions})

output.to_csv('submission.csv', index=False)

