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

from sklearn.naive_bayes import GaussianNB

# Open data csvs
titanicTrain = pd.read_csv("../input/titanic/cleanTitanicTrain.csv")
titanicTest = pd.read_csv("../input/titanic/cleanTitanicTest.csv")

# Columns used to predict
predictorNames = ["Pclass", 
                  "Sex", 
                  "Age", 
                  "SibSp", 
                  "Parch", 
                  "Fare", 
                  "Embarked"]

# Create the predictors and targets
trainPredictors = titanicTrain[predictorNames]
targetName = ["Survived"]
trainTarget = titanicTrain["Survived"]
testPredictors = titanicTest[predictorNames]

# Randomly sample the train set to match the length of the test set
trainPredictors = trainPredictors.sample(len(testPredictors))
trainTarget = trainTarget.sample(len(testPredictors))

# Create the gaussian naive bayes model
gnb = GaussianNB(var_smoothing = 1)

# Train the model
gnb.fit(trainPredictors,
        trainTarget)

# Predict using the test values
predictedValues = gnb.predict(testPredictors)

# Print the results
print("Number of mislabeled points out of a total %d points: %d"% 
      (titanicTest.shape[0],
      (trainTarget != predictedValues).sum()))
print(pd.crosstab(gnb.predict(testPredictors), 
                  targetName))