# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:07:10 2017

@author: Jean Piere Rukundo
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Data sets
TRAINING = "../input/train.csv"
TEST = "../input/test.csv"


# Load datasets.
training_set = pd.read_csv(TRAINING, sep=',',header=0)
prediction_set = pd.read_csv(TEST, sep=',',header=0)

# Clean up datasets
training_set["Sex"] = training_set["Sex"].astype('category').cat.codes
training_set["Embarked"] = training_set["Embarked"].astype('category').cat.codes
prediction_set["Sex"] = prediction_set["Sex"].astype('category').cat.codes
prediction_set["Embarked"] = prediction_set["Embarked"].astype('category').cat.codes
 
training_set=training_set.interpolate()
prediction_set=prediction_set.interpolate()
 
# Define the training inputs
x = training_set[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
y = training_set.Survived.values

# Build MLPClassifier Clasiffier.
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# Fit model.
classifier.fit(x,y)

# Classify new samples.
new_samples = prediction_set[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
pred = classifier.predict(new_samples).reshape((418,1))
ident = prediction_set.PassengerId.values.reshape((1,418))
identPred = np.concatenate((ident.T, pred), axis=1)
identPredDf = pd.DataFrame(identPred, columns=['PassengerId', 'Survived'])
identPredDf.to_csv('PassengerId Survived Py 2.csv', index=False)
print("Predictions in ident label: \n{}".format(identPredDf))


