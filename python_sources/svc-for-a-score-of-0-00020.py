# One of my first steps is to always run a round-robin of classifiers to try to find the 
# best performer using k-fold train-test-splits.
# The support vector classifier shown actually scored a 0.00020 on the public leaderboard.
# Considering the number of generating vector elements influenced to influence the formation
# of glasses in the images, it makes sense that a SVM should be able to classify images
# generated from those vectors to some degree.  
import numpy as np
import pandas as pd
import os

DATADIR = "../input/applications-of-deep-learningwustl-spring-2020/"

print("Loading training data...")
df = pd.read_csv(DATADIR+"train.csv")

X = df.drop(['id','glasses'], axis=1)
y = df['glasses']

X = np.array(X).reshape(-1, 512)
y = np.array(y).reshape(-1)

from sklearn.svm import SVC

print("Initializing support vector classifier...")
classifier = SVC(kernel='poly', degree=2, probability=True, tol=1e-12, verbose=True)
print("Fitting training data to the classifier...")
classifier.fit(X, y)

print("Loading test data...")
df = pd.read_csv(DATADIR+"test.csv")

X = df.drop(['id'], axis=1)
X = np.array(X).reshape(-1, 512)

print("Calculating predictions for submission...")
y = classifier.predict_proba(X)

df['glasses'] = y[:,1]

df[['id','glasses']].to_csv("submission.csv", index=False)
print("Done!")