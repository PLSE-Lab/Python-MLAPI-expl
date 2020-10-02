from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
train = dataset.iloc[:,1:].values
target = dataset[[0]].values.ravel()
test = pd.read_csv("../input/test.csv").values

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)
pred = rf.predict(test)

np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')