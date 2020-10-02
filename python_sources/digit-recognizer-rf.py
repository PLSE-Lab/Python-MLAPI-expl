# import pandas as pd

# # The competition datafiles are in the directory ../input
# # Read competition data files:
# train = pd.read_csv("../input/train.csv")
# test  = pd.read_csv("../input/test.csv")

# # Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# # Any files you write to the current directory get shown as outputs
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../input/test.csv")

print("dataset[[0]] has {0[0]} rows and {0[1]} columns".format(dataset[[0]].shape))
# print(dataset[[0]].values.ravel()[0:4])

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)
pred = rf.predict(test)

np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
# print(dataset.head())