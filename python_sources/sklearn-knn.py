import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# The competition datafiles are in the directory ../input
# Read competition data files:
dataset = pd.read_csv("../input/train.csv")

label = dataset[[0]].values.ravel()
traindata = dataset.ix[:,1:].values
test  = pd.read_csv("../input/test.csv").values

# Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

# knnClf = KNeighborsClassifier()
# knnClf.fit(traindata,label)
# pred = knnClf.predict(test)

rf = RandomForestClassifier(n_estimators=5)
rf.fit(traindata,label)
pred = rf.predict(test)

np.savetxt('submission_rf.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')