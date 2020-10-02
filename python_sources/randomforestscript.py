import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

labels = train['label']
train = train.drop('label', axis=1)

clf = RandomForestClassifier(n_estimators=25)
clf = clf.fit(train, labels)

results = clf.predict(test)

np.savetxt('results.csv',
           np.c_[range(1, len(test) + 1), results],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')