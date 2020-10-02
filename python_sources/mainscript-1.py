import pandas as pd
import numpy as np

from sklearn import tree

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

import platform
print(platform.processor())

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

X = train.drop('label',axis=1)
Y = train['label']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

