import pandas as pd
import numpy as np
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# The competition datafiles are in the directory ../input
# Read competition data files:
data_set = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")

target = data_set[[0]].values.ravel()
train = data_set.iloc[:,1:].values

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train,target)


# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

np.savetxt('RandomForestClassifier.csv', np.c_[range(1,len(test_data)+1),output], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')