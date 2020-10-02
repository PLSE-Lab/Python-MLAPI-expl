# Use random forest method to identify images of digits
# By MHardin on 2016-07-26

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

datadir = '../input/'

# Read in the training and test data
train = pd.read_csv(datadir + 'train.csv', header=0)
test  = pd.read_csv(datadir + 'test.csv', header=0)

# Set up a random forest
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train.values[:, 1:], train.values[:, 0])

# Use forest to predict numbers in test set
predict = forest.predict(test.values)

# Store predictions in DataFrame 'output'
outdict = {'ImageId': np.arange(len(test['pixel0'])) + 1, 'Label': predict}
output = pd.DataFrame(outdict)

output.to_csv('digits_rand_forest.csv', index=False)