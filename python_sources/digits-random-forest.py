import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
print("1. loading data")
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs


labels = train.values[:,0]
data = train.values[:,1:]

# configure knn
c = RandomForestClassifier(n_estimators=1000, bootstrap=False)

print("2. training classifier")
c.fit(data,labels)
s = c.score(data,labels)
print("score: {0}".format(s))

print("3. make prediction")
pred = c.predict(test.values)

print("4. write file")
header = 'ImageId,Label'
ids = range(1, len(pred)+1)
data = np.array([ids, pred]).transpose()

np.savetxt('submission.csv', data, fmt='%d', delimiter=',', header=header, comments='')