import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
data_train= train.iloc[0:28000,:]
data_test= train.iloc[28000:420001,:]
#print(hand.head(5))

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

alg = RandomForestClassifier(random_state=1,n_estimators=1000)
scores = cross_validation.cross_val_score(alg, data_train.iloc[:,1:786],data_train["label"],cv=3)
print(scores.mean()) 