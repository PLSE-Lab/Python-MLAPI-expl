import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
dataset = pd.read_csv("../input/train.csv")
target=dataset[[0]].values.ravel()
train=dataset.iloc[:,1:].values
test  = pd.read_csv("../input/test.csv").values

rf=RandomForestClassifier(n_estimators=100)
rf.fit(train,target)
pred=rf.predict(test)

np.savetxt('submission.csv',np.c_[range(1,len(test)+1),pred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs