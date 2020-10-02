import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

trees=[5,10,20,50,100]
scoreList=[]

for i in trees:
    alg = RandomForestClassifier(random_state=1, n_estimators=i, min_samples_split=5, min_samples_leaf=1,n_jobs=2)
    scores = cross_validation.cross_val_score(alg, train.drop("label",axis=1), train["label"], cv=3)
    scoreList.append(np.mean(scores))
    print('The score for ',i,'trees is: ',np.mean(scores))
    
alg.fit(train.drop("label",axis=1), train["label"])
predictions = alg.predict(test)
np.savetxt('digitsPy.csv', np.c_[range(1,len(test)+1),predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs