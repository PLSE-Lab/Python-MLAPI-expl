import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

X =  np.array(train.values[:, 1:].astype(float))[:5000]
y =  np.array(train.values[:, 0])[:5000]

from sklearn.manifold import Isomap  
iso = Isomap(n_components=20)
data_project = iso.fit_transform(X)

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data_project, y)


from sklearn.ensemble import RandomForestClassifier
isorfc = RandomForestClassifier(n_estimators=100, random_state=0)
isorfc.fit(Xtrain,ytrain)
ypred = isorfc.predict(Xtest)

from sklearn.metrics import confusion_matrix
print (confusion_matrix(ytest, ypred))

from sklearn.metrics import accuracy_score  
print (accuracy_score(ytest, ypred))

