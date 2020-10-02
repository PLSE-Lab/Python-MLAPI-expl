import pandas as pd
from sklearn import svm

# The competition datafiles are in the directory ../input
# Read competition data files:
#train = pd.read_csv("../input/train.csv")
#test  = pd.read_csv("../input/test.csv")

train_data =[[1,1], [0,0]]
train_target = [1,0]

svc = svm.SVC(kernel = 'linear', gamma = 0.7).fit(train_data, train_target)

print (svc.predict([2., 2.]))
# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs