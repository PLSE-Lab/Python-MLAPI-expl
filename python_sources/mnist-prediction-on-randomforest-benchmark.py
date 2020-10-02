import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

Y_train = train['label']

train = train.drop(['label'],axis = 1)

X_train = train.values

X_test = test.values

clf = RandomForestClassifier()

clf = clf.fit(X_train,Y_train)

pred = clf.predict(X_test)

result = pd.DataFrame(pred)

result.columns = ['Label']

imageid = [i+1 for i in range(28000)]

result['ImageID'] = imageid

result['label'] = result['Label']

result = result.drop(['Label'],axis = 1)

result.to_csv('result.csv',index = False)