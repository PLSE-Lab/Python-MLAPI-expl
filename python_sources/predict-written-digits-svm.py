import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

X_train = train.drop('label', 1)
Y_train = train['label']
X_test = test

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)
lsvc_y_predict = lsvc.predict(X_test)
lsvc_submission = pd.DataFrame({'ImageId':list(range(1, 28001)),'Label':lsvc_y_predict})
lsvc_submission.to_csv('lsvc_submission.csv', index = 'False')