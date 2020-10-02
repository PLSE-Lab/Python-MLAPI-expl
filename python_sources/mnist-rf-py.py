import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

X_train = train.as_matrix()[:, 1:]
Y_train = train.as_matrix()[:, 0]

X_test = test.as_matrix()

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300, n_jobs=3)
clf.fit(X_train, Y_train)

from sklearn import metrics

print(metrics.classification_report(Y_train, clf.predict(X_train)))

Y_test = pd.DataFrame(dict(ImageId=range(1, X_test.shape[0]+1), Label=clf.predict(X_test)))
Y_test.set_index('ImageId').to_csv('submit.csv')