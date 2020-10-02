import pandas as pd

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

print(train.head())
# X = digits.data
# y = train['label']
# print(y)

# from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier(n_estimators = 100)

# from sklearn import cross_validation
# kfold = cross_validation.KFold(len(X_digits), n_folds=3)

# [forest.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
#     for train, test in kfold]

# print "cross validation score"
# print cross_validation.cross_val_score(forest, X_digits, y_digits, cv=kfold, n_jobs=-1) #[ 0.94991653  0.94991653  0.93656093]

# print forest.predict(X_digits[10]) == y_digits[10] #[ True]

# import pandas as pd

# # The competition datafiles are in the directory ../input
# # Read competition data files:
# train = pd.read_csv("../input/train.csv")
# test  = pd.read_csv("../input/test.csv")

# # Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# print(train.head())
# # Any files you write to the current directory get shown as outputs