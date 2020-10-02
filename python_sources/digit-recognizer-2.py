import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import grid_search
from sklearn import cross_validation
# The competition datafiles are in the directory ../input
# Read competition data files:
df = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(df.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

target = df.ix[:,0]
train = df.drop('label', 1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, test_size=0.2, random_state=1)
parameters = {'min_samples_leaf' :[1, 10, 20, 30], 'max_depth': [None, 1, 10, 50, 100]}
clf = tree.DecisionTreeClassifier(random_state=1)
grid_obj = grid_search.GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)
print(grid_obj.score(X_test, y_test))
pred = grid_obj.predict(test)
np.savetxt('submission_2.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

print("done!")