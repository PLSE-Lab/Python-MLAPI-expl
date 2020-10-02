import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

data = pd.read_csv('../input/covtype.csv')

X = data[data.columns[:-1]]
y = data['Cover_Type']

print("X.shape: ", X.shape)
print("y.shape: ", y.shape)
print("y.value_counts() / y.count(): ", y.value_counts() / y.count())

# Accuracy using train_test_split

def print_tts_accuracy(stratify=None):

    tts_tuple = train_test_split(X, y,
        test_size=0.25, random_state=1, stratify=stratify)
    X_train, X_test, y_train, y_test = tts_tuple

    """
    with_without = "without" if stratify is None else "with"
    print("Data has been split {} stratification.".format(with_without))
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)
    print()
    print("y_test.value_counts() / y.value_counts():",
        y_test.value_counts() / y.value_counts())
    print()
    """

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    tts_accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy =", tts_accuracy)

print_tts_accuracy()
# Accuracy = 0.938521063248
print_tts_accuracy(y)
# Accuracy = 0.937453959643

# Accuracy using cross_val_score

dtree = DecisionTreeClassifier()
res = cross_val_score(dtree, X, y, cv=4, scoring='accuracy')
# print("cross_val scores:", res)
print("mean cross_val score:", res.mean())
# mean cross_val score 0.541924489597
