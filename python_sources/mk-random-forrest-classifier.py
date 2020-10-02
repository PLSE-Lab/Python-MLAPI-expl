
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan, precision=1, nanstr="X")

#data_dir = "/home/mkurban/host/datasets/titanic/"
#data_train = pd.read_csv(data_dir + "train.csv")
#data_test = pd.read_csv(data_dir + "test.csv")

data_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
data_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

"""
print("Train Shape:")
print(data_train.shape)
print("Train Columns:")
print(data_train.keys().values)
print("Top of the training data:")
print(data_train.head())
print("Summary statistics of training data")
print(data_train.describe())
"""

# Convenience definition
driving_labels, driven_label,useless_labels = \
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked'], \
    ['Survived'], \
    ['PassengerId'] #,'Name','Ticket','Cabin'

# Split of the training data
X_train, y_train = \
    data_train.loc[:, driving_labels], \
    data_train.loc[:, driven_label]

# Split of the test data
X_test, results = \
    data_test.loc[:, driving_labels], \
    data_test.loc[:, useless_labels]

# Convenience definition
print("Xtrain's Head:", X_train.head())

# Better so than with the get_dummies aux call
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
X_train["Sex"] = lb.fit_transform(X_train["Sex"])
X_test["Sex"] = lb.transform(X_test["Sex"])

print("Xtrain's Head:", X_train.head())

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

print("Xtrain's Head:", X_train.head())

age_map = lambda a : a if not math.isnan(a) else 0.0
X_train["Age"] = X_train["Age"].map(age_map) # Needed here
X_test["Age"] = X_test["Age"].map(age_map) # Needed here?
X_train["Fare"] = X_train["Fare"].map(age_map)# NOT Needed here
X_test["Fare"] = X_test["Fare"].map(age_map) # Needed here

print("Xtrain's Head:", X_train.head())

labels= X_train.columns
print("Labels:", labels)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=19, random_state=1, n_jobs=2)
rfc.fit(X_train, y_train)
importances = rfc.feature_importances_
print("Importances:", importances)
indices = np.argsort(importances)[::-1]
print("Indices:", indices)
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, labels[indices[f]], importances[indices[f]]))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           labels[indices],
           rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

y_test = rfc.predict(X_test)

results["Survived"] = y_test

results.to_csv('results_rfc.csv', index=False)

print(results.head())
