import numpy as np
import pandas as pd

dataset = pd.read_csv("../input/data.csv", header = None)

X = dataset.iloc[1:, [2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]].values
y = dataset.iloc[1:, 1].values

from sklearn.preprocessing import LabelEncoder
encode_y = LabelEncoder()
y = encode_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 0, n_estimators = 14)
clf.fit(X_train, y_train)
y_pred_r = clf.predict(X_test)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred_r, normalize = True)

print(ac)