import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
file = pd.read_csv("../input/heart.csv")
features = file.iloc[:,:-1]
labels = file.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier(criterion="gini",min_samples_split = 5)
clf = clf.fit(X_train, y_train)
predict_tree = clf.predict(X_test)

print(accuracy_score(y_test, predict_tree))
