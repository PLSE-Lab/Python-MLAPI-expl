import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

train['Sex'] = train['Sex'].replace('male', 1).replace('female', 0)

train = train[np.isfinite(train['Age'])]

from sklearn.tree import DecisionTreeClassifier
X = train.filter(['Pclass', 'Sex', 'Age', 'Fare'])
y = train[['Survived']]
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances=clf.feature_importances_

print(importances)