import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv").drop("id",axis=1)
X_train, y_train = train.drop("target",axis=1), train["target"]
test = pd.read_csv("../input/test.csv")
X_test = test.drop("id",axis=1)

clf = LogisticRegression()
clf.fit(X_train,y_train)
test["target"] = clf.predict_proba(X_test)[:,1]

test[["id","target"]].to_csv("baseline.csv",index=False)