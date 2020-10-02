import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.read_csv('../input/creditcard.csv')
data = data.drop(['Time'] , axis=1)

outliers = data.loc[data['Class']==1]
normal = data.loc[data['Class']==0]

outliers = outliers.drop(['Class'] , axis=1)
normal = normal.drop(['Class'] , axis=1)

X_train = np.array(normal.iloc[0:142403,:])
X_dev = np.array(normal.iloc[142403:,:])
X_test = np.array(outliers)


clf = IsolationForest(max_samples=100)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_dev = clf.predict(X_dev)
y_pred_test = clf.predict(X_test)

print("Accuracy dev :", list(y_pred_dev).count(1)/y_pred_dev.shape[0])
print("Accuracy test:", list(y_pred_test).count(-1)/y_pred_test.shape[0])

