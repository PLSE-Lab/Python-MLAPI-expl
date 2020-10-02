import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('../input/fashion-mnist_train.csv', dtype=int) # read train data
dft = pd.read_csv('../input/fashion-mnist_test.csv', dtype=int) # read test data

X_train = df.drop('label', axis=1)
y_train = df['label']
X_test = dft.drop('label', axis=1)
y_test = dft['label']

# It seems random forest is much faster than MLP and more accurate
model = RandomForestClassifier(n_estimators=64, n_jobs=-1) # 0.8827, 29 seconds
# model = MLPClassifier(max_iter=700) # 0.8557, 190 seconds
model.fit(X_train, y_train.values.ravel())

# Predict
y_pred = model.predict(X_test)

# Print result
print(accuracy_score(y_test, y_pred))
