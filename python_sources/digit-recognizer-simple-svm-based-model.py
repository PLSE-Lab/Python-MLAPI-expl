# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

print("loading train data")
data = pd.read_csv('../input/train.csv', sep=',', index_col=0)
X_tr = data.iloc[:, 1:].values  # iloc ensures X_tr will be a dataframe
y_tr = data.iloc[:, 0].index

# SVC parameters were found based on GridSearchCV with a lot of different combinations.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('SVM', SVC(kernel='poly', C=0.001, gamma=10))
])

print("training")
pipeline.fit(X_tr, y_tr)

print("loading test data")
test_data = (pd.read_csv('../input/test.csv', sep=',', index_col=0))
X = test_data.iloc[:].values

print("predicting")
result = pipeline.predict(X)

print("dumping results")
output = pd.DataFrame({
    'ImageId': (np.arange(result.size) + 1),
    'Label': result
})
output.to_csv('../result.csv', index=False)
print(output)