# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR

x_train = pd.read_csv("../input/train.csv").values
x_test = pd.read_csv("../input/test.csv").values

x_train = np.concatenate((x_train, np.tile(x_train[x_train[:, -1] == 1, :], (24, 1))))
target = x_train[:, -1]
X = x_train[:, 1:-1]
del x_train
x_test_noid = x_test[:, 1:]

pca = PCA(n_components=0.9)
X_new = pca.fit_transform(X)
clf = SVR()
clf.fit(X_new, target)

result = clf.predict(pca.transform(x_test_noid))
result = pd.DataFrame({"ID": x_test[:, 0].astype('int'), "TARGET": result})
result.to_csv('submission_SVR.csv', index=False)