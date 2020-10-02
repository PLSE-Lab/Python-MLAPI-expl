import numpy as np
import pandas as pd

import seaborn as sns

from time import time

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import math

train = pd.read_csv('../input/train.csv')

X_train = train.drop(['label'], axis='columns', inplace=False)
y_train = train['label']

from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.30, random_state=4)

# PCA
n_components = 16
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
X_train_pca = pca.transform(X_train)

# clf
t0 = time()
clf = SVC(C=0.1, kernel='rbf', gamma=0.1)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))

# predict
print('score ', clf.score(pca.transform(X_ts), y_ts))

# submission
val = pd.read_csv('../input/test.csv')
pred = clf.predict(pca.transform(val))
# ImageId,Label

val['Label'] = pd.Series(pred)
val['ImageId'] = val.index +1
sub = val[['ImageId','Label']]

sub.to_csv('submission.csv', index=False)