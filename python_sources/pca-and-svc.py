import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 0. read and make dataset
train = pd.read_csv('../input/train.csv')
X_train = train.drop(['label'], axis='columns', inplace=False)
y_train = train['label']

from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.30, random_state=4)
# 1. PCA
n_components = 27
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
X_train_pca = pca.transform(X_train)


t0 = time()
'''
# 2-0 GridSearch
params = {'C': [0.1,1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}
clf = GridSearchCV(SVC(), params, verbose=1)
clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print('best estimator : ', clf.best_estimator_)
'''
# 2. clf(SVC)
clf = SVC(C=0.1, kernel='rbf', gamma=0.1)
clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))

# 3. score for test set
print('score ', clf.score(pca.transform(X_ts), y_ts))

# 4. submission
val = pd.read_csv('../input/test.csv')
pred = clf.predict(pca.transform(val))

val['Label'] = pd.Series(pred)
val['ImageId'] = val.index +1
sub = val[['ImageId','Label']]

sub.to_csv('submission.csv', index=False)
