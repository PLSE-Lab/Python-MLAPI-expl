import sys

import numpy as np
import pandas as pd
import colorsys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

np.random.seed(17411)

# The competition datafiles are in the directory ../input
# Read competition data files:
df = pd.read_csv("../input/train.csv")
X = df.values.copy()
np.random.shuffle(X)

# Test training and validations sets
# Note in this case training data has -
# predictors: starting from second column to end
# targets: in first column
# X_train, X_valid, y_train, y_valid = train_test_split(
#    X[1:, 1:], X[1:, 0], train_size=train_size,
#    )

train_size=0.8
X_train, X_valid, y_train, y_valid = train_test_split(
    X[:, 1:], X[:, 0], train_size=train_size,
    )
print(" -- Loaded data.")
print("Training set has {0[0]} rows and {0[1]} columns".format(X.shape))

# Setup classification trainer
# clf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=4)

#pca = PCA(n_components=2)
#X_r = pca.fit(X_train).transform(X_train)

tsne = TSNE(n_components=2)
X_r = tsne.fit_transform(X_train)

# Start training
# print(" -- Start training Random Forest Classifier. Number of trees = "+str(n_estimators))
# clf.fit(X_train, y_train)
# y_prob = clf.predict_proba(X_valid)
# print(" -- Finished training.")

# Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s'
#      % str(pca.explained_variance_ratio_))

plt.figure()
N = 7
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

for c, i, target_name in zip(RGB_tuples, range(0,N), ['0','1','2','3','4','5','6']):
    plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of MNIST dataset')

plt.show()
