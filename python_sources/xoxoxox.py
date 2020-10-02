import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog


# The competition datafiles are in the directory ../input
# Read competition data files:
# train = pd.read_csv("/Users/toby/Dropbox/0000-present/1_python/0_kaggle/1_digits_rec/input/train.csv")
# test  = pd.read_csv("/Users/toby/Dropbox/0000-present/1_python/0_kaggle/1_digits_rec/input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

#   ===========================================================================
#   Split data into x,y and train,test
X1_train = np.array(train)[:,1:]
y1_train = np.array(train)[:,0]
X_train, X_test, y_train, y_test = train_test_split(X1_train, y1_train, test_size=0.2)

h = 28
w = 28


#   ===========================================================================
#   train

# 01
# param = {'C':np.logspace(-5,5,8),
#          'gamma': [0.0001, 0.005, 0.1]}
# clf = GridSearchCV(svm.SVC(), param, n_jobs=8)

# 02
clf = RandomForestClassifier(n_estimators=200,n_jobs=8)

t0 = time()
clf.fit(X_train[:,:], y_train[:])

print('$$$$ Train cost %0.3f seconds...' % (time()-t0) )

# print('$$$$ Best Classifier is :')
# print(clf.best_estimator_)



#   ===========================================================================
#   Test
print('$$$$ Test score is : %.5f.' % clf.score(X_test[:,:], y_test[:]))
print('$$$$ Test score is : %.5f.' % clf.score(X_train[:,:], y_train[:]))



# tmp = clf.predict(X_test[:1000,:])
# ShowRandFig(X_test[:1000,:], y_test[:1000], 16, h, w)





#   ===========================================================================
#   Common function

def ShowRandFig(X, y, num, h, w):
    n_cols = np.ceil(np.sqrt(num))
    n_rows = np.ceil(float(num)/n_cols)
    plt.figure()
    for i in range(num):
        idx = np.random.choice(len(X))
        fig = X[idx,:]
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(fig.reshape(h,w), cmap='gray')
        plt.title(str(y[idx]))










