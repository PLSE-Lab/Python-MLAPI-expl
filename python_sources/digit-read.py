import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

import numpy as np

train_t = np.transpose(train.as_matrix())

Y_train = train_t[0][:40000]
Y_val = train_t[0][:-2000]

X_train = np.transpose(train_t[1:])[:40000]
X_val = np.transpose(train_t[1:])[:-2000]

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(algorithm='sgd', activation = 'logistic', shuffle = True, alpha=1e-5,batch_size= 1000, learning_rate = 'adaptive', max_iter = 200, hidden_layer_sizes=(10,), random_state=1)
clf.fit(X_train,Y_train)
clf.score(X_val,Y_val)
clf.n_iter_