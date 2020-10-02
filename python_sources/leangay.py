import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

dataset = pd.read_csv("../input/train.csv")
dataset = pd.read_csv("../input/train.csv")
Y = dataset[[0]].values.ravel()
X = dataset.iloc[:,25:].values
datatest = pd.read_csv("../input/test.csv")
test = datatest.iloc[:,24:].values

dev_cutoff = len(Y) * 0.9
X_train = X[:dev_cutoff]
Y_train = Y[:dev_cutoff]
X_test = X[dev_cutoff:]
Y_test = Y[dev_cutoff:]



mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=90, alpha=1e-4,
                 algorithm='sgd', verbose=10, tol=1e-4, random_state=1)

mlp.fit(X_train, Y_train)
print(mlp.score(X_test,Y_test))
pred = mlp.predict(test)

#yPred = cnn.predict_classes(testX)

np.savetxt('submission_nnSkLearn.csv', np.c_[range(1,len(test)+1),pred],
delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')  


