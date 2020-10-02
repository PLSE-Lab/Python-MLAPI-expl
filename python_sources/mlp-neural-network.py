import pandas as pd
import numpy as np
#from sknn.mlp import Classifier, Layer
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score
from sknn.mlp import Classifier, Layer

train = pd.read_csv("../input/train.csv")


X_tr = train.values[:, 1:].astype(float)/255
y_tr = train.values[:, 0]

n_trees = 80
recognizer = MLPClassifier(hidden_layer_sizes=(1, ), learning_rate='adaptive')
score = cross_val_score(recognizer, X_tr, y_tr)
score = np.mean(score)
print(score)