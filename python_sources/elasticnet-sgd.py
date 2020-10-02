import numpy as np # linear algebra
from scipy.stats import pearsonr
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import union_categoricals
from matplotlib import pyplot as plt 
import seaborn as sns

from os import listdir
from os import path

## Much code copied from the Faces example taken from scikit 
## (https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import neighbors

print(__doc__)
## Display progress logs on stdout
logging.basicConfig(level = logging.INFO,
                   format = '%(asctime)s %(message)s')




train_data = pd.read_csv(path.join("..", "input", "learn-together", "train.csv"))
test_data = pd.read_csv(path.join("..", "input", "learn-together", "test.csv"))




X = train_data.drop(['Cover_Type', 'Id'], axis = 1)
X_test = test_data.drop(['Id'], axis = 1)

y = train_data['Cover_Type']
test_Id = test_data.Id
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 124127)




print("Fitting the classifiers to the training set")
t0 = time()
param_grid = {
    'l1_ratio': [0.005, 0.01,  0.1, 0.25, 0.5, 0.75],
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}
clf = GridSearchCV(SGDClassifier(loss = 'log',
                      penalty = 'elasticnet',
                                fit_intercept = False),
                      param_grid,
                      cv = 10,
                      iid = False)
clf = clf.fit(X_train, y_train)

fivenn = neighbors.KNeighborsClassifier(p = 1, n_neighbors = 1)
nnfit = fivenn.fit(X_train, y_train)

ensembled_preds = pd.DataFrame(clf.predict(X_train), nnfit.predict(X_train))

ensemble = svm.SVC(gamma = 'auto')
ensemblefit = ensemble.fit(ensembled_preds, y_train)
print("done in {:3f}s".format(time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



# Quantitative evaluation of the predictions using matplotlib
print("Predicting forest cover on the validation set to work on featurizing.")
t0 = time()
y_pred_SGD = clf.predict(X_valid)
y_pred_nn = fivenn.predict(X_valid)

ensembled_preds_Val = pd.DataFrame(y_pred_SGD, y_pred_nn)

y_pred = ensemble.predict(ensembled_preds_Val)
print("Done in {:3f}s".format(time() - t0))
target_names = np.array(['Spruce/Fir',
                         'Lodgepole Pine',
                         'Ponderosa Pine',
                         'Cottonwood/Willow',
                         'Aspen',
                         'Douglas-fir',
                         'Krummholz'])
n_classes = len(target_names)
print(classification_report(y_valid, y_pred, 
                            target_names = target_names))
print(confusion_matrix(y_valid, y_pred, labels = range(n_classes)))




test_pred_SGD = clf.predict(X_test)
test_pred_nn = fivenn.predict(X_test)

ensembled_preds_test = pd.DataFrame(test_pred_SGD, test_pred_nn)

test_pred = ensemble.predict(ensembled_preds_test)

submission = pd.DataFrame({ 'Id': test_Id,
                            'Cover_Type': test_pred })
submission.to_csv("submission_enet_5nn.csv", index=False)






