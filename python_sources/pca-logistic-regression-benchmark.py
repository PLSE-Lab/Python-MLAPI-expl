"""
Numerai PCA + logistic regression benchmark.
"""

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss

print("Loading data...")
data_file = '../input/numerai_training_data.csv'
data = pd.read_csv( data_file )
X = data.drop( 'target', axis = 1 )
y = data.target

print( "Creating train and test set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Construct pipeline of PCA (to decorrelate features) and logistic regression
pipe = Pipeline([('pca', PCA()), ('lr', LR())])

# define grid for parameter search
n_components = np.arange(16,19) # number of PCA components
c_s = np.power(10.0, np.arange(-3, 1)) # regularization
grid = dict(pca__n_components=n_components, lr__C=c_s)

print("\nRunning parameter grid search CV...")
kf = KFold(n_splits=5, shuffle=True)
start_time = time.time()
estimator = GridSearchCV(pipe, grid, scoring='neg_log_loss', cv=kf, verbose=1, n_jobs=-1)
estimator.fit(X_train, y_train)

print("\nBest validation logloss: {:.6}".format(-estimator.best_score_))
print("Best parameters: "+str(estimator.best_params_))
print("Elapsed time: %.2s seconds\n" % (time.time() - start_time))

# Predict test outputs
y_pred = estimator.predict_proba(X_test)

print("Logloss on test set: {:.5}\n".format(log_loss(y_test, y_pred)))
