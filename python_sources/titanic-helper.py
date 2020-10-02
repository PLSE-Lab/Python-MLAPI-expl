# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:34:41 2019
"""


import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def create_dev_set(X_train, Y_train):
    ## split 889 into 869 and 20 (0.022) (711, 178 for 0.2)
    return train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)

def build_knn(init=True):
    clf = KNeighborsClassifier()
    param_grid = {}
    if init == False:
        param_grid = {"n_neighbors": range(2, 60),
                      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                      "leaf_size":range(2, 50)}
    return clf, param_grid

def build_gnb(init=True):
    clf = GaussianNB()
    param_grid = {}
    if init == False:
        param_grid = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7, 1e9]}
    return clf, param_grid

def build_decision_tree(init=True):
    clf = DecisionTreeClassifier()
    param_grid = {}
    if init == False:
        param_grid = {"max_leaf_nodes": [2, 5, 10, 20]}
    return clf, param_grid

def build_random_forest(seed,init=True):
    clf = RandomForestClassifier(n_estimators=10, random_state=seed)
    param_grid = {}
    if init == False:
        param_grid = {"n_estimators": [2, 5, 10, 25, 50, 100, 250, 500, 1000],
                      "max_leaf_nodes": [2, 5, 10, 20, 40, 50, 100]}
    return clf, param_grid

def build_svm(init=True):
    clf = SVC(gamma="auto")
    param_grid = {}
    if init == False:
#        param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#                      "kernel": ["rbf", "linear", "sigmoid", "poly"],
#                      "gamma": ["auto", "scale"]}
        param_grid = {"C":[0.4, 0.5, 0.6],
                      "kernel": ["rbf"],
                      "gamma": ["scale"]}
    return clf, param_grid

def build_mlp(init=True):
    clf = MLPClassifier(solver='lbfgs', random_state=1)
    param_grid = {}
    if init == False:
        param_grid = {"hidden_layer_sizes": [(1,), (5,), (10,), (20,), (50,), (100,)],
                      "alpha":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
                      "activation": ["identity", "logistic", "tanh", "relu"]}
    return clf, param_grid

def build_log_reg(init=True):
    clf = LogisticRegression(solver="liblinear")
    param_grid = {}
    if init == False:
        param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                      "solver":["liblinear", "lbfgs", "newton-cg", "sag", "saga"]}
    return clf, param_grid

def initial_run(clf, X_train, Y_train):
    t0 = time.time()
    clf.fit(X_train, Y_train)
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    t1 = time.time()
    return [round(np.mean(scores), 4), round(1e4*np.std(scores)**2, 3), round(1000*(t1-t0))]

def find_best(clf, param_grid, X_train, Y_train):
    kfold = KFold(n_splits=5,random_state=7)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_
#    print (grid.best_score_*100, "%")

def find_best_svm(X_train, Y_train):
    clf, param_grid = build_svm(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_log_reg(X_train, Y_train):
    clf, param_grid = build_log_reg(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_knn(X_train, Y_train):
    clf, param_grid = build_knn(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_rfc(X_train, Y_train, seed):
    clf, param_grid = build_random_forest(seed, init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_mlp(X_train, Y_train):
    clf, param_grid = build_mlp(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf


def best_decision_tree():
    clf = DecisionTreeClassifier(max_leaf_nodes=5)
    return clf

def best_svm():
#    clf = SVC(C=0.5, kernel = "poly", gamma="auto")
    clf = SVC(C=1, kernel = "rbf", gamma="auto")
    return clf

def best_log_reg():
    clf = LogisticRegression(solver='lbfgs', random_state=None, C=1)
    return clf

def best_knn():
    clf = KNeighborsClassifier(leaf_size=3, n_neighbors=16, algorithm="auto")
    return clf

def best_rfc():
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=50,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=2, verbose=0, warm_start=False)
    return clf


def best_cvs(clf, X_train, Y_train):
    t0 = time.time()
    clf.fit(X_train, Y_train)
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    t1 = time.time()
    return [round(np.mean(scores), 4), round(1e4*np.std(scores)**2, 3), round(1000*(t1-t0))]


def find_accuracy(clf, X_test, Y_test):
    t0 = time.time()
    predicted = clf.predict(X_test)
    t1 = time.time()
    return(round(accuracy_score(predicted, Y_test),4), round(1e6*(t1-t0)))
    
    
def print_misclassified(clf, X_test, Y_test):
    predicted = clf.predict(X_test)
    diff = predicted - Y_test
    misclass_indexes = np.where(diff != 0)
    print (misclass_indexes[0])


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)