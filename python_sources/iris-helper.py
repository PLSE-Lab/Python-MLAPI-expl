# -*- coding: utf-8 -*-

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

def create_test_set(X, Y):
    ## split 150 into 125 and 25
    return train_test_split(X, Y, test_size = 0.1666, random_state = 0)

def create_dev_set(X_train, Y_train):
    ## split 125 into 100 and 25
    return train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)

def build_knn(init=True):
    clf = KNeighborsClassifier()
    param_grid = {}
    if init == False:
        param_grid = {"n_neighbors": range(2, 10),
                      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}
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

def build_random_forest(init=True):
    clf = RandomForestClassifier(n_estimators=10)
    param_grid = {}
    if init == False:
        param_grid = {"n_estimators": [2, 5, 10, 25, 50, 100, 250, 500, 1000],
                      "max_leaf_nodes": [2, 5, 10, 20]}
    return clf, param_grid

def build_svm(init=True):
    clf = SVC(gamma="auto")
    param_grid = {}
    if init == False:
        param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                      "kernel": ["rbf", "linear", "sigmoid", "poly"],
                      "gamma": ["auto", "scale"]}
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
        param_grid = {}
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

def find_best_decision_tree(X_train, Y_train):
    clf, param_grid = build_decision_tree(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_mlp(X_train, Y_train):
    clf, param_grid = build_mlp(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_gnb(X_train, Y_train):
    clf, param_grid = build_gnb(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf


def best_decision_tree():
    clf = DecisionTreeClassifier(max_leaf_nodes=5)
    return clf

def best_svm():
    clf = SVC(C=100, kernel = "linear", gamma="auto")
    return clf

def best_mlp():
    clf = MLPClassifier(solver='lbfgs', random_state=1, hidden_layer_sizes=(20,), alpha=1e-04, activation="identity")
    return clf

def best_gnb():
    clf = GaussianNB(var_smoothing=1e-09)
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