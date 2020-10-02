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

def create_test_set(X, Y):
    ## split 569 into 483 and 86
    return train_test_split(X, Y, test_size = 0.15, random_state = 0)

def create_dev_set(X_train, Y_train):
    ## split 483 into 397 and 86
    return train_test_split(X_train, Y_train, test_size = 0.178, random_state = 0)

def build_knn(init=True):
    clf = KNeighborsClassifier()
    param_grid = {}
    if init == False:
        param_grid = {}
    return clf, param_grid

def build_gnb(init=True):
    clf = GaussianNB()
    param_grid = {}
    if init == False:
        param_grid = {}
    return clf, param_grid

def build_decision_tree(init=True):
    clf = DecisionTreeClassifier()
    param_grid = {}
    if init == False:
        param_grid = {}
    return clf, param_grid

def build_random_forest(init=True):
    clf = RandomForestClassifier(n_estimators=10)
    param_grid = {}
    if init == False:
        param_grid = {}
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
        param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0, 1000.0]}
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
#    print (grid.best_estimator_)
#    print (grid.best_score_*100, "%")

def find_best_svm(X_train, Y_train):
    clf, param_grid = build_svm(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_mlp(X_train, Y_train):
    clf, param_grid = build_mlp(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf

def find_best_log_reg(X_train, Y_train):
    clf, param_grid = build_log_reg(init=False)
    upd_clf = find_best(clf, param_grid, X_train, Y_train)
    return upd_clf


def best_svm():
    clf = SVC(C=10.0, kernel = "rbf", gamma="auto")
    return clf

def best_mlp():
    clf = MLPClassifier(solver='lbfgs', random_state=1, hidden_layer_sizes=(100,), alpha=0.1, activation="relu")
    return clf

def best_log_reg():
    clf = LogisticRegression(C=0.05, solver="liblinear")
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
    
def print_misclassified_svm(X, Y):
    clf = best_svm()
    print_misclassified(clf, X, Y)

def print_misclassified_mlp(X, Y):
    clf = best_mlp()
    print_misclassified(clf, X, Y)
    
def print_misclassified_log_reg(X, Y):
    clf = best_log_reg()
    print_misclassified(clf, X, Y) 