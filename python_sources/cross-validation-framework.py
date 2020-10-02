import numpy as np

import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer

from copy import deepcopy


class Estimator(object):
    
    def get_estimator(self):
        raise NotImplementedError
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError


class ScikitLearnEstimator(Estimator):
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def get_estimator(self):
        return self.estimator
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        self.estimator.fit(x_train, y_train)
    
    def predict(self, x):
        return self.estimator.predict(x)


class ScikitLearnPredictProbaEstimator(ScikitLearnEstimator):
    
    def get_estimator(self):
        return self
    
    def predict(self, x):
        return self.estimator.predict_proba(x)[:, 1]


class GridSearchCvEstimator(Estimator):
    
    def __init__(self, estimator, param_grid, scoring, cv):
        self.grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, verbose=0, n_jobs=-1)
    
    def get_estimator(self):
        return self.grid.best_estimator_
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        self.grid.fit(x_train, y_train)
        print('Best score:', self.grid.best_score_)
        print('Best parameters', self.grid.best_params_)
    
    def predict(self, x):
        return self.grid.best_estimator_.predict(x)


# returns estimator trained on subset of data, it's score and OOF
def fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx):
    # prepare train and validation data
    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]
    # fit estimator
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)
    oof = estimator.predict(x_train_valid)
    score = scoring(y_train_valid, oof)
    print('Score:', score)
    return estimator.get_estimator(), score, oof


# returns list of estimators trained on folds and OOF
def fit(estimator, scoring, x_train, y_train, cv):
    print('Fit')
    oof = np.zeros(x_train.shape[0])
    trained_estimators = []
    for train_idx, valid_idx in cv.split(x_train, y_train):
        # fit estimator using one fold
        e, score, oof_part = fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx)
        trained_estimators.append(deepcopy(e))
        # collect OOF
        oof[valid_idx] = oof_part
    print('Final score:', scoring(y_train, oof))
    return oof, trained_estimators


# returns predictions average from given estimators
def predict(trained_estimators, x):
    print('Predict')
    y = np.zeros(x.shape[0])
    for estimator in trained_estimators:
        y_part = estimator.predict(x)
        y += y_part / len(trained_estimators)
    return y


# returns score
def validate(estimator, scoring, x_train, y_train, cv):
    print('Validate')
    oof = np.zeros(x_train.shape[0])
    for train_idx, valid_idx in cv.split(x_train, y_train):
        # fit estimator using one fold
        _, score, oof_part = fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx)
        # collect OOF
        oof[valid_idx] = oof_part
    score = scoring(y_train, oof)
    print('Final score:', score)
    return score


# roc auc metric robust to one class in y_pred
def robust_roc_auc_score(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

robust_roc_auc = make_scorer(robust_roc_auc_score)


# LightGBM wrapper

import lightgbm as lgb


class LightGBM(Estimator):
    
    def __init__(self, params):
        self.params = params
    
    def get_estimator(self):
        return self
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        lgb_train = lgb.Dataset(data=x_train.astype('float32'), label=y_train.astype('float32'))
        lgb_valid = lgb.Dataset(data=x_valid.astype('float32'), label=y_valid.astype('float32'))
        self.lgb_model = lgb.train(self.params, lgb_train, valid_sets=lgb_valid, verbose_eval=1000)
    
    def predict(self, x):
        return self.lgb_model.predict(x.astype('float32'), num_iteration=self.lgb_model.best_iteration)
