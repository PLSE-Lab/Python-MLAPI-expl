#!/usr/bin/python3

import gc

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

#from sklearn.pipeline import Pipeline
#from preprocessor import Preprocessor

#from tsetlin_tk import TsetlinMachineClassifier


def hyper_objective(train_X, train_y, nfolds, ncvjobs, ncvrep,
                    #nepochs,
                    #number_of_pos_neg_clauses_per_label,
                    seed,
                    n_jobs,
                    space):
    xgb_kwargs = \
    {
        #'objective': 'binary:logistic',
        'objective': 'rank:pairwise',
        'learning_rate': 0.045,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'max_depth': 7,
        'n_estimators': 10,
        'nthread': n_jobs,
        'n_jobs': n_jobs,
        'random_state': seed,
        'seed': seed,
        #missing': float('nan')
        #'scoring': NegQWKappaScorer
    }
    kwargs = xgb_kwargs

    for k, v in space.items():
        #if k in ['boost_true_positive_feedback', 'number_of_states',
        #         'threshold']:
        #    v = int(v)
        #    pass
        if k in ['n_estimators', 'max_depth', 'min_child_weight', 'num_pairsample']:
            v = int(v)
            pass
        kwargs[k] = v
        pass

    #pre = Preprocessor(nbits=3)
    #clf = Pipeline(steps=[
    #        ('preprocessor', pre),
    #        ('clf', TsetlinMachineClassifier(random_state=seed,
    #                                         clause_output_tile_size=64,
    #                                         number_of_pos_neg_clauses_per_label=number_of_pos_neg_clauses_per_label,
    #                                         number_of_states=16000,
    #                                         boost_true_positive_feedback=1,
    #                                         n_jobs=n_jobs,
    #                                         **kwargs))])

    clf = XGBClassifier(**kwargs)
    #print(clf)

    from sklearn.model_selection import StratifiedKFold

    scores = []

    from sklearn.model_selection import cross_val_score
    for i in range(ncvrep):
        kf = StratifiedKFold(n_splits=nfolds, random_state=seed + i, shuffle=True)
        #scores.extend(cross_val_score(clf, train_X, train_y, cv=kf, n_jobs=ncvjobs, fit_params={'clf__n_iter': nepochs}))
        scores.extend(cross_val_score(clf, train_X, train_y, scoring='roc_auc', cv=kf, n_jobs=ncvjobs))

    score = np.mean(scores)

    print('{' + 'best score: {:.5f},  best params: {}'.format(score, kwargs) + '}')

    del clf
    gc.collect()

    return -score


def evaluate_hyper(train_X, train_y, objective,
                   neval, nfolds, ncvjobs, ncvrep,
                   njobs, seed,
                   #number_of_pos_neg_clauses_per_label,
                   #nepochs,
                   #states_range, threshold_range, s_range
                   ):
    from hyperopt import fmin, tpe, hp

    #states_min, states_max, states_step = map(int, states_range.split(','))
    #threshold_min, threshold_max, threshold_step = map(int, threshold_range.split(','))
    #s_min, s_max = map(float, s_range.split(','))

    """
    space = {
#        'boost_true_positive_feedback': hp.choice("x_boost_true_positive_feedback", [0, 1]),
#        'number_of_states': hp.quniform("x_number_of_states", states_min, states_max, states_step),
        'threshold': hp.quniform ('x_threshold', threshold_min, threshold_max, threshold_step),
        's': hp.uniform ('x_s', s_min, s_max),
        }
    """
    space = {
            'n_estimators': hp.quniform("x_n_estimators", 1000, 2000, 20),
            'max_depth': hp.quniform("x_max_depth", 6, 20, 1),
            'min_child_weight': hp.quniform ('x_min_child', 5, 120, 5),
            #'gamma': hp.uniform ('x_gamma', 0.0, 2.0),
            'scale_pos_weight': hp.quniform ('x_scale_pos_weight', 0.2, 1.0, 0.02),

            'num_pairsample': hp.quniform ('x_num_pairsample', 1, 8, 1),
            'learning_rate': hp.quniform ('x_learning_rate', 0.01, 0.05, 0.002),

            'subsample': hp.quniform ('x_subsample', 0.3, 1.0, 0.02),
            'colsample_bytree': hp.quniform ('x_colsample_bytree', 0.4, 1.0, 0.02)
            }

    from functools import partial
    objective_xy = partial(objective, train_X, train_y,
                           nfolds, ncvjobs, ncvrep,
                           #nepochs,
                           #number_of_pos_neg_clauses_per_label,
                           seed, njobs)

    best = fmin(fn=objective_xy,
            space=space,
            algo=tpe.suggest,
            max_evals=neval,
            )
    return best


def work(neval,
         nfolds,
         ncvjobs,
         ncvrep,
         njobs,
         seed,
         #number_of_pos_neg_clauses_per_label,
         #nepochs,
         #states_range,
         #threshold_range,
         #s_range
    ):

    gc.enable()

    df = pd.read_csv('../input/train.csv', nrows=20000)
    #df = pd.read_csv('../input/train.csv.bz2', nrows=20000)
    y = df['target'].values
    del df['ID_code']
    del df['target']
    X = df.values.astype(np.float32)
    print(y.shape)
    print(X.shape)

    del df
    gc.collect()


    print('Hyperopt start')

    best = evaluate_hyper(X, y, hyper_objective,
                          neval=neval, nfolds=nfolds, ncvjobs=ncvjobs, ncvrep=ncvrep,
                          njobs=njobs, seed=seed,
                          #number_of_pos_neg_clauses_per_label=number_of_pos_neg_clauses_per_label,
                          #nepochs=nepochs,
                          #states_range=states_range,
                          #threshold_range=threshold_range,
                          #s_range=s_range
                      )

    print('Final best: {}'.format(best))

    pass


work(neval=200,     # hyperopts iterations
     nfolds=5,      # cv folds (skf)
     ncvjobs=1,     # cv jobs (cross_val_score)
     ncvrep=1,      # repetitions of cross_val_score
     njobs=4,       # njobs for the estimator
     seed=1
    )