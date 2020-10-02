import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import Lasso, LogisticRegression
import os

#print(os.listdir('../input'))
with open('../input/s4-dummysplits/X_train.pickle','rb') as fx, open('../input/s4-dummysplits/y_train.pickle','rb') as fy:
    X_train = pkl.load(fx)
    y_train = pkl.load(fy)
with open('../input/s5-cvfolds/folds_list.pickle', 'rb') as f:
    folds = pkl.load(f)

#X_train.iloc[folds[0][0]] the training set X for first iteration of cv
## gets fed to lasso
#y_train[folds[0][0]] the training set y for first iteration
##gets fed to lasso

#the fit from this determines the coeffs to use
## X_train and y_train from abvegets fed to logistic regression but need to subset predictors
#obtain the fit

#X_train.iloc[folds[0][1]] the validation X for first iteration of cv
#predict on these
#y_train[folds[0][1]] the validation y for first iteration
##test against these

def cv_one_lambda(X_train, y_train, folds, lam):

    tr = 0
    va = 1
    results = []
    
    for fold in folds:
    
        _X_tr = X_train.iloc[fold[tr]]
        _y_tr = y_train[fold[tr]]
        lasso = Lasso(alpha=lam)
        print('fitting lasso...')
        lasso.fit(_X_tr, _y_tr)
        nonzero_coef = lasso.coef_ != 0
        print('fitting logistic regression...')
        logreg = LogisticRegression(random_state=5, solver='sag')
        X_tr = _X_tr.loc[:, nonzero_coef].copy()
        y_tr = _y_tr
        print(f'--- Cols: {X_tr.columns}')
        logreg.fit(X_tr, y_tr)
        X_va = X_train.iloc[fold[va], nonzero_coef]
        y_va = y_train[fold[va]]
        print('scoring logistic regression...')
        score = logreg.score(X_va, y_va)
        print(score)
        results.append(score)
        
    return results

with open('cv_one_lambda_funct.pickle','wb') as f:
    pkl.dump(cv_one_lambda, f)

lambd = 0.0025
outcomes = cv_one_lambda(X_train, y_train, folds, lambd)
print(f'lambda: {lambd}')
print(f'avg score: {np.mean(outcomes)}')