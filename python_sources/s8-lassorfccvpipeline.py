from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
import os
import pickle as pkl
import numpy as np


print(os.listdir('../input'))
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

def cv_one_rfc_model(X_train, y_train, folds, lam, n_est, max_d):

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
        print('building support vector machine')
        rfc = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, n_jobs=-1)
        X_tr = _X_tr.loc[:, nonzero_coef].copy()
        y_tr = _y_tr
        print(f'--- Cols: {X_tr.columns}')
        rfc.fit(X_tr, y_tr)
        X_va = X_train.iloc[fold[va], nonzero_coef]
        y_va = y_train[fold[va]]
        print('scoring RFC...')
        score = rfc.score(X_va, y_va)
        print(score)
        results.append(score)
        
    return results

lamb = 0.0025
n_est = 50
max_d = 10

outcome = cv_one_rfc_model(X_train, y_train, folds, lamb, n_est, max_d)

print(f'number of trees: {n_est}')
print(f'max depth: {max_d}')
print(f'lambda: {lamb}')
print(f'avg acc: {np.mean(outcome)}')