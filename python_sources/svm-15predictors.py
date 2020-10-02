from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Lasso
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle as pkl


# Any results you write to the current directory are saved as output.
with open('../input/X_train.pickle','rb') as fx, open('../input/y_train.pickle','rb') as fy:
    X_train = pkl.load(fx)
    y_train = pkl.load(fy)

with open('../input/X_test.pickle','rb') as fx, open('../input/y_test.pickle','rb') as fy:
    X_test = pkl.load(fx)
    y_test = pkl.load(fy)

def svm_model(X_train, y_train, X_test, y_test, tune, lam):

    _X_tr = X_train
    _y_tr = y_train
    lasso = Lasso(alpha=lam)
    print('fitting lasso...')
    lasso.fit(_X_tr, _y_tr)
    nonzero_coef = lasso.coef_ != 0
    print('building SVM...')
    svm = LinearSVC(C=tune, loss='hinge')
    X_tr = _X_tr.loc[:, nonzero_coef].copy()
    with open('X_tr.pickle') as x:
        pkl.dump(X_tr)
    y_tr = _y_tr
    with open('y_tr.pickle') as y:
        pkl.dump(y_tr)
    print(f'--- Cols: {X_tr.columns}')
    svm.fit(X_tr, y_tr)
    X_te = X_test.iloc[:, nonzero_coef]
    with open('X_te.pickle') as x:
        pkl.dump(X_te)
    y_te = y_test
    with open('y_te.pickle') as y:
        pkl.dump(y_te)
    print('scoring SVM...')
    score = svm.score(X_te, y_te)
    print(score)
    print(confusion_matrix(y_te, svm.predict(X_te)))
    
        
    return 

tune = 5
lambd = 0.0009
svm_model(X_train, y_train, X_test, y_test, tune, lambd)
print(f'lambda: {lambd}')