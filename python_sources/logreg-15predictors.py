# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle as pkl
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open('../input/X_train.pickle','rb') as fx, open('../input/y_train.pickle','rb') as fy:
    X_train = pkl.load(fx)
    y_train = pkl.load(fy)

with open('../input/X_test.pickle','rb') as fx, open('../input/y_test.pickle','rb') as fy:
    X_test = pkl.load(fx)
    y_test = pkl.load(fy)

def log_reg_model(X_train, y_train, X_test, y_test, lam):

    
    _X_tr = X_train
    _y_tr = y_train
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
    X_te = X_test.iloc[:, nonzero_coef]
    y_te = y_test
    print('scoring logistic regression...')
    score = logreg.score(X_te, y_te)
    print(score)
    print(confusion_matrix(y_te, logreg.predict(X_te)))
    
        
    return 


lambd = 0.0009
log_reg_model(X_train, y_train, X_test, y_test, lambd)
print(f'lambda: {lambd}')
