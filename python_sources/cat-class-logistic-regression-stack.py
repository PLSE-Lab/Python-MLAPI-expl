#!/usr/bin/env python
# coding: utf-8

# # Cat Class: Logistic Regression Stack
# 
# _By Nick Brooks, January 2020_
# 
# V2 - 23/01/2020 - Increase C parameter

# In[ ]:


pip install --upgrade scikit-learn


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn import preprocessing
from sklearn import metrics
from sklearn.inspection import permutation_importance

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

from contextlib import contextmanager
import time
import gc
notebookstart = time.time()

@contextmanager
def timer(name):
    """
    Time Each Process
    """
    t0 = time.time()
    yield
    print('\n[{}] done in {} Minutes'.format(name, round((time.time() - t0)/60,2)))


# In[ ]:


seed = 50
debug = False

if debug:
    nrow = 5000
else:
    nrow = None


# In[ ]:


with timer("Load"):
    PATH = "/kaggle/input/cat-in-the-dat-ii/"
    train = pd.read_csv(PATH + "train.csv", index_col = 'id', nrows = nrow)
    test = pd.read_csv(PATH + "test.csv", index_col = 'id')
    submission_df = pd.read_csv(PATH + "sample_submission.csv")
    [print(x.shape) for x in [train, test, submission_df]]

    traindex = train.index
    testdex = test.index

    y = train.target.copy()
    print("Target Distribution:\n",y.value_counts(normalize = True).to_dict())

    df = pd.concat([train.drop('target',axis = 1), test], axis = 0)
    del train, test, submission_df


# [Feature Engineering](https://www.kaggle.com/superant/oh-my-cat) by SuperRant

# In[ ]:


with timer("FE 1"):
    drop_cols=["bin_0"]

    # Split 2 Letters; This is the only part which is not generic and would actually require data inspection
    df["ord_5a"]=df["ord_5"].str[0]
    df["ord_5b"]=df["ord_5"].str[1]
    drop_cols.append("ord_5")

    xor_cols = []
    nan_cols = []
    for col in df.columns:
        # NUll Values
        tmp_null = df.loc[:,col].isnull().sum()
        if tmp_null > 0:
            print("{} has {} missing values.. Filling".format(col, tmp_null))
            nan_cols.append(col)
            if df.loc[:,col].dtype == "O":
                df.loc[:,col].fillna("NAN", inplace=True)
            else:
                df.loc[:,col].fillna(-1, inplace=True)
        
        # Categories that do not overlap
        train_vals = set(df.loc[traindex, col].unique())
        test_vals = set(df.loc[testdex, col].unique())
        
        xor_cat_vals=train_vals ^ test_vals
        if xor_cat_vals:
            df.loc[df[col].isin(xor_cat_vals), col]="xor"
            print("{} has {} xor factors, {} rows".format(col, len(xor_cat_vals),df.loc[df[col] == 'xor',col].shape[0]))
            xor_cols.append(col)


    # One Hot Encode None-Ordered Categories
    ordinal_cols=['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5a', 'day', 'month']
    X_oh=df[df.columns.difference(ordinal_cols)]
    oh1=pd.get_dummies(X_oh, columns=X_oh.columns, drop_first=True, sparse=True)
    ohc1=oh1.sparse.to_coo()


# In[ ]:


from sklearn.base import TransformerMixin
from itertools import repeat
import scipy

class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        
        possible_values = sorted(self.value_map_.values())
        
        idx1 = []
        idx2 = []
        
        all_indices = np.arange(len(X))
        
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
            
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
            
        return result


# In[ ]:


other_classes = ["NAN", 'xor']

with timer("Thermometer Encoder"):
    thermos=[]
    for col in ordinal_cols:
        if col=="ord_1":
            sort_key=(other_classes + ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']).index
        elif col=="ord_2":
            sort_key= (other_classes + ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']).index
        elif col in ["ord_3", "ord_4", "ord_5a"]:
            sort_key=str
        elif col in ["day", "month"]:
            sort_key=int
        else:
            raise ValueError(col)

        enc=ThermometerEncoder(sort_key=sort_key)
        thermos.append(enc.fit_transform(df[col]))


# In[ ]:


ohc=scipy.sparse.hstack([ohc1] + thermos).tocsr()
display(ohc)

X = ohc[:len(traindex)]
test = ohc[len(traindex):]

print(X.shape)
print(test.shape)

del ohc; gc.collect()


# In[ ]:


LogisticRegression().get_params()


# In[ ]:


scoring = "roc_auc"
model_names = ['stack',
               'logistic1',
               'logistic2'
              ]

n_models = len(model_names)

Logistic_params_1 = {
    "C": 0.123,
    "penalty":"l2",
    "solver": "lbfgs",
    "max_iter": 5000
}

Logistic_params_2 = {
    "C": 50,
    'penalty':"l2",
    "solver": "lbfgs",
    "max_iter": 5000
}

folds = KFold(n_splits=3, shuffle=True, random_state=1)
fold_preds = np.zeros([test.shape[0],n_models])
oof_preds = np.zeros([X.shape[0],n_models])
results = {}

with timer("Fit Model"):
    estimators = [
        ('logistic_regression_2', LogisticRegression(**Logistic_params_1)),
        ('logistic_regression_1', LogisticRegression(**Logistic_params_2))
    ]

    # Fit Folds
    f, ax = plt.subplots(1,3,figsize = [14,5])
    for i, (trn_idx, val_idx) in enumerate(folds.split(X)):
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            )
        with timer("Fold {}".format(str(i))):
            clf.fit(X[trn_idx], y.loc[trn_idx])
        tmp_pred = clf.predict_proba(X[val_idx])[:,1]
    
        oof_preds[val_idx,0] = tmp_pred
        fold_preds[:,0] += clf.predict_proba(test)[:,1] / folds.n_splits
        
        estimator_performance = {}
        estimator_performance['stack_score'] = metrics.roc_auc_score(y.loc[val_idx], tmp_pred)
        for ii, est in enumerate(estimators):
            model = clf.named_estimators_[est[0]]
            plot_roc_curve(model, X[val_idx], y.loc[val_idx], ax=ax[i])
            pred = model.predict_proba(X[val_idx])[:,1]
            oof_preds[val_idx, ii+1] = pred
            fold_preds[:,ii+1] += model.predict_proba(test)[:,1] / folds.n_splits
            estimator_performance[est[0]+"_score"] = metrics.roc_auc_score(y.loc[val_idx], pred)
            
        stack_coefficients = {x+"_coefficient":y for (x,y) in zip([x[0] for x in estimators], clf.final_estimator_.coef_[0])}
        stack_coefficients['intercept'] = clf.final_estimator_.intercept_[0]
        
        results["Fold {}".format(str(i+1))] = [
            estimator_performance,
            {est[0]+"_iterations":clf.named_estimators_[est[0]].n_iter_ for est in estimators},
            stack_coefficients
        ]

        plot_roc_curve(clf, X[val_idx], y.loc[val_idx], ax=ax[i])
        ax[i].plot([0.0, 1.0])
        ax[i].set_title("Fold {} - ROC AUC".format(str(i + 1)))

plt.tight_layout(pad=2)
plt.show()

f, ax = plt.subplots(1,2,figsize = [11,5])
sns.heatmap(pd.DataFrame(oof_preds, columns = model_names).corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="magma",ax=ax[0])
ax[0].set_title("OOF PRED - Correlation Plot")
sns.heatmap(pd.DataFrame(fold_preds, columns = model_names).corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="inferno",ax=ax[1])
ax[1].set_title("TEST PRED - Correlation Plot")
plt.tight_layout(pad=3)
plt.show()


# In[ ]:


results_pd = pd.DataFrame(results).T.reset_index()
results_pd.columns = ['Fold','Score','EarlyStopping', 'Coefficients']
results_pd = pd.concat(
    [pd.io.json.json_normalize(results_pd['Score']).reset_index(drop=True),
     pd.io.json.json_normalize(results_pd['EarlyStopping']).reset_index(drop=True),
     pd.io.json.json_normalize(results_pd['Coefficients']).reset_index(drop=True),
     results_pd.reset_index(drop=True)
    ], axis = 1)
display(results_pd)


# In[ ]:


with timer("Submission"):
    pd.DataFrame({'id': testdex, 'target': fold_preds[:,0]}).to_csv('logistic_stacked_oof_submission.csv', index=False)
    pd.DataFrame({'id': testdex, 'target': fold_preds[:,1]}).to_csv(estimators[0][0] + '_oof_submission.csv', index=False)
    pd.DataFrame({'id': testdex, 'target': fold_preds[:,2]}).to_csv(estimators[1][0] + '_oof_submission.csv', index=False)


# In[ ]:


print("Notebook Runtime: %0.2f Hours"%((time.time() - notebookstart)/60/60))


# In[ ]:




