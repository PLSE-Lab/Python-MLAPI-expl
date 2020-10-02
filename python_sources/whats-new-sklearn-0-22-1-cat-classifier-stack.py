#!/usr/bin/env python
# coding: utf-8

# # Whats new Sklearn 0.22.1 - Classifier Model Stack on Categorical Data
# 
# **Goal:** <br>
# Check out the new functionalities of [Sklearn 0.22.1](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_22_0.html), which include:
# - Stacking
# - HistogramGradient Boosting
# - Feature Permutation

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

if not debug:
    nrow = None # 25000
    max_trees = 2000 # 5
else:
    nrow = 5000 # 25000
    max_trees = 5 # 5


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


# In[ ]:


with timer("Categorical Processing"):
    categorical = df.columns
    # Encoder:
    for col in categorical:
        diff = list(set(df.loc[testdex, col].unique()) - set(df.loc[traindex,col].unique()))
        if diff:
            print("Column {} has {} unseen categories in test set".format(col, len(diff)))
            df.loc[df[col].isin(diff),col] = 999
        if df[col].dtypes == object:
            df[col] = df[col].astype(str)
        lbl = preprocessing.LabelEncoder()
        df[col] = lbl.fit_transform(df[col].values).astype(int)


# In[ ]:


# Define Data
X = df.loc[traindex,:]
test = df.loc[testdex,:]

print(X.shape)
print(test.shape)


# In[ ]:


HistGradientBoostingClassifier().get_params()


# In[ ]:


scoring = "roc_auc"

HistGBM_param = {
    'l2_regularization': 0.0,
    'loss': 'auto',
    'max_bins': 255,
    'max_depth': 15,
    'max_iter': max_trees,
    'max_leaf_nodes': 31,
    'min_samples_leaf': 20,
    'n_iter_no_change': 50,
    'random_state': seed,
    'scoring': scoring,
    'tol': 1e-07,
    'validation_fraction': 0.15,
    'verbose': 0,
    'warm_start': False   
}

HistGBM_param_deep = HistGBM_param.copy()
HistGBM_param_shallow = HistGBM_param.copy()

# Configure Subsets
HistGBM_param_deep["max_depth"] = 15
HistGBM_param_deep['learning_rate'] = 0.1
HistGBM_param_deep["max_leaf_nodes"] = 70
HistGBM_param_deep["min_samples_leaf"] = 20

HistGBM_param_shallow["max_depth"] = 5
HistGBM_param_shallow['learning_rate'] = 0.1
HistGBM_param_shallow["max_leaf_nodes"] = 70
HistGBM_param_shallow["min_samples_leaf"] = 35

folds = KFold(n_splits=3, shuffle=True, random_state=1)
fold_preds = np.zeros([test.shape[0],3])
oof_preds = np.zeros([X.shape[0],3])
results = {}

with timer("Fit Model"):
    estimators = [
        ('hgbc_deep', HistGradientBoostingClassifier(**HistGBM_param_deep)),
        ('hgbc_shallow', HistGradientBoostingClassifier(**HistGBM_param_shallow))
    ]

    # Fit Folds
    f, ax = plt.subplots(1,3,figsize = [14,5])
    for i, (trn_idx, val_idx) in enumerate(folds.split(X)):
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            )
        clf.fit(X.loc[trn_idx,:], y.loc[trn_idx])
        tmp_pred = clf.predict_proba(X.loc[val_idx,:])[:,1]
    
        oof_preds[val_idx,0] = tmp_pred
        fold_preds[:,0] += clf.predict_proba(test)[:,1] / folds.n_splits
        
        estimator_performance = {}
        estimator_performance['stack_score'] = metrics.roc_auc_score(y.loc[val_idx], tmp_pred)
        for ii, est in enumerate(estimators):
            model = clf.named_estimators_[est[0]]
            plot_roc_curve(model, X.loc[val_idx,:], y.loc[val_idx], ax=ax[i])
            pred = model.predict_proba(X.loc[val_idx,:])[:,1]
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

        plot_roc_curve(clf, X.loc[val_idx,:], y.loc[val_idx], ax=ax[i])
        ax[i].plot([0.0, 1.0])
        ax[i].set_title("Fold {} - ROC AUC".format(str(i)))

plt.tight_layout(pad=2)
plt.show()

f, ax = plt.subplots(1,2,figsize = [11,5])
sns.heatmap(pd.DataFrame(oof_preds, columns = ['stack','shallow','deep']).corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="magma",ax=ax[0])
ax[0].set_title("OOF PRED - Correlation Plot")
sns.heatmap(pd.DataFrame(fold_preds, columns = ['stack','shallow','deep']).corr(),
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


# ### Linear Stacker

# In[ ]:


stacker_preds_test = np.zeros([test.shape[0],2])
stacker_preds_train = np.zeros([X.shape[0],2])

# LinearRegression Stacker to get Weights
Linear_regression_model = LinearRegression(fit_intercept=True)
Linear_regression_model_results = cross_val_score(Linear_regression_model, oof_preds[:,1:], y, scoring = scoring, cv = 5)

Linear_regression_model.fit(oof_preds[:,1:], y)
Linear_regression_model_coefficients = Linear_regression_model.coef_
Linear_regression_model_intercept = Linear_regression_model.intercept_
stacker_preds_train[:,0] = Linear_regression_model.predict(oof_preds[:,1:])

print("Linear Regression Stacker Results:\nCV Score: {:.4f} +/- {:.4f}\nTraining Score: {:.4f}\nCoefficients: {}\nIntecept: {}".format(
    np.mean(Linear_regression_model_results), np.std(Linear_regression_model_results),
    metrics.roc_auc_score(y, stacker_preds_train[:,0]),
    Linear_regression_model_coefficients, Linear_regression_model_intercept
))

# LogisticRegression Stacker to get Weights
Logistic_regression_model = LogisticRegression()
Logistic_regression_model_results = cross_val_score(Logistic_regression_model, oof_preds[:,1:], y, scoring = scoring, cv = 5)

Logistic_regression_model.fit(oof_preds[:,1:], y)
Logistic_regression_model_coefficients = Logistic_regression_model.coef_
Logistic_regression_model_intercept = Logistic_regression_model.intercept_
stacker_preds_train[:,1] = Logistic_regression_model.predict_proba(oof_preds[:,1:])[:,1]

print("\nLogistic Regression Stacker Results:\nCV Score: {:.4f} +/- {:.4f}\nTraining Score: {:.4f}\nCoefficients: {}\nIntecept: {}".format(
    np.mean(Logistic_regression_model_results), np.std(Logistic_regression_model_results),
    metrics.roc_auc_score(y, stacker_preds_train[:,1]),
    Logistic_regression_model_coefficients, Logistic_regression_model_intercept
))

# Test predict
stacker_preds_test[:,0] = Linear_regression_model.predict(fold_preds[:,1:])
stacker_preds_test[:,1] = Logistic_regression_model.predict_proba(fold_preds[:,1:])[:,1]


# In[ ]:


with timer("Submission"):
    pd.DataFrame({'id': testdex, 'target': stacker_preds_test[:,0] }).to_csv("Linear_OOF_stack_submission.csv", index=False)
    pd.DataFrame({'id': testdex, 'target': stacker_preds_test[:,1] }).to_csv('Logistic_OOF_stack_submission.csv', index=False)


# ### Stacker on Full Training Data
# 
# Now that I have the optimal iterations, run GBMs on full data without early stopping..

# In[ ]:


HistGBM_param_shallow_final = HistGBM_param_shallow.copy()
HistGBM_param_shallow_final['n_iter_no_change'] = None
HistGBM_param_shallow_final['max_iter'] = int(results_pd['hgbc_shallow_iterations'].mean())

HistGBM_param_deep_final = HistGBM_param_deep.copy()
HistGBM_param_deep_final['n_iter_no_change'] = None
HistGBM_param_deep_final['max_iter'] = int(results_pd['hgbc_shallow_iterations'].mean())

with timer("Fit Model"):
    estimators = [
        ('hgbc_deep', HistGradientBoostingClassifier(**HistGBM_param_deep_final)),
        ('hgbc_shallow', HistGradientBoostingClassifier(**HistGBM_param_shallow_final))
    ]
    full_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )
    full_clf.fit(X, y)


# In[ ]:


with timer("Feature Permutation on Stack"):
    result = permutation_importance(full_clf, X, y, n_repeats=3, random_state=seed, n_jobs=-1, scoring=scoring)

    fig, ax = plt.subplots(figsize = [10,5])
    sorted_idx = result.importances_mean.argsort()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=df.iloc[:,sorted_idx].columns)
    ax.set_title("Permutation Importance of each feature: ROC-AUC")
    ax.set_ylabel("Features")
    fig.tight_layout()
    plt.show()


# In[ ]:


with timer("Submission"):
    pd.DataFrame({'id': testdex, 'target': clf.predict_proba(test)[:,1]}).to_csv('full_train_stacker_submission.csv', index=False)


# In[ ]:


print("Notebook Runtime: %0.2f Hours"%((time.time() - notebookstart)/60/60))


# In[ ]:




