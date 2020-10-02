#!/usr/bin/env python
# coding: utf-8

# # Intro

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
from sklearn import *


# In[ ]:


raw = pd.read_csv('../input/ufcdata/raw_total_fight_data.csv', sep=';')
display(raw.shape)
display(raw.head())


# In[ ]:


fighter = pd.read_csv('../input/ufcdata/raw_fighter_details.csv')
fighter


# In[ ]:


data = raw     .merge(fighter.rename(lambda x: f'R_{x}', axis=1), left_on='R_fighter', right_on='R_fighter_name')    .merge(fighter.rename(lambda x: f'B_{x}', axis=1), left_on='B_fighter', right_on='B_fighter_name')    .drop(['R_fighter_name', 'B_fighter_name'], axis=1)
data


# # Preprocess

# In[ ]:


data.iloc[0]


# In[ ]:


from datetime import datetime, timedelta
from dateutil.parser import parse
import re

data.drop([c for c in data.columns if c.endswith('_pct')], axis=1, inplace=True)

for c in ['date', 'R_DOB', 'B_DOB']:
    data[c] = data[c].map(lambda x: parse(x) if type(x) == str else None)

data['R_age'] = (data['date'] - data['R_DOB']).map(lambda x: x.days / 365.25 if hasattr(x, 'days') else None)
data['B_age'] = (data['date'] - data['B_DOB']).map(lambda x: x.days / 365.25 if hasattr(x, 'days') else None)

data['R_Weight'] = data['R_Weight'].map(lambda x: int(re.search('\d+', x)[0]) if type(x) == str else None)
data['B_Weight'] = data['B_Weight'].map(lambda x: int(re.search('\d+', x)[0]) if type(x) == str else None)

def time2int(x):
    m, s = x.split(':')
    return 60*int(m) + int(s)
data['last_round_time'] = data['last_round_time'].map(time2int)

def winner2color(row):
    return         'Blue' if row['Winner'] == row['B_fighter'] else         'Red'  if row['Winner'] == row['R_fighter'] else         'Draw'
data['Winner'] = data.apply(winner2color, axis=1)

def feet2inch(x):
    if type(x) != str:
        return None
    feet = re.findall('(\d+)\'', x)
    feet = int(feet[0]) if feet else 0
    inch = re.findall('(\d+)\"', x)
    inch = int(inch[0]) if inch else 0
    return feet*12 + inch

for c in data.columns:
    if c[2:] in ['Height', 'Reach']:
        data[c] = data[c].map(feet2inch)

for column in data.columns:
    it = data[column].iloc[0]
    if type(it) == str and 'of' in it:
        data[column+'_landed'] = data[column].map(lambda x: int(x.split('of')[0]))
        data[column+'_att']    = data[column].map(lambda x: int(x.split('of')[1]))
        data.drop(column, axis=1, inplace=True)

data.drop(['R_fighter', 'B_fighter'], axis=1, inplace=True)

for (column, dtype) in data.dtypes.items():
    if dtype == object:
        data[column] = data[column].astype('category')
        
print(data.shape)
data.head()


# In[ ]:


uniques = data.select_dtypes('category').nunique().sort_values()
uniques


# # Utilities

# In[ ]:


from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        
    def fit(self, *_):
        return self
    
    def predict(self, X):
        votes = sum(
            clf.predict_proba(X)
            for clf in self.estimators
        )
        return np.array(self.estimators[0].classes_)[votes.argmax(axis=1)]
    
    
def best_model(models):
    models = pd.DataFrame(models)
    return models['estimator'][models['test_score'].idxmax()]


def permutation_importance(clf, X, y, n_repeats=10, scoring=None):
    scoring = scoring or metrics.balanced_accuracy_score
    y_base = scoring(y, clf.predict(X))
    def aux(X, column):
        X = X.copy()
        acc = []
        for _ in range(n_repeats):
            it = X[column]
            it[:] = it[np.random.permutation(it.index)]
            acc.append(scoring(y, clf.predict(X)) - y_base)
        return column, acc
    results = Parallel(n_jobs=-1)(delayed(aux)(X, column) for column in tqdm(X.columns))
    return pd.DataFrame(dict(results))

# followings are not used.
def smoothed_lower_bound_binomial(c,C):
    c += 0.5
    C += 1
    v = c*(C-c)/C
    return max(0, c - v**.5) / C


def smoothed_lower_bound_beta(c,C):
    c += 0.5
    C += 1
    v = c*(C-c)/C/C/(1+C)
    return max(0, c - v**.5) / C


# In[ ]:


def check(clf, X, y, scoring=None):
    print(f'{metrics.classification_report(y, clf.predict(X))}')
    importances = permutation_importance(clf, X, y, scoring=scoring)
    return importances.abs().mean().sort_values().tail(10).plot.barh(title='Importance')

def X_y(data):
    return data.dropna().drop(['Winner'], axis=1),  data.dropna()['Winner']


# We use this `check` function to inspect how the model is good or not.

# In[ ]:


from sklearn.base import BaseEstimator, ClassifierMixin

class TestClassifier(BaseEstimator, ClassifierMixin):
    def predict(self, X):
        return np.random.choice(['Red', 'Draw', 'Blue'], len(X))
    
X, y = X_y(data)
check(TestClassifier(), X, y)


# we use `cross_validate` function to make estimators for `EnsembleClassifier`.

# In[ ]:


def cross_validate(
    model, X, y,
    fit=lambda model, X_train, y_train, X_valid, y_valid: model.fit(X_train, y_train),
    predict=lambda model, X: model.predict(X),
    scoring=lambda y_true, y_pred: metrics.balanced_accuracy_score(y_true, y_pred),
    cv=5,
):
    cv = model_selection.check_cv(cv)
        
    result = {
        'train_score': [],
        'test_score': [],
        'estimator': [],
    }
    
    def aux(model, X, train_index, valid_index):
        _model = sklearn.base.clone(model)
        X_train, y_train = sklearn.utils.metaestimators._safe_split(_model, X, y, train_index)
        X_valid, y_valid = sklearn.utils.metaestimators._safe_split(_model, X, y, valid_index, train_index)
        fit(_model, X_train, y_train, X_valid, y_valid)
        train_score = scoring(y_train, predict(_model, X_train))
        test_score = scoring(y_valid, predict(_model, X_valid))
        return _model, train_score, test_score

    it = Parallel(n_jobs=-1)(
        delayed(aux)(model, X, train_index, valid_index)
        for train_index, valid_index in tqdm(cv.split(X, y))
    )
    for _model, train_score, test_score in it:
        result['estimator'].append(_model)
        result['train_score'].append(train_score)
        result['test_score'].append(test_score)
        
    return result


# In[ ]:


def cross_validate_catboost(X, y, **kwargs):
    import catboost, utils
    models = cross_validate(
        catboost.CatBoostClassifier(
            od_type='Iter',
            od_wait=10,
            eval_metric='AUC', # this is important to speed up the learning with overfitting detector.
            **kwargs
        ),
        X, y,
        fit=lambda m,Xt,yt,Xv,yv: m.fit(Xt, yt, eval_set=(Xv,yv),
                                        cat_features=np.where(X.dtypes == 'category')[0],
                                        verbose=0),
    )
    return pd.DataFrame(models)


# # Exploring data by Learning
# Here, we use `CatBoost` to analyse the data by observing how it fits to the data. In this way, we can compare categorical and numerical variables with a simple importance calculation.

# In[ ]:


X, y = X_y(data)

get_ipython().run_line_magic('time', 'models = cross_validate_catboost(X, y)')

model = EnsembleClassifier(models['estimator'])
get_ipython().run_line_magic('time', 'check(model, X, y)')


# ## Drop unsound features
# `win_by` looks like the most important variable here, but it's actually not related to the color code of fighters. It means that `Winner` is explained almost only by the variable which never reflects who fights in the match.
# 
# Does this make sense? Now let's drop those color-independent variables to prevent this.

# In[ ]:


neutral_columns = [c for c in data.dtypes.index 
                   if not c.startswith('R_') 
                   and not c.startswith('B_')
                   and not c == 'Winner']
neutral_columns


# In[ ]:


get_ipython().run_cell_magic('time', '', "def X_y_neutral(data):\n    return X_y(data.drop(neutral_columns, axis=1))\n\nX, y = X_y_neutral(data)\nmodels = cross_validate_catboost(X, y,)\nmodel = EnsembleClassifier(models['estimator'])\ncheck(model, X, y)")


# Now the model says `*_KD`(# of knockdowns) are important to tell `Winner`. But Red's and Blue's `KD` have different importances. This might be caused by imbalance of labels.

# ## Deal with Imbalance
# Now we want to make this model be _symmetric_ about the fighter's color code. To this end, let's try the followings:
# - balancing classes by their weights, and
# - flipping the side of fighters.

# In[ ]:


def balancing_params(y):
    counts = y.value_counts()
    return { 'class_names': list(counts.index), 'class_weights': len(y) / counts.values }
balancing_params(y)


# In[ ]:


def flip_concat(raw):
    data = raw.copy()
    def flip_color(c):
        return             f'R_{c[2:]}' if c.startswith('B_') else             f'B_{c[2:]}' if c.startswith('R_') else             c
    raw.columns = data.columns.map(flip_color)
    data['Winner'] = data['Winner'].map({ 'Red': 'Blue', 'Blue': 'Red', 'Draw': 'Draw'})
    it = pd.concat([raw, data], ignore_index=True, sort=False)
    return it.astype({ c: 'category' if t == object else t for c, t in it.dtypes.items() })
flipped = flip_concat(data)
flipped.nunique().sort_values().tail(10)


# ### Balance the weights

# In[ ]:


X, y = X_y_neutral(data)
models = cross_validate_catboost(X, y, **balancing_params(y))
model = EnsembleClassifier(models['estimator'])
check(model, X, y)


# Balancing improved recall over `Draw` class.

# ### Flip the data too

# In[ ]:


def X_y_flip(data):
    data = flip_concat(data)
    return X_y_neutral(data)

X, y = X_y_flip(data)

models = cross_validate_catboost(X, y, **balancing_params(y))
model = EnsembleClassifier(models['estimator'])
check(model, X, y)


# As we expected, the disparity between the color codes is reduced.

# ## Check overfitting
# Let's make sure those ensembled models not to be overfitted to the data.

# In[ ]:


data_train, data_test = model_selection.train_test_split(data, test_size=1/5)


# In[ ]:


X_train, y_train = X_y_flip(data_train)
X_test , y_test  = X_y_flip(data_test)

models = cross_validate_catboost(X_train, y_train, **balancing_params(y_train))
model = EnsembleClassifier(models['estimator'])

print('Train')
check(model, X_train, y_train)


# In[ ]:


print('Test')
check(model, X_test, y_test)


# It seems to be okay if we ignore the `Draw` scores. Balanced accuracy score(`macro avg recall`) is degraded by the low `Draw` class scores.

# # Feature Engineering
# We have good prediction scores except `Draw` class now. How to improve the score for `Draw` class?
# 
# `Draw` score dropped to zero after discarding the "color-independent" features in the early stage of previous section.
# It means we need some direct features in our data to indicate that "There is no significant difference between two."
# 
# Let's make such features by simple way.

# In[ ]:


def difference_features(data):
    data = data.copy()
    dtypes = data.select_dtypes([np.int, np.float]).dtypes
    for c in dtypes.index:
        if c.startswith('R_'):
            # exp(-|diff|) is 1 if |diff| is small and 0 if |diff| is large.
            data[f'diff_{c[2:]}'] = np.exp(-(data[f'R_{c[2:]}'] - data[f'B_{c[2:]}']).abs())
    return data
difference_features(data)


# In[ ]:


def X_y_draw(data):
    X, y = X_y_flip(data)
    return difference_features(X), y

X_train, y_train = X_y_draw(data_train)
X_test, y_test = X_y_draw(data_test)

models = cross_validate_catboost(X_train, y_train, **balancing_params(y_train))
model = EnsembleClassifier(models['estimator'])

print('Train')
check(model, X_train, y_train)


# In[ ]:


print('Test')
check(model, X_test, y_test)


# This slightly imporoves the `Draw` scores with CatBoost but it overfit on `Draw` class more significantly than before.

# # Compare with linear model
# Let's try some linear models to compare the above ways with CatBoost. Especially, we want to check whether the above feature engineering improves the `Draw` scores or not.

# In[ ]:


def X_y_linear(data, X_y=X_y_flip):
    X, y = X_y(data)
    # we can drop these features safely, they were always low important.
    # this simplify the preprocessing for linear model.
    return X.select_dtypes(exclude=['category', 'datetime']), y

logreg = pipeline.Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('clf', linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', class_weight='balanced')),
])


# ## without the difference features

# In[ ]:


X_train, y_train = X_y_linear(data_train)
X_test , y_test  = X_y_linear(data_test)

models = cross_validate(logreg, X_train, y_train)
model = EnsembleClassifier(models['estimator'])
print('Train')
check(model, X_train, y_train)


# In[ ]:


print('Test')
check(model, X_test, y_test)


# ## with the difference features

# In[ ]:


X_train, y_train = X_y_linear(data_train, X_y=X_y_draw)
X_test , y_test  = X_y_linear(data_test, X_y=X_y_draw)

models = cross_validate(logreg, X_train, y_train)
model = EnsembleClassifier(models['estimator'])
print('Train')
check(model, X_train, y_train)


# In[ ]:


print('Test')
check(model, X_test, y_test)


# It has greater influence on `Draw` scores with linear model than with CatBoost, but the recalls slightly worse than CatBoost.
# 
# Anyway, it seems to be really hard to fight against the low scores of low proportional class.

# # Conclusion
# We've tried an incremental way to inspect the given data with CatBoost and made a predictor with 0.8~ accuracy score(`weighted avg recall`). Same thing can be done with other models like linear model, but you need to build up a preprocessing pipeline carefully.
# 
# CatBoost is fast enough and versatile way to learn the data for analysis while letting us ignoring about which features are categorical, numerical or other dtype.
