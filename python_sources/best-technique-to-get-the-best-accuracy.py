import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as pp
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

df = pd.read_csv('Churn_Modelling.csv')
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

for col in df.select_dtypes(object):
    dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummy], axis=1)
    df.drop(col, axis=1, inplace=True)

df['Credit Rate'] = df['CreditScore']
df.loc[(df['Credit Rate'] <= 100), 'Credit Rate'] = 1
df.loc[(df['Credit Rate'] > 100) & (df['Credit Rate'] <= 200), 'Credit Rate'] = 2
df.loc[(df['Credit Rate'] > 200) & (df['Credit Rate'] <= 300), 'Credit Rate'] = 3
df.loc[(df['Credit Rate'] > 300) & (df['Credit Rate'] <= 400), 'Credit Rate'] = 4
df.loc[(df['Credit Rate'] > 400) & (df['Credit Rate'] <= 500), 'Credit Rate'] = 5
df.loc[(df['Credit Rate'] > 500) & (df['Credit Rate'] <= 600), 'Credit Rate'] = 6
df.loc[(df['Credit Rate'] > 600) & (df['Credit Rate'] <= 700), 'Credit Rate'] = 7
df.loc[(df['Credit Rate'] > 700) & (df['Credit Rate'] <= 800), 'Credit Rate'] = 8
df.loc[(df['Credit Rate'] > 800) & (df['Credit Rate'] <= 900), 'Credit Rate'] = 9
df.loc[(df['Credit Rate'] > 900) & (df['Credit Rate'] <= 1000), 'Credit Rate'] = 10

df['balsal'] = df['EstimatedSalary'] / df['CreditScore']

df['Age old'] = df['Age']
df.loc[df['Age old'] < 20, 'Age old'] = 1
df.loc[(df['Age old'] >= 20) & (df['Age old'] < 40), 'Age old'] = 2
df.loc[(df['Age old'] >= 40) & (df['Age old'] < 60), 'Age old'] = 3
df.loc[(df['Age old'] >= 60) & (df['Age old'] < 80), 'Age old'] = 4
df.loc[df['Age old'] >= 60, 'Age old'] = 5

df['Age Decade'] = df['Age'] // 10 * 10

scale = StandardScaler()
X = scale.fit_transform(df.drop('Exited', axis=1))
label = df.Exited

# ------------------------------------------ Select Best Models ------------------------------------------ #

                        # --------------------- Grid Search --------------------- #

# it's better to try as more models as you can and try them with bagging classifier (if not tree-based model like XGboost, Adaboost, RandomForest...)

# GridSearch for the following algorithms:
RFdict = {'bootstrap': [True, False],
          'max_depth': [5, 10, 20, 30, 40, 50, 60, None],
          'max_features': ['auto', 'sqrt', 0.8, 0.5],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [.2, .5, .10],
          'n_estimators': pp.arange(10, 500, 30)}

GBdict = {'learning_rate': [1.0, 0.4, 0.1, 0.01],
          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
          'max_features': ['auto', 'sqrt', 0.8, 0.5],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [.2, .5, .10],
          'n_estimators': pp.arange(10, 500, 30)}

LGBMdict = {'num_leaves': pp.arange(50, 300, 50),
            'min_data_in_leaf': pp.arange(5, 40, 10),
            'max_depth': pp.arange(10, 100, 20),
            'feature_fraction': [0.3, 0.5, 0.8, 1.0],
            'max_bin': pp.arange(10, 50, 5),
            'learning_rate ': [0.1, 0.01, 0.4, 1.0]}

XGboostdict = {'nthread': [4],  # when use hyperthread, xgboost may become slower
               'learning_rate': [.01, 1.0, 0.1, 0.5],
               'max_depth': pp.arange(8, 40, 3),
               'min_child_weight': [4, 2, 8],
               'silent': [1],
               'subsample': [0.7, 1.0, 0.5],
               'n_estimators': pp.arange(100, 500, 50)}

AdaBoostClassifier_dict = {'learning_rate': [0.01, 0.5, 0.1, 1.],
                           'n_estimators': pp.arange(100, 500, 50)}

LGBM_grid = RandomizedSearchCV(LGBMClassifier(), LGBMdict, 1500, 'f1', cv=3)
LGBM_grid.fit(X, label)
print(LGBM_grid.best_score_, LGBM_grid.best_estimator_)
# ---------------#
XGBoost_grid = RandomizedSearchCV(XGBClassifier(), XGboostdict, 1500, 'f1', cv=3)
XGBoost_grid.fit(X, label)
print(XGBoost_grid.best_score_, XGBoost_grid.best_estimator_)

# _--------------#
AdaBoostClassifier_grid = RandomizedSearchCV(AdaBoostClassifier(), AdaBoostClassifier_dict, 1500, 'f1', cv=3)
AdaBoostClassifier_grid.fit(X, label)
print(AdaBoostClassifier_grid.best_score_, AdaBoostClassifier_grid.best_estimator_)

# ---------------- #
GradientBoostingClassifier_grid = RandomizedSearchCV(GradientBoostingClassifier(), GBdict, 1500, 'f1', cv=3)
GradientBoostingClassifier_grid.fit(X, label)
print(GradientBoostingClassifier_grid.best_score_, GradientBoostingClassifier_grid.best_estimator_)

# ------------- #
RandomForestClassifier_grid = RandomizedSearchCV(RandomForestClassifier(), RFdict, 1500, 'f1', cv=3)
RandomForestClassifier_grid.fit(X, label)

print(RandomForestClassifier_grid.best_score_, RandomForestClassifier_grid.best_estimator_)

# ------------- #

ExtraTreesClassifier_grid = RandomizedSearchCV(ExtraTreesClassifier(), RFdict, 1500, 'f1', cv=3)
ExtraTreesClassifier_grid.fit(X, label)

print(ExtraTreesClassifier_grid.best_score_, ExtraTreesClassifier_grid.best_estimator_)

                        # --------------------- Cross Val Score --------------------- #

cv = 10
model = LGBM_grid.best_estimator_
scores = cross_val_score(model, X, label, cv=cv)
print(f'''
LGBMClassifier:
        mean: {pp.round(pp.mean(scores), 4)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

model = XGBoost_grid.best_estimator_
scores = cross_val_score(model, X, label, cv=cv)
print(f'''
XGBClassifier:
        mean: {pp.round(pp.mean(scores), 2)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

model = AdaBoostClassifier_grid.best_estimator_
scores = cross_val_score(model, X, label, cv=cv)
print(f'''
AdaBoostClassifier:
        mean: {pp.round(pp.mean(scores), 2)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

model = RandomForestClassifier_grid.best_estimator_
scores = cross_val_score(model, X, label, cv=cv)
print(f'''
RandomForestClassifier:
        mean: {pp.round(pp.mean(scores), 2)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

model = ExtraTreesClassifier_grid.best_estimator_
scores = cross_val_score(model, X, label, cv=cv)
print(f'''
ExtraTreesClassifier:
        mean: {pp.round(pp.mean(scores), 2)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

'''
LGBMClassifier:
        mean: 0.86 | STD: 0.0
       

XGBClassifier:
        mean: 0.86 | STD: 0.0
       
AdaBoostClassifier:
        mean: 0.86 | STD: 0.0
       

GradientBoostingClassifier:
        mean: 0.86 | STD: 0.0
       

RandomForestClassifier:
        mean: 0.85 | STD: 0.0
       

ExtraTreesClassifier:
        mean: 0.85 | STD: 0.0 
'''
# ---------------------------- Last Improvements for the best models ---------------------------- #

            # --------------------- Feature Importances --------------------- #
'''
model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                               criterion='gini', max_depth=10, max_features=0.8,
                               max_leaf_nodes=None, max_samples=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=0.1,
                               min_weight_fraction_leaf=0.0, n_estimators=130,
                               n_jobs=None, oob_score=False, random_state=None,
                               verbose=0, warm_start=False)

model2 = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                              criterion='gini', max_depth=None, max_features=0.8,
                              max_leaf_nodes=None, max_samples=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=2, min_samples_split=0.1,
                              min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                              oob_score=False, random_state=None, verbose=0,
                              warm_start=False)

model3 = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                    learning_rate=0.4, loss='deviance', max_depth=60,
                                    max_features='sqrt', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=4, min_samples_split=0.1,
                                    min_weight_fraction_leaf=0.0, n_estimators=40,
                                    n_iter_no_change=None, presort='deprecated',
                                    random_state=None, subsample=1.0, tol=0.0001,
                                    validation_fraction=0.1, verbose=0,
                                    warm_start=False)

model4 = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
                       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                       importance_type='gain', interaction_constraints=None,
                       learning_rate=0.1, max_delta_step=0, max_depth=14,
                       min_child_weight=8, n_estimators=100, n_jobs=4, nthread=4, num_parallel_tree=1,
                       objective='binary:logistic', random_state=0, reg_alpha=0,
                       reg_lambda=1, scale_pos_weight=1, silent=1, subsample=0.5,
                       tree_method=None, validate_parameters=False, verbosity=None)
model5 = LGBMClassifier(boosting_type='gbdt', class_weight=None, importance_type='split', learning_rate=0.1,
                        max_bin=10, max_depth=70, min_child_weight=0.001,
                        min_data_in_leaf=35, min_split_gain=0.0, n_estimators=100,
                        n_jobs=-1, num_leaves=50, objective=None, random_state=None,
                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                        subsample_for_bin=200000, subsample_freq=0)
model6 = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                            n_estimators=100, random_state=None)
'''
model = ExtraTreesClassifier_grid.best_estimator_
model2 = RandomForestClassifier_grid.best_estimator_
model3 = GradientBoostingClassifier_grid.best_estimator_
model4 = XGBoost_grid.best_estimator_
model5 = LGBM_grid.best_estimator_
model6 = AdaBoostClassifier_grid.best_estimator_

model.fit(X, label)
model2.fit(X, label)
model3.fit(X, label)
model4.fit(X, label)
model5.fit(X, label)
model6.fit(X, label)

Importance1 = pd.Series(model.feature_importances_, index=df.drop('Exited', axis=1).columns)
Importance2 = pd.Series(model2.feature_importances_, index=df.drop('Exited', axis=1).columns)
Importance3 = pd.Series(model3.feature_importances_, index=df.drop('Exited', axis=1).columns)
Importance4 = pd.Series(model4.feature_importances_, index=df.drop('Exited', axis=1).columns)
Importance5 = pd.Series(model5.feature_importances_, index=df.drop('Exited', axis=1).columns)
Importance6 = pd.Series(model6.feature_importances_, index=df.drop('Exited', axis=1).columns)
Importances = pd.Series(Importance1 + Importance2 + Importance3 + Importance4 + Importance5 + Importance6)

# Importances.nlargest(15).plot(kind='barh')
# plt.show()

                # ----------------- select best cols that affect the models accuracy ----------------- #
best_clf = dict(cols=[], score=[], clf=[])
cv = 10
for i in range(3, len(Importance1.index)):
    X = df[Importance1.nlargest(i).index]

    scores = cross_val_score(model, X, label, cv=cv)

    best_clf['clf'].append('GB')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.round(pp.mean(scores), 3))

    scores = cross_val_score(model2, X, label, cv=cv)
    best_clf['clf'].append('XGBoost')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.round(pp.mean(scores), 3))

    scores = cross_val_score(model3, X, label, cv=cv)
    best_clf['clf'].append('LGBM')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.round(pp.mean(scores), 3))

    scores = cross_val_score(model4, X, label, cv=cv)
    best_clf['clf'].append('RFC')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.round(pp.mean(scores), 3))

    scores = cross_val_score(model5, X, label, cv=cv)
    best_clf['clf'].append('ExtraRFC')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.round(pp.mean(scores), 3))

    scores = cross_val_score(model6, X, label, cv=cv)
    best_clf['clf'].append('Adaboost')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.round(pp.mean(scores), 3))

best_clf = pd.DataFrame(best_clf)
best_clf = best_clf.sort_values('score', ascending=False)
n_cols = best_clf.loc[best_clf['score'] == max(best_clf['score']), 'cols'].drop_duplicates()
        # --------- Combining best models by (StackingClassifier, VotingClassifier, BaggingClassifier) ------------- #
'''
F Score:

0.5894956452420831 LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               feature_fraction=1.0, importance_type='split', learning_rate=0.1,
               learning_rate =1.0, max_bin=10, max_depth=70,
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=35, min_split_gain=0.0, n_estimators=100,
               n_jobs=-1, num_leaves=50, objective=None, random_state=None,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
               
0.5876419337165254 XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.1, max_delta_step=0, max_depth=14,
              min_child_weight=8, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=4, nthread=4, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, silent=1, subsample=0.5,
              tree_method=None, validate_parameters=False, verbosity=None)

0.5714435913451202 AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=100, random_state=None)

0.5978304603563802 GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.4, loss='deviance', max_depth=60,
                           max_features='sqrt', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=4, min_samples_split=0.1,
                           min_weight_fraction_leaf=0.0, n_estimators=40,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
0.5448264700432848 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=0.8,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=0.1,
                       min_weight_fraction_leaf=0.0, n_estimators=130,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
0.5318791819166913 ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features=0.8,
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=2, min_samples_split=0.1,
                     min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
                     oob_score=False, random_state=None, verbose=0,
                     warm_start=False)

'''
                                # -------- Best Models ------- #
    
ExtraRFC = ExtraTreesClassifier_grid.best_estimator_
RFC = RandomForestClassifier_grid.best_estimator_
GB = GradientBoostingClassifier_grid.best_estimator_
XGBoost = XGBoost_grid.best_estimator_
LGBM = LGBM_grid.best_estimator_
Adaboost = AdaBoostClassifier_grid.best_estimator_
'''
ExtraRFC = model2
RFC = model
GB = model3
XGBoost = model4
LGBM = model5
Adaboost = model6
'''

best_cols = Importances.nlargest(20).index
X = df[best_cols]
X = scale.fit_transform(X)

                    # ----- VotingClassifier ----- #
vote = VotingClassifier([('ExtraRFC', ExtraRFC), ('RFC', RFC), ('GB', GB), ('Adaboost', Adaboost), ('XGBoost', XGBoost),
                         ('LGBM', LGBM)])

scores = cross_val_score(vote, X, label, cv=10)
print(f'''
VotingClassifier:
        mean: {pp.round(pp.mean(scores), 3)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

                    # ----- StackingClassifier ----- #
stack = StackingClassifier([('ExtraRFC', ExtraRFC), ('RFC', RFC), ('GB', GB),
                            ('Adaboost', Adaboost), ('XGBoost', XGBoost), ('LGBM', LGBM)])

scores = cross_val_score(stack, X, label, cv=10)
print(f'''
StackingClassifier:
        mean: {pp.round(pp.mean(scores), 3)} | STD: {pp.round(pp.std(scores), 2)}
       ''')

'''
VotingClassifier:
        mean: 0.862 | STD: 0.03       
StackingClassifier:
        mean: 0.862 | STD: 0.03
'''

                # ----- BaggingClassifier: StackingClassifier ----- #
bstack = BaggingClassifier(base_estimator=stack)
params = {'max_features': pp.arange(0.3, 1.0), 'max_samples': pp.arange(0.3, 1.0),
          'n_estimators': pp.arange(5, 20, 3)}

grid = RandomizedSearchCV(bstack, params, cv=3, n_iter=1400)
grid.fit(X, label)

print(f'bstackGrid: {grid.best_score_} ... {grid.best_estimator_}')
scores = cross_val_score(grid.best_estimator_, X, label, cv=10)

print(f'''
BaggingStackingClassifier:
        mean: {pp.round(pp.mean(scores), 3)} | STD: {pp.round(pp.std(scores))}
       ''')

                # ----- BaggingClassifier: VotingClassifier ----- #
bvote = BaggingClassifier(base_estimator=vote)
params = {'max_features': pp.arange(0.3, 1.0), 'max_samples': pp.arange(0.3, 1.0),
          'n_estimators': pp.arange(5, 20, 3)}

grid = RandomizedSearchCV(bvote, params, cv=3, n_iter=1400)
grid.fit(X, label)
print(f'bvoteGrid: {grid.best_score_} ... {grid.best_estimator_}')

scores = cross_val_score(grid.best_estimator_, X, label, cv=10)
print(f'''
BaggingVotingClassifier:
        mean: {pp.round(pp.mean(scores), 3)} | STD: {pp.round(pp.std(scores))}
       ''')

'''
Best Classifier:

    BaggingVotingClassifier: accuracy %86
'''

