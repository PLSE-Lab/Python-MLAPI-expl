#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas
import numpy
import itertools

from sklearn import metrics
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    PolynomialFeatures)
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    ShuffleSplit,
    GridSearchCV,
    cross_val_score)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import (
    BaggingClassifier,
    VotingClassifier,
    RandomForestClassifier)

from xgboost import XGBClassifier


# In[ ]:


train = pandas.read_csv('../input/train.csv')
test = pandas.read_csv('../input/test.csv')

features_numerical = [
    'bone_length',
    'rotting_flesh',
    'hair_length',
    'has_soul'
]

features_categorical = ['color']

target = 'type'
le_target = LabelEncoder().fit(train[target])
labels = le_target.transform(train[target])
classes = list(le_target.classes_)

train = train.drop([target, 'id'], axis=1)
test_ids = test.pop('id')

train = train.drop(features_categorical, axis=1)
test = test.drop(features_categorical, axis=1)


# In[ ]:


combined_features = Pipeline([
    ('feature_set', FeatureUnion([
        ('f_poly', Pipeline([
            ('poly', PolynomialFeatures(
                degree=2,
                interaction_only=False,
                include_bias=False
            )),
            ('scaler', StandardScaler()),
        ])),
        ('f_kpca', Pipeline([
            ('scaler', StandardScaler()),
            ('kpca', KernelPCA(
                n_components=15,
                kernel="rbf",
                fit_inverse_transform=True,
                gamma=1
            )),
        ])),
    ])),
    ('rfe', RFE(
        estimator=SVC(kernel='linear', C=1),
        n_features_to_select=10,
        step=1
    )),
])


# In[ ]:


classifiers = {
    'blinky': LogisticRegression(
        C=1,
        tol=1e-3,
        multi_class= 'multinomial',
        penalty='l2',
        solver='lbfgs'
    ),
    'pinky': RandomForestClassifier(
        n_estimators=250,
        criterion='entropy',
        max_depth=5,
        min_samples_leaf=8,
        min_samples_split=3
    ),
    'inky': GaussianProcessClassifier(
        kernel=1.0 * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-1, 10.0)
        ),
        warm_start=True
    ),
    'clyde': XGBClassifier(
        n_estimators=250,
        objective="multi:softprob",
        max_depth=6,
        learning_rate=0.1,
        gamma=0,
        nthread=6
    ),
}


# In[ ]:


clf_acc = []

ss = ShuffleSplit(
    n_splits=10,
    test_size=0.2,
    random_state=0
)

estimators = dict()

print('\nResults:')
for i, (train_index, val_index) in enumerate(ss.split(train, labels), start=1):
    X_train, X_val = train.as_matrix()[train_index], train.as_matrix()[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('features', combined_features),
            (name, clf),
        ])
        estimator = pipeline.fit(X_train, y_train)
        estimators['%s_%s' % (name, i)] = estimator

        train_predictions = estimator.predict(X_val)
        acc = metrics.accuracy_score(y_val, train_predictions)
        print('acc: {:.2%} | [split {}] {}'.format(acc, i, name))

        clf_acc.append([name, i, acc])

clf_acc = pandas.DataFrame(clf_acc, columns=['classifier', 'split', 'acc'])


# In[ ]:


clf_cv_acc = clf_acc[['classifier', 'acc']]     .groupby('classifier')     .mean()     .sort_values(by='acc', ascending=False
).reset_index()

print('\nMean classifier accuracy:\n%r' % (clf_cv_acc.head(10)))

top_estimators_acc = clf_cv_acc.head(3).classifier.values
vc_estimators = {k: classifiers[k] for k in top_estimators_acc}


# In[ ]:


ss = ShuffleSplit(
    n_splits=3,
    test_size=0.1,
    random_state=0
)

max_w = 3
weight_scores = []
w = range(1, max_w + 1)
for weights in itertools.product(w, w, w):
    if (len(set(weights)) != 1) or (int(sum(weights)) == 3):
        eclf = VotingClassifier(
            estimators=vc_estimators.items(),
            voting='soft',
            weights=weights
        )
        
        # cross_val_score not working in kernels for me, so using below instead...
        for i, (train_index, val_index) in enumerate(ss.split(train, labels), start=1):
            X_train, X_val = train.as_matrix()[train_index], train.as_matrix()[val_index]
            y_train, y_val = labels[train_index], labels[val_index]
            
            estimator = eclf.fit(X_train, y_train)
            acc = metrics.accuracy_score(y_val, estimator.predict(X_val))
            weight_scores.append([weights, i, acc])
            print('%s: [split %s] %s' % (weights, i, acc))
        

cols = ['weights', 'splits', 'acc']
df = pandas.DataFrame(weight_scores, columns=cols)
df = df[['weights', 'acc']]     .groupby('weights')     .mean()     .sort_values(by='acc', ascending=False
).reset_index()

print('\n\n%r' % df.head())
weights, _ = df.as_matrix()[0]


# In[ ]:


eclf = VotingClassifier(
    estimators=vc_estimators.items(),
    voting='soft',
    weights=weights
)
eclf.fit(X_train, y_train)

test_predictions = eclf.predict_proba(test.as_matrix())
test_predictions = [le_target.classes_[numpy.argmax(i)] for i in test_predictions]

submission = pandas.DataFrame(test_predictions, columns=[target])
submission.insert(0, 'id', test_ids)

submission.to_csv('cherries.csv', index=False)

