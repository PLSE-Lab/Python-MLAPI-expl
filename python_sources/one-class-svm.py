import joblib
import json
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM


# Load csv, fill NaN/NA with 0 and return only numeric columns
data = pd.read_csv('/kaggle/input/ansiblenoveltydetection/metrics.csv').fillna(0).select_dtypes(include='number')

Xpos = data[data.defective == 0].drop(['committed_at', 'defective'], axis=1)    # normal class
Xneg = data[data.defective == 1].drop(['committed_at', 'defective'], axis=1)    # anomalous class


# Creating (train, test) tuples of indices for k-folds cross-validation
# We split the positive class (normal data) as we only want the positive examples in the training set.
# Negative examples (abnormal data) are then added to the test set (see https://stackoverflow.com/a/58459322/3673842)
splits = KFold(n_splits=10).split(Xpos)    

X = np.concatenate([Xpos, Xneg], axis=0)
y = np.concatenate([np.repeat(1.0, len(Xpos)), np.repeat(-1.0, len(Xneg))])

n, m = len(Xpos), len(Xneg)
splits = ((train, np.concatenate([test, np.arange(n, n + m)], axis=0)) for train, test in splits)

# Training and validation
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('kbest', SelectKBest(chi2)),
    ('estimator', OneClassSVM())    
])

search_params = dict(
    kbest__k = np.linspace(1, 108, 100, dtype=np.int32), 
    estimator__gamma = np.linspace(0.01, 100, 10),
    estimator__nu = np.linspace(0.01, 0.5, 100),
    estimator__shrinking = [True, False]
)

scoring = dict(
    average_precision = 'average_precision',
    precision = 'precision',
    recall = 'recall',
    mcc = make_scorer(matthews_corrcoef)
)

search = RandomizedSearchCV(pipe, search_params, cv=splits, scoring=scoring, refit='mcc', verbose=5)
search.fit(X, y)

mask = search.best_estimator_.named_steps['kbest'].get_support() #list of booleans
selected_features = [] # The list of K best features

for selected, feature in zip(mask, data.drop(['committed_at', 'defective'], axis=1).columns):
    if selected:
        selected_features.append(feature)

report = pd.DataFrame(search.cv_results_).iloc[[search.best_index_]] # Take only the scores at the best index

# Save performance report
with open('./report.json', 'w') as outfile:
    report.to_json(outfile, orient='table', index=False)

# Save selected features
with open('./features.json', 'w') as outfile:
    json.dump(selected_features, outfile)

# Save model
joblib.dump(search.best_estimator_, './model.pkl')