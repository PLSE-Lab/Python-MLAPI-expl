import os
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

random_state = 1
random.seed(random_state)
np.random.seed(random_state)


print('> Loading data')
X_train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')
X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

y_train = X_train.pop('Cover_Type').astype('int8')


print('> Processing features')
# - https://www.kaggle.com/jakelj/basic-ensemble-model
# - https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575
# - https://www.kaggle.com/kwabenantim/forest-cover-feature-engineering

# Join train and test
X = pd.concat([X_train, X_test])

# Add new features
X['Hydro_Elevation_diff'] = (X['Elevation'] - 
                             X['Vertical_Distance_To_Hydrology'])

X['Hydro_Fire_sum'] = (X['Horizontal_Distance_To_Hydrology'] + 
                       X['Horizontal_Distance_To_Fire_Points'])

X['Hydro_Fire_diff'] = (X['Horizontal_Distance_To_Hydrology'] - 
                        X['Horizontal_Distance_To_Fire_Points']).abs()

X['Hydro_Road_sum'] = (X['Horizontal_Distance_To_Hydrology'] +
                       X['Horizontal_Distance_To_Roadways'])

X['Hydro_Road_diff'] = (X['Horizontal_Distance_To_Hydrology'] -
                        X['Horizontal_Distance_To_Roadways']).abs()

X['Road_Fire_sum'] = (X['Horizontal_Distance_To_Roadways'] + 
                      X['Horizontal_Distance_To_Fire_Points'])

X['Road_Fire_diff'] = (X['Horizontal_Distance_To_Roadways'] - 
                       X['Horizontal_Distance_To_Fire_Points']).abs()

X['Soil_Type'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
X['Wilderness_Area'] = sum(i * X['Wilderness_Area{}'.format(i)] for i in range(1, 5))

soil_count = X['Soil_Type'].value_counts().to_dict()
X['Soil_count'] = X['Soil_Type'].map(soil_count)

soil_elevation = X.groupby('Soil_Type')['Elevation'].median().to_dict()
X['Soil_Elevation'] = X['Soil_Type'].map(soil_elevation)

X['Soil_Area'] = (X['Soil_Type'] + 40 * (X['Wilderness_Area'] - 1)).astype('category')

area_count = X['Wilderness_Area'].value_counts().to_dict()
X['Area_count'] = X['Wilderness_Area'].map(area_count)

# Drop features not useful for classification
drop_cols = ['Aspect', 'Slope',  'Hillshade_3pm']
drop_cols += ['Soil_Type', 'Wilderness_Area']
drop_cols += ['Soil_Type{}'.format(i) for i in range(1, 41)]
drop_cols += ['Wilderness_Area{}'.format(i) for i in range(1, 5)]
drop_cols = [col for col in drop_cols if col in X.columns]

if drop_cols:
    X = X.drop(drop_cols, axis='columns')

# Drop features with low variance in training set
lo_var_cols = []
max_mode_freq = len(X_train) - 10

for col in X.columns:
    if  X.loc[X_train.index, col].value_counts().iat[0] > max_mode_freq:
        lo_var_cols.append(col)

if lo_var_cols:
    X = X.drop(lo_var_cols, axis='columns')

# Scale and bin features
scale_cols = set(X.columns.tolist()) - {'Soil_Area'}

for col in scale_cols:
    X[col] = 100 * (X[col] - X[col].min()) / (X[col].max() - X[col].min())
    X[col] = np.floor(X[col]).astype('int8')

# Separate train and test
X_train = X.loc[X_train.index, :]
X_test = X.loc[X_test.index, :]

del X
print('> No. of features = {}'.format(X_train.columns.size))


print('> Setting up classifiers')
max_features = min(30, X_train.columns.size)

ab_clf = AdaBoostClassifier(n_estimators=200,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=2,
                                random_state=random_state),
                            random_state=random_state)

et_clf = ExtraTreesClassifier(n_estimators=300,
                              min_samples_leaf=2,
                              min_samples_split=2,
                              max_depth=50,
                              max_features=max_features,
                              random_state=random_state,
                              n_jobs=1)

lg_clf = LGBMClassifier(n_estimators=300,
                        num_leaves=128,
                        verbose=-1,
                        random_state=random_state,
                        n_jobs=1)

rf_clf = RandomForestClassifier(n_estimators=300,
                                random_state=random_state,
                                n_jobs=1)

ensemble = [('AdaBoostClassifier', ab_clf),
            ('ExtraTreesClassifier', et_clf),
            ('LGBMClassifier', lg_clf),
            ('RandomForestClassifier', rf_clf)]


print('> Cross-validating classifiers')
for label, clf in ensemble:
    score = cross_val_score(clf, X_train, y_train,
                            cv=5,
                            scoring='accuracy',
                            verbose=0,
                            n_jobs=-1)

    print('  -- {: <24} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))


print('> Fitting stack')
# - https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17
# - https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

stack = StackingCVClassifier(classifiers=[ab_clf, et_clf, lg_clf],
                             meta_classifier=rf_clf,
                             cv=5,
                             stratify=True,
                             shuffle=True,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=-1)

stack = stack.fit(X_train, y_train)


print('> Making predictions')
predictions = stack.predict(X_test)
predictions = pd.Series(predictions, index=X_test.index, dtype=y_train.dtype)

print('> Creating submission')
predictions.to_csv('submission.csv', header=['Cover_Type'], index=True, index_label='Id')

print('> Done !')