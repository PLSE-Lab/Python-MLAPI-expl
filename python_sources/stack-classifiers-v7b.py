#!/usr/bin/env python
"""Create a submission using a stack of classifiers for predicting:

- AdaBoostClassifier
- LGBMClassifier
- RandomForestClassifier
- ExtraTreesClassifier

The feature engineering is done in other notebook and the result is applied here.
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

__author__ = "Juanma Hernández"
__copyright__ = "Copyright 2019"
__credits__ = ["Juanma Hernández", "Kwabena"]
__license__ = "GPL"
__version__ = "3"
__maintainer__ = "Juanma Hernández"
__email__ = "https://twitter.com/juanmah"
__status__ = "Development"

FEATURES_SELECTED = ['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                     'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type2', 'Soil_Type4', 'Soil_Type6', 'Soil_Type10',
                     'Soil_Type13', 'Soil_Type19', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
                     'Soil_Type26', 'Soil_Type30', 'Soil_Type32', 'Soil_Type34', 'Soil_Type40',
                     'Hydro_Elevation_sum', 'Hydro_Elevation_diff', 'Hydro_Distance_sum', 'Hydro_Distance_diff',
                     'Hydro_Fire_diff', 'Hydro_Fire_mean', 'Hydro_Fire_median', 'Hydro_Road_diff', 'Hydro_Road_mean',
                     'Hydro_Road_median', 'Road_Fire_diff', 'Road_Fire_mean', 'Road_Fire_median',
                     'Hydro_Road_Fire_mean', 'Hillshade_max', 'Stoneyness', 'Wilderness_Area1', 'Wilderness_Area2']
RANDOM_STATE = 1
N_JOBS = -1
N_ESTIMATORS = 2000

"""
Changelog
---------

# v6 (2019.09.16)

- Add more features
- Use *Features engineering: improve* v4 features
- Add N_ESTIMATORS constant

# v5 (2019.09.15)

- Fix error

# v4 (2019.09.15)

- Fitting & predicting for each Wilderness_Area

# v3 (2019.09.14)

- Apply new format style
- Use *Features engineering: improve* v2 features
- Add new features from Kwabena (https://www.kaggle.com/kwabenantim/forest-cover-stacking-multiple-classifiers)

"""

# noinspection PyPep8Naming
def get_accuracy(estimator, X, y):
    """Wrapper to do get the accuracy

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    accuracy: float
        The average score of all cross validation scores.
    """
    scores = cross_val_score(estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    return np.mean(scores)


# noinspection PyPep8Naming
def add_features(X):
    """Add features to the independent variables list
    The new features are created by calculations on the original features

    Parameters
    ----------
    X : array-like
        The data to fit. Can be for example a list, or an array.

    Returns
    -------
    X : array-like
        Original data plus the new features.

    """
    X = X.copy()

    X['Hydro_Elevation_sum'] = X[['Elevation',
                                  'Vertical_Distance_To_Hydrology']
    ].sum(axis='columns')

    X['Hydro_Elevation_diff'] = X[['Elevation',
                                   'Vertical_Distance_To_Hydrology']
                                ].diff(axis='columns').iloc[:, [1]]

    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology'] ** 2 +
                                   X['Vertical_Distance_To_Hydrology'] ** 2)

    X['Hydro_Manhattan'] = (X['Horizontal_Distance_To_Hydrology'] +
                            X['Vertical_Distance_To_Hydrology'].abs())

    X['Hydro_Distance_sum'] = X[['Horizontal_Distance_To_Hydrology',
                                 'Vertical_Distance_To_Hydrology']
    ].sum(axis='columns')

    X['Hydro_Distance_diff'] = X[['Horizontal_Distance_To_Hydrology',
                                  'Vertical_Distance_To_Hydrology']
                               ].diff(axis='columns').iloc[:, [1]]

    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Fire_Points']
    ].sum(axis='columns')

    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
                           ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
    ].mean(axis='columns')

    X['Hydro_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Fire_Points']
    ].median(axis='columns')

    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Roadways']
    ].sum(axis='columns')

    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
                           ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Road_mean'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
    ].mean(axis='columns')

    X['Hydro_Road_median'] = X[['Horizontal_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Roadways']
    ].median(axis='columns')

    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',
                            'Horizontal_Distance_To_Fire_Points']
    ].sum(axis='columns')

    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
                          ].diff(axis='columns').iloc[:, [1]].abs()

    X['Road_Fire_mean'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
    ].mean(axis='columns')

    X['Road_Fire_median'] = X[['Horizontal_Distance_To_Roadways',
                               'Horizontal_Distance_To_Fire_Points']
    ].median(axis='columns')

    X['Hydro_Road_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',
                                   'Horizontal_Distance_To_Roadways',
                                   'Horizontal_Distance_To_Fire_Points']
    ].mean(axis='columns')

    X['Hydro_Road_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',
                                     'Horizontal_Distance_To_Roadways',
                                     'Horizontal_Distance_To_Fire_Points']
    ].median(axis='columns')

    X['Hillshade_sum'] = X[['Hillshade_9am',
                            'Hillshade_Noon',
                            'Hillshade_3pm']
    ].sum(axis='columns')

    X['Hillshade_mean'] = X[['Hillshade_9am',
                             'Hillshade_Noon',
                             'Hillshade_3pm']
    ].mean(axis='columns')

    X['Hillshade_median'] = X[['Hillshade_9am',
                               'Hillshade_Noon',
                               'Hillshade_3pm']
    ].median(axis='columns')

    X['Hillshade_min'] = X[['Hillshade_9am',
                            'Hillshade_Noon',
                            'Hillshade_3pm']
    ].min(axis='columns')

    X['Hillshade_max'] = X[['Hillshade_9am',
                            'Hillshade_Noon',
                            'Hillshade_3pm']
    ].max(axis='columns')

    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?
    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1,
                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0,
                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4,
                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

    # Compute Soil_Type number from Soil_Type binary columns
    X['Stoneyness'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))

    # Replace Soil_Type number with "stoneyness" value
    X['Stoneyness'] = X['Stoneyness'].replace(range(1, 41), stoneyness)

    rocks = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
             1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 1, 1, 0, 1,
             0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    X['Rocks'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
    X['Rocks'] = X['Rocks'].replace(range(1, 41), rocks)

    return X


# noinspection PyPep8Naming
def get_wa_data(wilderness_area, X, y=None):
    """Get data for a Wilderness_Area

    Parameters
    ----------
    wilderness_area : int
        Wilderness Area

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    X_wa : array-like
        The subset of X that corresponds to the Wilderness Area
    y_wa : array-like
        The subset of y that corresponds to the Wilderness Area
    """
    X_wa = X[X['Wilderness_Area{}'.format(wilderness_area)] == 1]

    # Drop Wilderness_Area columns
    X_wa = X_wa.drop(['Wilderness_Area{}'.format(i) for i in range(1, 5)],
                     axis='columns')

    if y is not None:
        y_wa = y.loc[X_wa.index]
        return X_wa, y_wa

    return X_wa, None


# noinspection PyPep8Naming
def upsample(X, y):
    """Up-sample minority classes

    Parameters
    ----------
    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.


    Returns
    -------
    X : array-like
        Resampled X

    y : array-like, optional, default: None
        Resampled y

    """
    X = X.copy()
    y = y.copy()

    max_samples = y.value_counts().iat[0]
    classes = y.unique().tolist()
    sampling_strategy = dict((c, max_samples) for c in classes)

    sampler = SMOTE(sampling_strategy=sampling_strategy,
                    random_state=RANDOM_STATE)

    x_columns = X.columns.tolist()
    X, y = sampler.fit_resample(X, y)
    X = pd.DataFrame(X, columns=x_columns)
    y = pd.Series(y)

    return X, y


print('> Define constants')
print('  -- Set the features that worked well for other notebooks')
print('     Use features from \'Features engineering: improve\' v4 + including all Wilderness_Area:')
print('     {}'.format(FEATURES_SELECTED))

FEATURES_SELECTED = ['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                     'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type2', 'Soil_Type4', 'Soil_Type6', 'Soil_Type10',
                     'Soil_Type13', 'Soil_Type19', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
                     'Soil_Type26', 'Soil_Type30', 'Soil_Type32', 'Soil_Type34', 'Soil_Type40',
                     'Hydro_Elevation_sum', 'Hydro_Elevation_diff', 'Hydro_Distance_sum', 'Hydro_Distance_diff',
                     'Hydro_Fire_diff', 'Hydro_Fire_mean', 'Hydro_Fire_median', 'Hydro_Road_diff', 'Hydro_Road_mean',
                     'Hydro_Road_median', 'Road_Fire_diff', 'Road_Fire_mean', 'Road_Fire_median',
                     'Hydro_Road_Fire_mean', 'Hillshade_max', 'Stoneyness', 'Wilderness_Area1', 'Wilderness_Area2']

print('  -- Set model parameters')
RANDOM_STATE = 1
print('     random_state: {}'.format(RANDOM_STATE))
N_JOBS = -1
print('     n_jobs: {}'.format(N_JOBS))
N_ESTIMATORS = 2000
print('     n_estimators: {}'.format(N_ESTIMATORS))


print('> Prepare data')
print('  -- Read training and test files')
X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')

print('  -- Define the dependent variable')
y_train = X_train['Cover_Type'].copy()

print('  -- Define a training set')
X_train = X_train.drop(['Cover_Type'], axis='columns')

print('  -- Add new features')
X_train = add_features(X_train)
X_test = add_features(X_test)

print('> Set classifiers')
print('  -- AdaBoost')
ab_clf = AdaBoostClassifier(n_estimators=N_ESTIMATORS // 10,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=2,
                                random_state=RANDOM_STATE),
                            random_state=RANDOM_STATE)

print('  -- LightGBM')
lg_clf = LGBMClassifier(n_estimators=N_ESTIMATORS // 5,
                        num_leaves=100,
                        verbosity=0,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS)

print('  -- Random forest')
rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS // 5,
                                min_samples_leaf=1,
                                verbose=0,
                                random_state=RANDOM_STATE,
                                n_jobs=N_JOBS)

print('  -- Extra-trees')
xt_clf = ExtraTreesClassifier(n_estimators=N_ESTIMATORS,
                              random_state=RANDOM_STATE,
                              n_jobs=N_JOBS)

ensemble = [('ab', ab_clf),
            ('lg', lg_clf),
#             ('rf', rf_clf),
            ('xt', xt_clf)]

print('> Set stack')
stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],
                             meta_classifier=rf_clf,
                             cv=5,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=RANDOM_STATE,
                             n_jobs=N_JOBS)

print('> Fit & predict for each Wilderness_Area')

predictions = pd.Series(dtype=y_train.dtype)
wilderness_areas = ['Wilderness_Area{}'.format(i) for i in range(1, 5)]

for wa in range(1, 5):
    print('> Prepare data for Wilderness_Area{}'.format(wa))

    print('  -- Extract area data')
    X_train_wa, y_train_wa = get_wa_data(wa, X_train[FEATURES_SELECTED], y_train)
    X_test_wa, _ = get_wa_data(wa, X_test[FEATURES_SELECTED])

    # print('  -- Drop poor features')
    # X_train_wa = drop_features(X_train_wa)
    # X_test_wa = X_test_wa[X_train_wa.columns]

    print('  -- Up-sample for equal class counts')
    X_train_wa, y_train_wa = upsample(X_train_wa, y_train_wa)

    print('> Fit model for Wilderness_Area{}'.format(wa))
    stack = stack.fit(X_train_wa, y_train_wa)

    print('> Make predictions for Wilderness_Area{}'.format(wa))
    prediction_wa = stack.predict(X_test_wa)
    prediction_wa = pd.DataFrame(prediction_wa, index=X_test_wa.index)

    predictions = pd.concat([predictions, prediction_wa])

print('> Create submission')
predictions.to_csv('submission.csv', header=['Cover_Type'], index=True, index_label='Id')

print('> Done !')