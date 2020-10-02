import pandas as pd
import numpy as np
from scipy import stats

from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, Imputer, FunctionTransformer, LabelBinarizer, RobustScaler)
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LassoCV, RidgeCV

from xgboost import XGBRegressor


class ColumnSelector(TransformerMixin):
    def __init__(self, cols, subset):
        self.cols = cols
        self.subset = subset

    def transform(self, X, *_):
        col_indices = [i for i, c in enumerate(self.cols) if c in self.subset]
        return X[:, col_indices]

    def fit(self, *_):
        return self


class LogTransformSkewed(TransformerMixin):
    def __init__(self, tol=0.75):
        self.tol = tol

    def transform(self, X, *_):
        return np.apply_along_axis(func1d=self._log1p_skewed, axis=1, arr=X)

    def fit(self, *_):
        return self

    def _log1p_skewed(self, x):
        if stats.skew(x) > self.tol:
            return np.log1p(x)
        else:
            return x


def cv_split_generator(X, y, splitter):
    for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield i, X_train, X_val, y_train, y_val


def combined_model_weights(y_preds, y_vals, clfs, n_splits):
    w = []
    for i in range(n_splits):
        X = np.column_stack([y_preds[k][i] for k in clfs])
        weights = LassoCV().fit(X, y_vals[i]).coef_
        w.append(weights)
    return np.array(w)


classifiers = {
    'ransac_ridge_cv': RANSACRegressor(base_estimator=RidgeCV()),
    'xgb': XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        colsample_bytree=0.2,
        objective='reg:linear',
        reg_alpha=0.9,
        reg_lambda=0.6,
        gamma=0,
        nthread=-1,
    ),
}


train = pd.read_csv('../input/train.csv')

id_column = 'Id'
target = 'SalePrice'
y = train[target].values
y = np.log1p(y)

train = train.drop([target, id_column], axis=1)


f_numerical = [
    'LotArea', 'LotFrontage', 'GarageArea', 'MasVnrArea', 'LowQualFinSF',
    '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'BsmtHalfBath',
    'BsmtFullBath', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
    'BsmtFinSF1', 'BsmtFinSF2', 'YearBuilt', 'YearRemodAdd', 'BsmtUnfSF']

f_categorical = [
    'MSSubClass', 'MSZoning', 'LotShape', 'RoofStyle', 'Street',
    'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'Functional',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'MoSold', 'YrSold', 'SaleType',
    'SaleCondition']

f_included = f_numerical + f_categorical
f_excluded = [c for c in train.columns if c not in f_included]
print('Features excluded:\n%s\n' % f_excluded)


train_categorical = pd.get_dummies(train[f_categorical], drop_first=True)
train = pd.concat([train[f_numerical], train_categorical], axis=1)
del train_categorical

f_categorical_dummies = [c for c in train.columns if c not in f_numerical]

train_cols = train.columns.values
X = train.as_matrix()


pipeline_numerical = Pipeline([
    ('cols', ColumnSelector(train_cols, f_numerical)),
    ('imputer', Imputer(
        missing_values='NaN',
        strategy='mean',
        axis=0
    )),
    ('log1p_skewed', LogTransformSkewed()),
    # ('scaler', StandardScaler()),
    ('scaler', RobustScaler()),
    ('pca', PCA())
])

pipeline_categorical = Pipeline([
    ('catigorical_features', Pipeline([
        ('cols', ColumnSelector(train_cols, f_categorical_dummies)),
        ('imputer', Imputer(
            missing_values='NaN',
            strategy='most_frequent',
            axis=0
        )),
    ])),
])

features = Pipeline([
    ('all', FeatureUnion([
        ('numerical', pipeline_numerical),
        ('categorical', pipeline_categorical),
    ])),
])

X = features.fit_transform(X, y)


n_splits = 5
splitter = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=77)
cv_splits = cv_split_generator(X=X, y=y, splitter=splitter)

scores = []
measures = ['rmse', 'r2']
scores_cols = ['clf', 'split'] + measures

y_preds = defaultdict(lambda: defaultdict())  # predicted y's
y_vals = {}  # true y's

for i, X_train, X_val, y_train, y_val in cv_splits:
    for name, clf in classifiers.items():
        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
        r2 = metrics.r2_score(y_val, y_pred)

        y_preds[name][i] = y_pred
        y_vals[i] = y_val

        print('rmse = %.4f; r2 = %.4f | %s: %s' % (rmse, r2, i, name))
        scores.append([name, i, rmse, r2])


scores = pd.DataFrame(scores, columns=scores_cols)
scores = scores.sort_values(
    by=measures,
    ascending=[True, False]
).reset_index(drop=True)

scores_mean = scores.ix[:, scores.columns != 'split'] \
    .groupby(['clf']) \
    .mean() \
    .sort_values(by=measures, ascending=[True, False]) \
    .reset_index()

print('%s' % scores_mean)


# choose models: mix of techniques for coverage
clfs = ['ransac_ridge_cv', 'xgb']

# calculate weights for combined model from CV results
weights = combined_model_weights(y_preds, y_vals, clfs, n_splits)
print('models: %s\nweights: %s' % (clfs, weights))

# re-train chosen models on all data
models = {}
for name in clfs:
    models[name] = classifiers[name].fit(X, y)


# test preprocessing
test = pd.read_csv('../input/test.csv')
test_ids = test.pop(id_column)

test_categorical = pd.get_dummies(test[f_categorical], drop_first=True)
test = pd.concat([test[f_numerical], test_categorical], axis=1)
cols_missing = [c for c in train_cols if c not in test.columns]
for c in cols_missing:
    test[c] = np.zeros(len(test))
del test_categorical
test = test[train_cols]  # enforce correct column ordering for transform


# model prediction
X_test = features.transform(test.as_matrix())
y_tests = {}
for name, model in models.items():
    y_tests[name] = model.predict(X_test)

weights_mean = weights.mean(axis=0)
y_model_preds = np.row_stack([y_tests[k] for k in clfs])

y_test = np.dot(weights_mean, y_model_preds)  # multiply by weights
y_test = np.expm1(y_test)  # convert back to original form


# submission
submission = pd.DataFrame(y_test, index=test_ids, columns=[target])
submission.to_csv('submission_weighted.csv', index_label='Id')
