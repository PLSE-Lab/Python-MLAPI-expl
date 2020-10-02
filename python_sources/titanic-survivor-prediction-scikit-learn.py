from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from time import time
import numpy as np
import pandas as pd


class PandasScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.columns = columns
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(self.copy, self.with_mean, self.with_std)

    def fit(self, X, y=None):
        self.scaler.fit(X[X.columns[~X.columns.isin(self.columns)]], y)

        return self

    def transform(self, X, y=None, copy=None):
        X_scaled = pd.DataFrame(self.scaler.transform(X[X.columns[~X.columns.isin(self.columns)]]),
                                columns=X.columns[~X.columns.isin(self.columns)], index=X.index)

        X_not_scaled = X.loc[:, X.columns[X.columns.isin(self.columns)]]

        X_transform = pd.concat([X_scaled, X_not_scaled], axis=1)

        bias = X_transform['1']
        X_transform.drop(labels=['1'], axis=1, inplace=True)
        X_transform.insert(0, '1', bias)

        return X_transform


class PandasPoly(BaseEstimator, TransformerMixin):
    def __init__(self, columns, degree=2, interaction_only=False, include_bias=True):
        self.columns = columns
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly = PolynomialFeatures(self.degree, self.interaction_only, self.include_bias)

    def fit(self, X, y=None):
        self.poly.fit(X[X.columns[~X.columns.isin(self.columns)]], y)

        return self

    def transform(self, X):
        X_poly = pd.DataFrame(self.poly.transform(X[X.columns[~X.columns.isin(self.columns)]]),
                              columns=self.poly.get_feature_names(), index=X.index)

        X_no_poly = X.loc[:, X.columns[X.columns.isin(self.columns)]]

        X_transform = pd.concat([X_poly, X_no_poly], axis=1)

        return X_transform


def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    index = X.index

    X.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)  # Drop columns that we won't use

    # Impute missing categorical data
    X["Embarked"].fillna(method="ffill", inplace=True)
    X["Embarked"].fillna(method="bfill", inplace=True)

    # One-hot encoding of categorical text features
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(X.to_dict(orient="records"))

    # Impute missing values
    imp = Imputer(strategy="mean")
    X = imp.fit_transform(X)

    X = pd.DataFrame(X, index=index, columns=dv.get_feature_names())

    return X


def kaggle_script():
    filepath = "../input/train.csv"

    data = pd.read_csv(filepath, index_col=0)

    data = data.sample(frac=1)

    X = data.drop("Survived", axis=1)  # Feature matrix
    y = data["Survived"]  # Target vector

    del data  # This DataFrame no longer needed

    X = clean_data(X)

    X1, X2, y1, y2 = train_test_split(X, y)  # Training and testing sets

    # Setup grid search. Specify features that should not be altered.
    pipe = make_pipeline(PandasPoly(columns=["Embarked=C", "Embarked=Q", "Embarked=S", "Sex=female", "Sex=male"]),
                         PandasScaler(columns=["1", "Embarked=C", "Embarked=Q", "Embarked=S", "Sex=female",
                                               "Sex=male"]),
                         LogisticRegression())

    rand_param_grid = dict(pandaspoly__degree=np.random.randint(low=2, high=6, size=1000),
                           logisticregression__C=np.random.uniform(low=0.001, high=1, size=1000))

    n_iter_search = 40
    rand_grid = RandomizedSearchCV(pipe, param_distributions=rand_param_grid, cv=5, n_jobs=-1, n_iter=n_iter_search)
    start = time()
    rand_grid.fit(X1, y1)

    print("\n")
    print("Optimal parameters as found by RandomizedSearchCV in {:0.2f} seconds:".format(time() - start))

    for param_name, param in rand_grid.best_params_.items():
        print("{}: {:.4f}".format(param_name, param))

    print("\n")

    # Logistic regression using default parameters with hold-out set
    lr = LogisticRegression()
    lr.fit(X1, y1)
    y2_pred = lr.predict(X2)
    print("Accuracy score for logistic regression using default parameter: {:.2f}".format(accuracy_score(y2, y2_pred)))

    # Logistic regression using RandomizedSearchCV optimal parameters with hold-out set
    y2_pred = rand_grid.predict(X2)
    print("Accuracy score for logistic regression using RandomizedSearchCV "
          "optimal parameters: {:.2f}".format(accuracy_score(y2, y2_pred)))

    # Kaggle Submission ================================================================================================
    filepath = "../input/test.csv"

    X_test = pd.read_csv(filepath, index_col=0)

    X_test = clean_data(X_test)

    index = X_test.index

    # Logistic regression using RandomizedSearchCV optimal parameters with Kaggle hold-out set
    y_test_pred = rand_grid.predict(X_test)

    # Output predictions to CSV file.
    filepath = "./predictions.csv"
    y_test_pred = pd.Series(y_test_pred, dtype="int32", index=index, name="Survived")
    y_test_pred.to_csv(filepath, header=True, index_label="PassengerId")


if __name__ == '__main__':
    kaggle_script()