import sys
import pip
import warnings
import pandas as pd
import numpy as np
import csv as csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore") 
#######################################################################################
#######################################################################################
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
Y_train = train['Survived']
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
#######################################################################################
#######################################################################################
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

X_test = DataFrameImputer().fit_transform(X_test)
X_train = DataFrameImputer().fit_transform(X_train)

le=LabelEncoder()
for col in X_test.columns.values:
       if X_test[col].dtypes=='object':
            data=X_train[col].append(X_test[col])
            le.fit(data.values)
            X_train[col]=le.transform(X_train[col])
            X_test[col]=le.transform(X_test[col])



X_train_scale=scale(X_train)
X_test_scale=scale(X_test)

clf = AdaBoostClassifier(n_estimators=100)
clf = clf.fit(X_train_scale, Y_train)
#print(accuracy_score(Y_train,clf.predict(X_train_scale)))

output = pd.DataFrame({'PassengerId' : test['PassengerId'].values.tolist(),
                        'Survived' : clf.predict(X_test_scale)})
output.to_csv('output.csv', index=False)








