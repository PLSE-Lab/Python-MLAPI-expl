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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
warnings.filterwarnings("ignore") 
#######################################################################################
#######################################################################################
### 'PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp' , 'Parch', 'Ticket', 'Fare', 'Cabin' , 'Embarked'

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
train_0 = train[train['Survived'] == 0]
train_1 = train[train['Survived'] == 1]

frames = [train_0.iloc[1:340,],train_1.iloc[1:340,]]
train = pd.concat(frames)

train = train.iloc[np.random.permutation(len(train))]
X_train = train[['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp' , 'Parch', 'Ticket' , 'Fare', 'Cabin', 'Embarked']]
Y_train = train['Survived']
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
X_test = test[['PassengerId','Pclass', 'Name' , 'Sex', 'Age', 'SibSp' , 'Parch', 'Ticket' , 'Fare', 'Cabin', 'Embarked']]
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

X_temp_train = X_train_scale[1:580,]
Y_temp_train = Y_train[1:580,]

X_temp_test = X_train_scale[580:670,]
Y_temp_test = Y_train[580:670,] 
###########################################################
###########################################################
###########################################################
clf = RandomForestClassifier(n_estimators=250)
clf.fit(X_train_scale, Y_train)

#print(accuracy_score(Y_temp_test,clf.predict(X_temp_test)))
#print(cross_val_score(clf, X_temp_test, Y_temp_test).mean())


output = pd.DataFrame({'PassengerId' : test['PassengerId'].values.tolist(),
                        'Survived' : clf.predict(X_test_scale)})
output.to_csv('output.csv', index=False)