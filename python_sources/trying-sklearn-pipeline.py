# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        trans = X[self.columns].copy() 
        return trans

    def fit(self, X, y=None, **fit_params):
        return self

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
print(df.Species.unique())
y = df.Species
df = df.drop("Species",axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)

data_pipeline = Pipeline([
    ('select_col',SelectColumnsTransfomer(["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])),
    ('scaler',StandardScaler())
])

id_pipeline = Pipeline([
    ('select_col',SelectColumnsTransfomer(["Id"]))
    ])
    
preprocessing_features = FeatureUnion([
    ('id_feature', id_pipeline),
    ('other_feature', data_pipeline)
    ])
    
final_pipeline = Pipeline([
    ('preprocessing_features',preprocessing_features),
    ('Logistic_regression',LogisticRegression())
    ])

print(X_train.head())
final_pipeline.fit(X_train,y_train)
y_pred = final_pipeline.predict(X_test)

print(pd.DataFrame(y_pred))
print(accuracy_score(y_test, y_pred))

scores = cross_val_score(final_pipeline, df, y, cv=3)
print(scores)
print(scores.mean())

