import numpy as np

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, Imputer
from patsy import dmatrices, dmatrix

#Print you can execute arbitrary python code
df_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
df_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Drop NaNs
df_train.dropna(subset=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], inplace=True)

print("\n\nSummary statistics of training data")
print(df_train.describe())

# Age imputation
df_train.loc[df_train['Age'].isnull(), 'Age'] = np.nanmedian(df_train['Age'])
df_test.loc[df_test['Age'].isnull(), 'Age'] = np.nanmedian(df_test['Age'])

# Training/testing array creation
y_train, X_train = dmatrices('Survived ~ Age + Sex + Pclass + SibSp + Parch + Embarked', df_train)
X_test = dmatrix('Age + Sex + Pclass + SibSp + Parch + Embarked', df_test)

# Creating processing pipelines with preprocessing. Hyperparameters selected using cross validation
steps1 = [('poly_features', PolynomialFeatures(3, interaction_only=True)),
          ('logistic', LogisticRegression(C=5555., max_iter=16, penalty='l2'))]
steps2 = [('rforest', RandomForestClassifier(min_samples_split=15, n_estimators=73, criterion='entropy'))]
pipeline1 = Pipeline(steps=steps1)
pipeline2 = Pipeline(steps=steps2)

# Logistic model with cubic features
pipeline1.fit(X_train, y_train.ravel())
print('Accuracy (Logistic Regression-Poly Features (cubic)): {:.4f}'.format(pipeline1.score(X_train, y_train.ravel())))

# Random forest with calibration
pipeline2.fit(X_train[:600], y_train[:600].ravel())
calibratedpipe2 = CalibratedClassifierCV(pipeline2, cv=3, method='sigmoid')
calibratedpipe2.fit(X_train[600:], y_train[600:].ravel())
print('Accuracy (Random Forest - Calibration): {:.4f}'.format(calibratedpipe2.score(X_train, y_train.ravel())))

# Create the output dataframe
output = pd.DataFrame(columns=['PassengerId', 'Survived'])
output['PassengerId'] = df_test['PassengerId']

# Predict the survivors and output csv
output['Survived'] = pipeline1.predict(X_test).astype(int)
output.to_csv('output.csv', index=False)