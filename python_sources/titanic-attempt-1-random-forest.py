# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')

X_train = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = dataset[['Survived']]

X_test = dataset_test
X_test = X_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


# Taking care of missing data
from sklearn.preprocessing import Imputer

# X_train
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X_train[['Age']])
X_train[['Age']] = imputer.transform(X_train[['Age']])
X_train['Embarked'] = X_train['Embarked'].fillna("S")

# X_test
imputer2 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer2 = imputer2.fit(X_test[['Age']])
X_test[['Age']] = imputer2.transform(X_test[['Age']])
imputer3 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer3 = imputer3.fit(X_test[['Fare']])
X_test[['Fare']] = imputer2.transform(X_test[['Fare']])
X_test['Embarked'] = X_test['Embarked'].fillna("S")

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# Encode 'Embarked' and 'Sex' for X_train
X_train[['Embarked']] = labelencoder_X.fit_transform(X_train[['Embarked']])
labelencoder_X2 = LabelEncoder()
X_train[['Sex']] = labelencoder_X2.fit_transform(X_train[['Sex']])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:,1:]

# Encode 'Embarked' and 'Sex' for X_test
labelencoder_X3 = LabelEncoder()
X_test[['Embarked']] = labelencoder_X3.fit_transform(X_test[['Embarked']])
labelencoder_X4 = LabelEncoder()
X_test[['Sex']] = labelencoder_X4.fit_transform(X_test[['Sex']])
onehotencoder2 = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder2.fit_transform(X_test).toarray()
X_test = X_test[:,1:]

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5000, random_state = 0)
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
y_submission = np.where(y_pred[:] > 0.95, 1, 0)

submission = pd.DataFrame({
            'PassengerId': dataset_test['PassengerId'],
            'Survived': y_submission
            })

submission.to_csv('titanic.csv', index=False)