# Kaggle Titanic predictions

# Importing libraries
import numpy as np
import pandas as pd

# Making the splits
training = pd.read_csv('../input/train.csv')
X_train = training.iloc[:, [2, 4, 5, 6, 7, 9]].values
y_train = training.iloc[:, 1].values
testing = pd.read_csv('../input/test.csv')
X_test = testing.iloc[:, [1, 3, 4, 5, 6, 8]].values

# Reshaping to a matrix
X_train = X_train.reshape(-1, 6)
X_test = X_test.reshape(-1, 6)

# Filling in missing data
from sklearn.preprocessing import Imputer
train_imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
train_imputer = train_imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = train_imputer.transform(X_train[:, 2:3])

test_imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
test_imputer = test_imputer.fit(X_test[:, 2:6])
X_test[:, 2:6] = test_imputer.transform(X_test[:, 2:6])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder_X.transform(X_test[:, 1])

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 300, criterion = 'entropy')
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)