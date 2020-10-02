

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the training dataset
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:,1].values

# Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,2:3])
X[:, 2:3] = imputer.transform(X[:,2:3])
df1 = pd.DataFrame(X)

# Encode the categorical data

# Encode the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# label encode gender
X[:,1] = labelencoder_X.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
df2 = pd.DataFrame(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
df3 = pd.DataFrame(X)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Import the test dataset
dataset_test = pd.read_csv('../input/test.csv')
X_test = dataset_test.iloc[:, [1,3,4,5,6,8]].values
df1_test = pd.DataFrame(X_test)

# Take care of missing data - age
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X_test[:,2:3])
X_test[:, 2:3] = imputer.transform(X_test[:,2:3])
df2_test = pd.DataFrame(X_test)

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X_test[:,5:6])
X_test[:, 5:6] = imputer.transform(X_test[:,5:6])
df3_test = pd.DataFrame(X_test)

# label encode gender
labelencoder_X_test = LabelEncoder()
X_test[:,1] = labelencoder_X_test.fit_transform(X_test[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
df4_test = pd.DataFrame(X_test)

# Avoiding the Dummy Variable Trap
X_test = X_test[:, 1:]
df4_test = pd.DataFrame(X_test)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

y_pred_rounded = np.around(y_pred).astype(np.uint64)
#y_pred_rounded = np.round(y_pred,0))

submission = pd.DataFrame({
        "PassengerId": dataset_test["PassengerId"],
        "Survived": y_pred_rounded
    })
submission.to_csv('submission.csv', index=False)










