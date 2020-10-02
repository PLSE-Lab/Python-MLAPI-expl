# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import theano
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import Imputer


dataset = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv/train.csv")
X_test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

# Importing the dataset and define X and Y


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].value_counts()
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in X_test["Name"]]
X_test["Title"] = pd.Series(dataset_title)
X_test["Title"].value_counts()
X_test["Title"] = X_test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')

dataset["FamilyS"] = dataset["SibSp"] + dataset["Parch"] + 1
X_test["FamilyS"] = X_test["SibSp"] + X_test["Parch"] + 1

def family(x):
    if x < 2:
        return "Single"
    elif x <= 2:
        return "Couple"
    elif x <= 4:
        return "InterM"
    else:
        return "Large"

dataset["FamilyS"] = dataset["FamilyS"].apply(family)
X_test["FamilyS"] = X_test["FamilyS"].apply(family)


dataset = dataset.drop(["PassengerId","Name", "Ticket", "Embarked", "SibSp", "Parch", "Cabin"], axis=1)
X_test = X_test.drop(["PassengerId", "Name", "Ticket", "Embarked", "SibSp", "Parch", "Cabin"], axis=1)

X_train = dataset.iloc[:, 1:8].values
y_train = dataset.iloc[:, 0].values
X_test = X_test.values

####
imputer = Imputer(missing_values="NaN", strategy = "median", axis = 0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])
imputer = imputer.fit(X_train[:, 3:4])
X_train[:, 3:4] = imputer.transform(X_train[:, 3:4])

imputer = imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])
imputer = imputer.fit(X_test[:, 3:4])
X_test[:, 3:4] = imputer.transform(X_test[:, 3:4])

## Male and Female are still labeled, so first they have to be labelled to numbers.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_train[:, 1] = labelencoder_X_1.fit_transform(X_train[:, 1])
X_train[:, 4] = labelencoder_X_1.fit_transform(X_train[:, 4])
X_train[:, 5] = labelencoder_X_1.fit_transform(X_train[:, 5])

labelencoder_X_2 = LabelEncoder()
X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])
X_test[:, 5] = labelencoder_X_2.fit_transform(X_test[:, 5])

### Since Neural Networks can not handle categorical data, i will use dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 4, 5])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 4, 5])
X_test = onehotencoder.fit_transform(X_test).toarray()

## To not let one feature dominate another, i scale the features i the test and training set.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 14:15] = sc.fit_transform(X_train[:, 14:15])
X_train[:, 15:16] = sc.fit_transform(X_train[:, 15:16])
X_test[:, 14:15] = sc.fit_transform(X_test[:, 14:15])
X_test[:, 15:16] = sc.fit_transform(X_test[:, 15:16])


# Buildung the ANN 
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN with the parameters which were set before
classifier.fit(X_train, y_train, batch_size = 25, epochs = 250)

# Predicting the Test set results, using the 0.5 threshold and get true or false values to build a confusion matrix.
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)

### Converting the prediction to a 0/1 value at a 0.5 threshold.
def binary(x):
    if x < 0.5:
        return 0
    else:
        return 1

y_pred = y_pred[0].apply(binary)

### Since i deleted the dataset above, i load it again to get the PassengerId.

Ids = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

submission = pd.DataFrame({
        "PassengerId": Ids["PassengerId"],
        "Survived": y_pred
    })
    
submission.to_csv('titanic.csv', index=False)