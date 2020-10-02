import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, Imputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#Importing the data
training_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#Preparing the data
feature_cols_list = ["Pclass", "Age", "Sex", "SibSp", "Parch"]
X_train = training_data[feature_cols_list]
y_train = training_data.Survived
X_test = test_data[feature_cols_list]
labelencoder = LabelEncoder()
X_train.loc[:, "Sex"] = labelencoder.fit_transform(X_train.loc[:, "Sex"])
X_train.loc[:, "Age"] = (X_train.loc[:, "Age"]-X_train.loc[:, "Age"].min())/(X_train.loc[:, "Age"].max()-X_train.loc[:, "Age"].min())
X_test.loc[:, "Sex"] = labelencoder.fit_transform(X_test.loc[:, "Sex"])
X_test.loc[:, "Age"]=(X_test.loc[:, "Age"]-X_test.loc[:, "Age"].min())/(X_test.loc[:, "Age"].max()-X_test.loc[:, "Age"].min())
imputer = Imputer(missing_values="NaN", strategy = "mean")
imputer = imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


xgb = XGBClassifier(n_estimators = 100000, learning_rate = 0.01)
xgb.fit(X_train, y_train, verbose = True)
predictions = xgb.predict(X_test)

# print(accuracy_score(y_train, predictions))

# #Submission
my_submission = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
my_submission.to_csv("submission.csv", index = False)
    
