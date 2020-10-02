import numpy as np
import pandas as pd
#from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv") #, dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv") #, dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#print("\n\nTop of the testing data:")
#print(test.head())

train2 = train
test2 = test

# Convert the male and female groups to integer form
train2.loc[(train["Sex"] == "male"),"Sex"] = 0
train2.loc[(train["Sex"] == "female"), "Sex"] = 1

test2.loc[(test["Sex"] == "male"),"Sex"] = 0
test2.loc[(test["Sex"] == "female"), "Sex"] = 1

# Impute the Embarked variable
train2["Embarked"] = train["Embarked"].fillna("S")
train2["Age"] = train["Age"].fillna(train["Age"].median())
test2["Age"] = test["Age"].fillna(test["Age"].median())

# Convert the Embarked classes to integer form
train2.loc[(train["Embarked"] == "S"),"Embarked"] = 0
train2.loc[(train["Embarked"] == "C"),"Embarked"] = 1
train2.loc[(train["Embarked"] == "Q"),"Embarked"] = 2

test2.loc[(test["Embarked"] == "S"),"Embarked"] = 0
test2.loc[(test["Embarked"] == "C"),"Embarked"] = 1
test2.loc[(test["Embarked"] == "Q"),"Embarked"] = 2

# Impute the missing value with the median
test2.loc[152,"Fare"] = test["Fare"].median()

# Create the target and features numpy arrays: target, features_forest
target = train2["Survived"].values

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features = train2[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# split

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)


# scale

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# knn

knn = KNeighborsClassifier(n_neighbors = 15).fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test_scaled, y_test)))

# logistic regression

clf = LogisticRegression(C=1).fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

# Support Vector Classification

clf2 = SVC(C = .5).fit(X_train, y_train)
print('Accuracy of SVC no kernel classifier on training set: {:.2f}'.format(clf2.score(X_train, y_train)))
print('Accuracy of SVC no kernel classifier on test set: {:.2f}'.format(clf2.score(X_test, y_test)))

#clf3 = SVC(kernel = 'poly', degree = 2).fit(X_train, y_train)
#print('Accuracy of SVC poly kernel classifier on training set: {:.2f}'.format(clf3.score(X_train, y_train)))
#print('Accuracy of SVC poly kernel classifier on test set: {:.2f}'.format(clf3.score(X_test, y_test)))

clf4 = SVC(kernel = 'rbf', gamma = 0.01).fit(X_train, y_train)
print('Accuracy of SVC rbf kernel classifier on training set: {:.2f}'.format(clf4.score(X_train, y_train)))
print('Accuracy of SVC rbf kernel classifier on test set: {:.2f}'.format(clf4.score(X_test, y_test)))

# neural net

clfnn = MLPClassifier(hidden_layer_sizes = [10,100], alpha = 5.0, random_state = 0, solver = 'lbfgs').fit(X_train, y_train)
print('Accuracy of NN classifier on training set: {:.2f}'.format(clfnn.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clfnn.score(X_test, y_test)))

#print(test2["Embarked"].values)
#print(target)

# Building and fitting my_forest
#forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
#my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
#print(my_forest.score(features_forest, target))

#print(test2.describe())

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test2[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
#pred_forest = my_forest.predict(test_features)
#print(len(pred_forest))

clfsubmit = MLPClassifier(hidden_layer_sizes = [10,100], alpha = 5.0, random_state = 0, solver = 'lbfgs').fit(features, target)
submit_file = pd.DataFrame(test2, columns=["PassengerId","Survived"])
submit_file["Survived"] = clfsubmit.predict(test_features)


#print(type(test))
#print(type(pred_forest))

#print(submit_file)

#pred_forest.to_csv('copy_of_the_pred_forest.csv', index=False)

submit_file.to_csv("neural.csv", index = False)

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)