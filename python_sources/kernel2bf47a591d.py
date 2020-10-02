# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#House-Prices

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#import graphviz
#import keras


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', sep=',')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', sep=',')


Id = test_df["Id"]

fulldata = [train_df, test_df]

for dataset in fulldata:
    dataset["Bath"] = dataset["BsmtFullBath"] + dataset["BsmtHalfBath"] +\
    dataset["FullBath"] + dataset["HalfBath"]
                  
for dataset in fulldata:
    dataset["Porch"] = dataset["OpenPorchSF"] + dataset["EnclosedPorch"] +\
    dataset["3SsnPorch"] + dataset["ScreenPorch"] + dataset["WoodDeckSF"]

for dataset in fulldata:
    dataset["Area"] = dataset["GrLivArea"] + dataset["TotalBsmtSF"]
    
train_df.to_csv("train1.csv", sep=",")
test_df.to_csv("test1.csv", sep=",")

#print(train_df.head(10))
print(type(train_df))


X_train = train_df[["MSSubClass","OverallQual", "OverallCond", "LotArea", "TotRmsAbvGrd", "GarageArea", 
                 "KitchenAbvGr", "PoolArea", "BedroomAbvGr", "Bath", "Porch", 
                 "Area", "YearRemodAdd"]]
Y_train = train_df[["SalePrice"]]
X_test = test_df[["MSSubClass","OverallQual", "OverallCond", "LotArea", "TotRmsAbvGrd", "GarageArea", 
               "KitchenAbvGr", "PoolArea", "BedroomAbvGr", "Bath", "Porch", 
               "Area", "YearRemodAdd"]]

print(X_test.isnull().sum(axis=0))

X_test["Bath"] = X_test["Bath"] .fillna(1)
X_test["Area"] = X_test["Area"].fillna(0)
X_test["GarageArea"] = X_test["GarageArea"].fillna(0)


"""
np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("Y_train.csv", Y_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")
"""
            
X_train.to_csv("X_train.csv", sep=",")
Y_train.to_csv("Y_train.csv", sep=",")
X_test.to_csv("X_test.csv", sep=",")

X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values


#Logistic Regression
logreg = LogisticRegression()
logreg = logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


#DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


#SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)


#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)


#GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


#Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


#SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)



# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


print("Logreg = ", acc_log)
print("DecisionTreeClassifier = ", acc_decision_tree)
print("SVM = ", acc_svc)
print("KNeighborsClassifier = ", acc_knn)
print("GaussianNB = ", acc_gaussian)
print("Perceptron = ", acc_perceptron)
print("LinearSVC = ", acc_linear_svc)
print("SGDClassifier = ", acc_sgd)
print("Random Forest = ", acc_random_forest)

result = decision_tree.predict(X_test)
submission = pd.DataFrame({"Id":test_df["Id"], "SalePrice":result})
submission.to_csv("submission.csv", index=False)


