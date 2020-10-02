import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

from xgboost import XGBClassifier

# Import Data
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Map sex, port of embarkation to numeric values
mapping_sex = {'female':0,'male':1}
mapping_embarked = {'Q':0,'S':1,'C':2}

train.replace({'Sex':mapping_sex},inplace=True)
# train.dropna(axis=0,subset=['Age'],inplace=True)
# train.dropna(axis=0,subset=['Embarked'],inplace=True)
train.replace({'Embarked':mapping_embarked},inplace=True)

# Create DF with only numeric values for data, use imputer to replace NaNs with mean
X_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
imp = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
X_train = imp.fit_transform(X_train)
y_train = train[['Survived']]

# Scale data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# Cross Validation
# train_data, test_data, train_target, test_target = train_test_split(X_train,y_train,test_size=0.3,random_state=0)


# Trying a bunch of different classifiers
# clf = LinearSVC().fit(train_data,train_target.values.ravel())
# clf = tree.DecisionTreeClassifier().fit(train_data,train_target.values.ravel())
# clf = KNeighborsClassifier().fit(train_data,train_target.values.ravel())
# clf = DecisionTreeClassifier(max_depth=5).fit(train_data,train_target.values.ravel())
# clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(train_data,train_target.values.ravel())
# clf = MLPClassifier(alpha=1).fit(train_data,train_target.values.ravel())
# clf = GaussianNB().fit(train_data,train_target.values.ravel())
# clf = SVC(gamma=2, C=1).fit(train_data,train_target.values.ravel())
# clf = AdaBoostClassifier().fit(train_data,train_target.values.ravel())
clf = XGBClassifier().fit(X_train,y_train.values.ravel())

test.replace({'Sex':mapping_sex},inplace=True)
# test.dropna(axis=0,subset=['Age','Fare','Embarked'],inplace=True)
# # test.dropna(axis=0,subset=['Embarked'],inplace=True)
test.replace({'Embarked':mapping_embarked},inplace=True)

X_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
X_test = imp.fit_transform(X_test)
X_test = scaler.transform(X_test)
# X_test.to_csv('xtest.csv',sep=',')
# # y_test = test[['Survived']]

# X_test.is_copy

# # X_test.dropna(axis=1,inplace=True)

# clf = LinearSVC()
# clf.fit(X_train,y_train.values.ravel())
results = clf.predict(X_test)

print(results)


# #Print to standard output, and see the results in the "log" section below after running your script
# # print("\n\nTop of the training data:")
# # print(train.head(15))

# # print("\n\nSummary statistics of training data")
# # print(train.describe())

# #Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)