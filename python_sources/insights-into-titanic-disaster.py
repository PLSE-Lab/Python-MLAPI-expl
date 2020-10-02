import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())
print("----------------------------")

# Preview data

train.info()
print("----------------------------")
test.info()
print("----------------------------")

# Identifying required variables from dataset

train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test = test.drop(['Name', 'Ticket'], axis=1)

# Finding missing values

train["Embarked"] = train["Embarked"].fillna("S")
train.drop(['Embarked'], axis=1, inplace=True)
test.drop(['Embarked'], axis=1, inplace=True)

train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)

'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)   # axis = 0 > mean of column, axis = 1 > mean of row
imputer1 = imputer.fit(train['Age'])
train['Age'] = imputer1.transform(train['Age']).T
imputer2 = imputer.fit(test['Age'])
test['Age'] = imputer2.transform(test['Age']).T'''

test["Fare"].fillna(test["Fare"].median(), inplace=True)

train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)

train['Family'] = train['Parch'] + train['SibSp']
test['Family'] = test['Parch'] + test['SibSp']
train = train.drop(['SibSp','Parch'], axis=1)
test = test.drop(['SibSp','Parch'], axis=1)

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder
labelencoder1 = LabelEncoder()
train['Sex'] = labelencoder1.fit_transform(train['Sex']) # since only two categories not need to create dummy variables just convert to binary
test['Sex'] = labelencoder1.fit_transform(test['Sex']) # since only two categories not need to create dummy variables just convert to binary

dummy_class_train = pd.get_dummies(train['Pclass'])
dummy_class_train.columns = ['Class_1','Class_2','Class_3']
train = train.join(dummy_class_train)
train.drop(['Class_3'], axis=1, inplace=True) #avoid dummy trap
dummy_class_test = pd.get_dummies(test['Pclass'])
dummy_class_test.columns = ['Class_1','Class_2','Class_3']
test = test.join(dummy_class_test)
test.drop(['Class_3'], axis=1, inplace=True) #avoid dummy trap
train.drop(['Pclass'], axis=1, inplace=True)
test.drop(['Pclass'], axis=1, inplace=True)

# Seperating dependent and independent variables

X = train.drop('Survived', axis=1)
y = train['Survived']

# Splliting training data into two for training and validation and creating proper testing dataset

from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
X_test = test.drop(['PassengerId'], axis=1).copy()

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.fit_transform(X_val)
X_test = sc_X.transform(X_test)

# Creating classification Model

'''# Fitting Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)'''

'''# Fitting Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)'''

'''# Fitting KNN Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)'''

'''# Fitting SVM Model
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)'''

'''# Fitting Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)'''

# Fitting Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the validation set results
y_pred_val = classifier.predict(X_val)

# Making the Confusion Matrix for validation result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred_val)
val_accuracy = (cm[1,1]+cm[0,0])/len(X_val)
print('Validation Accuracy:' + str(val_accuracy))

# Predicting the test set results
y_pred_test = classifier.predict(X_test)

# Submitting the output
submission = pd.DataFrame({
             "PassengerId": test["PassengerId"],
             "Survived": y_pred_test
             })
submission.to_csv('Insights_into_titanic.csv', index=False)