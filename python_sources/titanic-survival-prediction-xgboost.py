# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Importing the dataset
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_data = [df_train, df_test]

PID = df_test['PassengerId']

# Creating new feature FamilySize from SibSp and Parch
for dataset in df_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Creating new feature IsAlone from FamilySize
for dataset in df_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Creating new feature HasCabin from Cabin
for dataset in df_data:
    dataset['HasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

# Creating new feature Title from Name
for dataset in df_data:    
    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(', ')[1])
    dataset['Title'] = dataset['Title'].apply(lambda x: x.split('.')[0])

# Filling missing data
for dataset in df_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    dataset['Age'] = dataset['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std))

# Categorising and encoding Age
for dataset in df_data:    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Categorising and encoding Fare
for dataset in df_data:    
    dataset.loc[dataset['Fare'] == 0, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 0) & (dataset['Fare'] <= 50), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 200), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 200) & (dataset['Fare'] <= 500), 'Fare'] = 4
    dataset.loc[dataset['Fare'] > 500, 'Fare'] = 5

# Categorising and encoding Title
for dataset in df_data:
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'the Countess', 'Sir', 'Lady', 'Capt', 'Don', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map({'Mr': 0,
                                             'Miss': 1,
                                             'Mrs': 2,
                                             'Master': 3,
                                             'Rare': 4})

# Encoding categorical data
for dataset in df_data:
    dataset['Sex'] = dataset['Sex'].map({'male': 0,
                                       'female': 1}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0,
                                                   'C': 1,
                                                   'Q': 2}).astype(int)

# Feature Selection
drop_elements = ['PassengerId',
                 'Name',
                 'Ticket',
                 'Cabin',
                 'SibSp',
                 'Parch']
df_train = df_train.drop(drop_elements, axis = 1)
df_test = df_test.drop(drop_elements, axis = 1)

# Creating Training and Test set arrays
X_train = df_train.iloc[:, 1:].values
X_test = df_test.values
y_train = df_train.iloc[:, 0].values

# Using One Hot Encoder to remove ordering of classes in Embarked and Title
onehotencoder = OneHotEncoder(categorical_features = [4, 8])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

# Avoiding dummy variable trap
X_train = np.delete(X_train, [0, 3], 1)
X_test = np.delete(X_test, [0, 3], 1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting different classifiers
log = LogisticRegression()
log.fit(X_train, y_train)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
svm = SVC()
svm.fit(X_train, y_train)
nb = GaussianNB()
nb.fit(X_train, y_train)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Applying k-Fold Cross Validation to all classifiers
col = ['Classifier', 'Mean', 'Standard Deviation']
accuracy = pd.DataFrame(index = range(0,7), columns = col)
accuracy['Classifier'] = ['Logistic Regression',
                        'K-NN',
                        'SVM',
                        'Naive Bayes',
                        'Decision Tree',
                        'Random Forest',
                        'XGBoost']
acc_mean = []
acc_std = []
score_log = cross_val_score(estimator = log, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_log.mean())
acc_std.append(score_log.std())
score_knn = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_knn.mean())
acc_std.append(score_knn.std())
score_svm = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_svm.mean())
acc_std.append(score_svm.std())
score_nb = cross_val_score(estimator = nb, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_nb.mean())
acc_std.append(score_nb.std())
score_dt = cross_val_score(estimator = dt, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_dt.mean())
acc_std.append(score_dt.std())
score_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_rf.mean())
acc_std.append(score_rf.std())
score_xgb = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
acc_mean.append(score_xgb.mean())
acc_std.append(score_xgb.std())
accuracy['Mean'] = acc_mean
accuracy['Standard Deviation'] = acc_std
print(accuracy)

# Predicting the Test set results
y_pred = xgb.predict(X_test)

# Generate Submission File
submission = pd.DataFrame({'PassengerId': PID, 'Survived': y_pred})
submission.to_csv('result.csv', index=False)