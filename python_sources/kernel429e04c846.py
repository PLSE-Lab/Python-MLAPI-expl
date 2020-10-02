import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
Test = test
#
train = train.drop(['PassengerId'], axis=1)
test = test.drop(['PassengerId'], axis=1)

def get_titles():
    global combined
    train['Title'] = train['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    test['Title'] = test['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    train['Title'] = train.Title.map(Title_Dictionary)
    test['Title'] = test.Title.map(Title_Dictionary)

get_titles()

median_age = (train.groupby(['Title'])['Age'].agg('mean'))
#print(median_age)
train.loc[(train['Title'] == 'Master'), 'Age'] = train.loc[(train['Title'] == 'Master'), 'Age'].fillna(median_age['Master'])
train.loc[(train['Title'] == 'Miss'), 'Age'] = train.loc[(train['Title'] == 'Miss'), 'Age'].fillna(median_age['Miss'])
train.loc[(train['Title'] == 'Mr'), 'Age'] = train.loc[(train['Title'] == 'Mr'), 'Age'].fillna(median_age['Mr'])
train.loc[(train['Title'] == 'Mrs'), 'Age'] = train.loc[(train['Title'] == 'Mrs'), 'Age'].fillna(median_age['Mrs'])
train.loc[(train['Title'] == 'Officer'), 'Age'] = train.loc[(train['Title'] == 'Officer'), 'Age'].fillna(median_age['Officer'])
train.loc[(train['Title'] == 'Royalty'), 'Age'] = train.loc[(train['Title'] == 'Royalty'), 'Age'].fillna(median_age['Royalty'])
          
test.loc[(test['Title'] == 'Master'), 'Age'] = test.loc[(test['Title'] == 'Master'), 'Age'].fillna(median_age['Master'])
test.loc[(test['Title'] == 'Miss'), 'Age'] = test.loc[(test['Title'] == 'Miss'), 'Age'].fillna(median_age['Miss'])
test.loc[(test['Title'] == 'Mr'), 'Age'] = test.loc[(test['Title'] == 'Mr'), 'Age'].fillna(median_age['Mr'])
test.loc[(test['Title'] == 'Mrs'), 'Age'] = test.loc[(test['Title'] == 'Mrs'), 'Age'].fillna(median_age['Mrs'])
test.loc[(test['Title'] == 'Officer'), 'Age'] = test.loc[(test['Title'] == 'Officer'), 'Age'].fillna(median_age['Officer'])
test.loc[(test['Title'] == 'Royalty'), 'Age'] = test.loc[(test['Title'] == 'Royalty'), 'Age'].fillna(median_age['Royalty'])

train.loc[(train['Age'] < 10) & (train['Age'] >= 0), 'AgeBucket'] = 1
train.loc[(train['Age'] < 20) & (train['Age'] >= 10), 'AgeBucket'] = 2
train.loc[(train['Age'] < 30) & (train['Age'] >= 20), 'AgeBucket'] = 3
train.loc[(train['Age'] < 40) & (train['Age'] >= 30), 'AgeBucket'] = 4
train.loc[(train['Age'] < 50) & (train['Age'] >= 40), 'AgeBucket'] = 5
train.loc[(train['Age'] < 60) & (train['Age'] >= 50), 'AgeBucket'] = 6
train.loc[(train['Age'] < 70) & (train['Age'] >= 60), 'AgeBucket'] = 7
train.loc[(train['Age'] < 80) & (train['Age'] >= 70), 'AgeBucket'] = 8
train.loc[(train['Age'] < 90) & (train['Age'] >= 80), 'AgeBucket'] = 9
train.loc[(train['Age'] < 100) & (train['Age'] >= 90), 'AgeBucket'] = 10
         
test.loc[(test['Age'] < 10) & (test['Age'] >= 0), 'AgeBucket'] = 1
test.loc[(test['Age'] < 20) & (test['Age'] >= 10), 'AgeBucket'] = 2
test.loc[(test['Age'] < 30) & (test['Age'] >= 20), 'AgeBucket'] = 3
test.loc[(test['Age'] < 40) & (test['Age'] >= 30), 'AgeBucket'] = 4
test.loc[(test['Age'] < 50) & (test['Age'] >= 40), 'AgeBucket'] = 5
test.loc[(test['Age'] < 60) & (test['Age'] >= 50), 'AgeBucket'] = 6
test.loc[(test['Age'] < 70) & (test['Age'] >= 60), 'AgeBucket'] = 7
test.loc[(test['Age'] < 80) & (test['Age'] >= 70), 'AgeBucket'] = 8
test.loc[(test['Age'] < 90) & (test['Age'] >= 80), 'AgeBucket'] = 9
test.loc[(test['Age'] < 100) & (test['Age'] >= 90), 'AgeBucket'] = 10
         
#print(train.groupby(['AgeBucket'])['Survived'].agg(['count']))

train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)
train = pd.concat([train, pd.get_dummies(train['Sex'])], axis = 1 )
test = pd.concat([test, pd.get_dummies(test['Sex'])], axis = 1 )
train = train.drop(['Sex'], axis=1)
test = test.drop(['Sex'], axis=1)
train = train.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Title', 'male'], axis=1)
test = test.drop(['Name', 'Ticket', 'Fare', 'Cabin','Title', 'male'], axis=1)

train['family_size'] = train['SibSp'] + train['Parch']
test['family_size'] = test['SibSp'] + test['Parch']
#print(train.groupby(['family_size'])['Survived'].agg(['count']))
train.loc[train['family_size'] > 1, 'isAlone'] = 0
train.loc[train['family_size'] <= 1, 'isAlone'] = 1
test.loc[test['family_size'] > 1, 'isAlone'] = 0
test.loc[test['family_size'] <= 1, 'isAlone'] = 1
 
train = train.drop(['SibSp', 'Parch', 'family_size'], axis=1)
test = test.drop(['SibSp', 'Parch', 'family_size'], axis=1)

train['Embarked'].fillna('S')
train = pd.concat([train, pd.get_dummies(train['Embarked'])], axis = 1 )
test = pd.concat([test, pd.get_dummies(test['Embarked'])], axis = 1 )
train = train.drop(['Embarked', 'S'], axis=1)
test = test.drop(['Embarked', 'S'], axis=1)

print(train.describe())

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
"""
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_traain)
Y_pred = knn.predict(test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
"""
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

submission = pd.DataFrame({
        "PassengerId": Test["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('output.csv', index=False)