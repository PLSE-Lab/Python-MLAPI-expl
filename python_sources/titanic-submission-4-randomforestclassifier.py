import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

test_id = test['PassengerId']
# drop not used feature, but ticket may mean something
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test = test.drop(['Name','Ticket', 'PassengerId'], axis=1)

# Embarked
train['Embarked'] = train['Embarked'].fillna("S")
test['Embarked'] = test['Embarked'].fillna('S')
embarked_dummies_train = pd.get_dummies(train['Embarked'])
embarked_dummies_test = pd.get_dummies(test['Embarked'])

train = train.join(embarked_dummies_train)
test = test.join(embarked_dummies_test)

train.drop('Embarked', axis=1, inplace=True)
test.drop('Embarked', axis=1, inplace=True)

# Fare
train['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Age
avg_age_train = train['Age'].mean()
std_age_train = train['Age'].std()
count_age_nan_train = train['Age'].isnull().sum()


avg_age_test = test['Fare'].mean()
std_age_test = test['Fare'].std()
count_age_nan_test = test['Age'].isnull().sum()

rand_1 = np.random.randint(avg_age_train - std_age_train, avg_age_train + std_age_train, size = count_age_nan_train)
rand_2 = np.random.randint(avg_age_test - std_age_test, avg_age_test + std_age_test, size = count_age_nan_test)

train['Age'][np.isnan(train["Age"])] = rand_1
test['Age'][np.isnan(test["Age"])] = rand_2

# Cabin
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# Family
train['Family'] = train['SibSp'] + train['Parch']
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] <= 0 ] = 0

test['Family'] = test['SibSp'] + test['SibSp']
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] <= 0 ] = 0

train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Sex
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

train['Person'] = train[['Age', 'Sex']].apply(get_person, axis=1);
test['Person'] = test[['Age', 'Sex']].apply(get_person, axis=1);

person_dummies_train = pd.get_dummies(train['Person'])
person_dummies_test = pd.get_dummies(train['Person'])

train = train.join(person_dummies_train)
test = test.join(person_dummies_test)

train.drop(['Sex', 'Person'], axis=1, inplace=True)
test.drop(['Sex', 'Person'], axis=1, inplace=True)

# Pclass
pclass_dummies_train = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['class_0', 'class_1', 'class_2']

pclass_dummies_test = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['class_0', 'class_1', 'class_2']

train = train.join(pclass_dummies_train)
test = test.join(pclass_dummies_test)

train.drop('Pclass', axis=1, inplace=True)
test.drop('Pclass', axis=1, inplace=True)

# scale
from sklearn.ensemble import RandomForestClassifier
X_train = train.drop(['Survived','S', 'Q', 'C', 'Family', 'male'],axis=1)
y_train = train['Survived']

X_test = test.drop(['S', 'Q', 'C', 'Family', 'male'], axis=1)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_test = random_forest.predict(X_test)
print(random_forest.score(X_train, y_train))

submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": y_test
    })
submission.to_csv('titanic.csv', index=False)