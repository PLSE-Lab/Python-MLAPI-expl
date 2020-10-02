import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

combine = [train, test]

# Convert string values 'male' and 'female' to int values
sex_mapping = {'male': 0, 'female': 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

# Manage missing age data

# We try to guess missing age data using the correlation among Age, Gender, and Pclass.
# Guess Age values using median values for Age across sets of Pclass and Gender feature combinations.
# So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
guess_ages = np.zeros((2,3))

for dataset in combine:

    for sex in range(0, 2):
        for pclass in range(0, 3):
            guess_df = dataset[
                (dataset['Sex'] == sex) &
                (dataset['Pclass'] == pclass+1)
            ]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[sex, pclass] = int(age_guess/0.5 + 0.5) * 0.5
    
    for sex in range(0, 2):
        for pclass in range(0, 3):
            dataset.loc[
                (dataset.Age.isnull()) &
                (dataset.Sex == sex) &
                (dataset.Pclass == pclass+1),
                'Age'
            ] = guess_ages[sex, pclass]

#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'SibSp', 'Parch', 'Embarked'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch', 'Embarked'], axis=1)

X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test  = test.drop("PassengerId", axis=1)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

# Any results you write to the current directory are saved as output.

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': Y_pred
})

submission.to_csv('titanic_using_DTC.csv', index=False)