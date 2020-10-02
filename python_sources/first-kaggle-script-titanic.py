import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import warnings
warnings.filterwarnings('ignore')

def harmonize_data(titanic):
    # Fill missing values of Age with median of all Age's
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    # Assign integer value for Sex (male/female = 0/1)
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    # Missing Embarked = S
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    # Assign integer value for Embarked (S/C/Q = 0/1/2)
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    # Fill missing values of Fare with median of all Fare's
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic


def create_submission(alg, train, test, predictors, filename):
    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(filename, index=False)


# start of script
def main():
    train = pd.read_csv("../input/train.csv", header=0)
    test = pd.read_csv("../input/test.csv", header=0)

    train_data = harmonize_data(train)
    test_data = harmonize_data(test)

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    alg = RandomForestClassifier(random_state=1,
                                 n_estimators=150,
                                 min_samples_split=4,
                                 min_samples_leaf=2)

    scores = cross_validation.cross_val_score(
        alg,
        train_data[predictors],
        train_data["Survived"],
        cv=3
    )

    print(scores.mean())

    create_submission(alg, train_data, test_data,
                      predictors, "submit.csv")


# Standard boilerplate to call the main() function to begin the program.
if __name__ == '__main__':
    main()
