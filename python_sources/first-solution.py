import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold

#Print you can execute arbitrary python code
titanic = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
titanic_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

alg = LinearRegression()

kf = KFold(titanic.shape[0],n_folds = 3,random_state = 1)

predictions = []

for train,test in kf:
    
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions,axis=0)

predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0



titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())

alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("first_solu.csv",index=False)
