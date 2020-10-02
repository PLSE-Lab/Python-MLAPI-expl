import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
def clean_data(sample):
    sample["Age"] = sample["Age"].fillna(sample["Age"].median())
    sample.loc[sample["Sex"] == 'female',"Sex"] = 0
    sample.loc[sample["Sex"] == 'male',"Sex"] = 1
    sample["Embarked"] = sample["Embarked"].fillna("S")
    for i, embarked in enumerate(('S','C','Q'),1): 
        sample.loc[sample["Embarked"] == embarked,"Embarked"] = i

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
print(".....")
print(test["PassengerId"], "test")
print("......")
#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#clean data 
clean_data(train)
clean_data(test)

#print(train["Embarked"])
#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
predictions = []
kf = KFold(train.shape[0],n_folds  = 3, random_state=True) 
alg= LinearRegression()
predictors  = ["Age", "Sex", "Parch", "SibSp", "Embarked","Fare","Pclass"]
for train2, test2 in kf: 
    print("working...")
    train_predictors = train[predictors].iloc[train2,:]
    train_target = train["Survived"].iloc[train2]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(train[predictors].iloc[test2,:])
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis = 0)
predictions[ predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
#accuracy = sum(predictions[train["Survived"] == predictions]) / len(predictions)
#print(accuracy)

alg = LogisticRegression(random_state = 1)

# Train the algorithm using all the training data
alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
del test["Cabin"]
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
nan_rows = test[test.isnull().T.any().T]
print(nan_rows)
print(nan_rows.describe())

predictions = alg.predict(test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index =False)
# log_predictions = []
# for train2, test in kf: 
#     train_predictors = train[predictors].iloc[train2,:]
#     train_target = train["Survived"].iloc[train2]
#     alg.fit(train_predictors, train_target)
#     test_predictions = alg.predict(train[predictors].iloc[test,:])
#     log_predictions.append(test_predictions)
# scores = cross_validation.cross_val_score(alg,train[predictors],train["Survived"],cv=3)
# print(scores.mean())
# #print(train[predictors].iloc[test,:])
# test_pred = alg.predict(train[predictors].iloc[test,:])
# print(test_pred)
# submission = pd.DataFrame({
#         "PassengerId": test["PassengerId"],
#         "Survived": test_pred
#     })
# submission.to_csv('titanic.csv', index=False)

