#import numpy as np
#import pandas as pd

#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)


import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold

def read_data(titanic_file):
    # Load the csv file
    data = pd.read_csv(titanic_file, dtype={"Age": np.float64}, )
    # print the data description       
    print ("\tOk\nData description...")
    print (data.describe())               # Data description
    
    # average the 'Age' column
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    
    # Replace all the occurences of male with the number 0.
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    
    # Find all the unique values for "Embarked".
    #print(data["Embarked"].unique())
    
    # Assume, that all missed "Embarked"-labels are "S", so
    # replace all empty marks with "S" symbol
    data["Embarked"] = data["Embarked"].fillna("S")
    
    
    # Replace the string values of "Embarked" with the order:
    # "S" == 0, "C" == 1, "Q" == 2
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
    
       
    return data

def LinearReg (data, predictors):
    # Import the linear regression class
    from sklearn.linear_model import LinearRegression


# Initialize our algorithm class
    alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
    kf = KFold(data.shape[0], n_folds=3, random_state=1)

    predictions = []
    for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
        train_predictors = (data[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
        train_target = data["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
        alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
        test_predictions = alg.predict(data[predictors].iloc[test,:])
        predictions.append(test_predictions)
    
# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
    predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <=.5] = 0
    accuracy = sum(predictions[predictions == data["Survived"]]) / len(predictions)
   
    
    return alg
    
    

def LogReg(data, predictors):
    from sklearn import cross_validation
    from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
    alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
    scores = cross_validation.cross_val_score(alg, data[predictors], data["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
    print(scores.mean())
    
    alg.fit(data[predictors], data["Survived"])


    return alg
    
# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    
print ("Let's start the disaster problem")
print ("Read the training data..."),


# read train data
titanic_train = read_data("../input/train.csv")

print ("Read test data..."),
titanic_test = read_data("../input/test.csv")

# teaching with LinearRegression
print ("First approach with LinearRegression..."),
LinearReg(titanic_train, predictors)
# teaching with Logistik Regression
print ("Second approach with LogisticaRegression..."),
titanic_LogReg = LogReg(titanic_train, predictors)

predictions = titanic_LogReg.predict(titanic_test[predictors])


# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })