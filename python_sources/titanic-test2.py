import numpy as np
import pandas as pd

titanic = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(titanic.head())

#print("\n\nSummary statistics of training data")
#print(titanic.describe())

#~ VARIABLE DESCRIPTIONS:
#~ survival        Survival  (0 = No; 1 = Yes)
#~ pclass          Passenger Class  (1 = 1st; 2 = 2nd; 3 = 3rd)
#~ sibsp           Number of Siblings/Spouses Aboard
#~ parch           Number of Parents/Children Aboard
#~ ticket          Ticket Number
#~ embarked        Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

#~ SPECIAL NOTES:
#~ Pclass is a proxy for socio-economic status (SES):  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

#~ Age is in Years; Fractional if Age less than One (1). If the Age is Estimated, it is in the form xx.5

#~ With respect to the family relation variables (i.e. sibsp and parch)
#~ some relations were ignored.  The following are the definitions used for sibsp and parch:
#~  Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
#~  Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
#~  Parent:   Mother or Father of Passenger Aboard Titanic
#~  Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

#~ Other family relatives excluded from this study include cousins, nephews/nieces, aunts/uncles, and in-laws.  
#~ Some children travelled only with a nanny, therefore parch=0 for them.  As well, some travelled with very 
#~ close friends or neighbors in a village, however, the definitions do not support such relations.

#Any files you save will be available in the output tab below
#titanic.to_csv('copy_of_the_training_data.csv', index=False)

########## CLEANING #############
# ALL COLS MUST BE NUMERICAL IN ORDER FOR ML MODELS TO PREDICT

#Add a column to keep track of missing ages, as this might have some predicting importance by itself:
#eg for survivors it might have been easier to record the ages
titanic["AgeUnknown"] = titanic["Age"].isnull()
#convert bool to 0/1
titanic.AgeUnknown = titanic.AgeUnknown.astype(int)

#now we can clean age, by putting the median for missing values
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Based on name column, origin etc could be deducted. Get some parameters from it before dropping:
titanic["MrJames"] = (titanic["Name"].str.contains('Mr.', na=False)) & (titanic["Name"].str.contains('James', na=False))
titanic["MrAsian"] = (titanic["Name"].str.contains('Mr.', na=False)) & (
            (titanic["Name"].str.contains('Choong', na=False)) |
            (titanic["Name"].str.contains('Chang', na=False)) |
            (titanic["Name"].str.contains('Fang', na=False)) |
            (titanic["Name"].str.contains('Lee', na=False))
        )
# again, convert bools to 0/1
titanic.MrJames = titanic.MrJames.astype(int)
titanic.MrAsian = titanic.MrAsian.astype(int)


# Ignore Ticket, Cabin, and Name cols because they are not meaningfull
titanic.drop('Ticket', axis=1, inplace=True)
titanic.drop('Cabin', axis=1, inplace=True)
titanic.drop('Name', axis=1, inplace=True)

# Convert male to 0 and female to 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Convert embarked column to codes
titanic["Embarked"] = titanic["Embarked"].fillna("S") #S is most used port, so use it for the two null records
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

print(titanic.head(100))

###############################

# MODEL to use: LINEAIR REGRESSION
# use cross validation to test accurracy

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "AgeUnknown", "MrJames", "MrAsian"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# PREDICT
predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

#### Now evaluate how well we did ###

# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy) # => 79.3% (not so good)

# TRY OTHER MODEL: LOGISTIC REGRESSION

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())






