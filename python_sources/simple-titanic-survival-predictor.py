'''
#Project: Simple Titanic Survival Predictor
#Author: Wessel van Lit
#Description: A program that tries to predict if a person surives on the Titanic based on certain factors like Age, Sex, Class, etc.
The program uses Machine Learning to calculate its predictions.
'''
# Import the libraries needed for this program
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import warnings

# Terminal Setup
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.expand_frame_repr', False)
# Add File Paths for the datasets

TestFilePath = '../input/test.csv'
TrainFilePath = '../input/train.csv'

# Assign the datasets to variables
testData = pd.read_csv(TestFilePath)
trainData = pd.read_csv(TrainFilePath)

# Create X & y for predicting the surival of passengers
cols_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
X = trainData[cols_to_use]
y = trainData.Survived
predictX = testData[cols_to_use]

# Create Usable Data with OneHotEncoding
one_hot_encoded_predict_X = pd.get_dummies(predictX)
one_hot_encoded_X = pd.get_dummies(X)
final_X, final_predict_X = one_hot_encoded_X.align(one_hot_encoded_predict_X, join='left', axis=1)

# Create test & validaton variables
train_X, test_X, train_y, test_y = train_test_split(final_X, y)

# Create Imputer & RandomForestRegressor
my_imputer = Imputer()
my_model = RandomForestClassifier(random_state=0)

# Impute train & test X
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)

# Fit Model & make predictions
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)

# Calculate MAE
print("MAE: ", str(mean_absolute_error(predictions, test_y)))

# Predict Competition Data
final_predict_X = my_imputer.transform(final_predict_X)
competitonPredictions = my_model.predict(final_predict_X)

# Create a CSV with predicitons
output = pd.DataFrame({'PassengerId': testData.PassengerId,
                       'Survived': competitonPredictions})

output.to_csv('submission.csv', index=False)
