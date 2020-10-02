
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Get absolute file paths for all files in current directory
import os
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# Select correct csv files, import the data
# Training csv has 'train' in the filename
train_filename = list([x for x in files if "train" in x])[0]
# Testing csv has 'test' in the filename
test_filename = list([x for x in files if "test" in x])[0]
train = pd.read_csv(train_filename)
to_predict = pd.read_csv(test_filename)

# Inspect columns, data formats, see what might be useful
print(train.head())

# We wouldn't expect name, ticket number to be useful for prediction
# The precise cabin number would not be useful for prediction
# but maybe whether or not the passenger had a cabin?

# A function that fills NaNs in column with a randomly chosen number
def column_fill_nan_random(dataframe, column_name):
    filler = np.random.randint(-10000, 10000)
    # Choose a new number so long as the chosen number already exists in the column
    while filler in dataframe[column_name].unique():
        filler = np.random.randint(-10000, 10000)
    dataframe[column_name] = dataframe[column_name].fillna(filler)
    return dataframe

# Conversion of text features to numerical features
def column_text_to_numerical_feature(dataframe, column_name):
    # Clear NaNs - here we don't care about the
    # precise value of the NaNs that we're replacing
    # we just want them to have the same numerical value
    # so that that next step will work
    dataframe = column_fill_nan_random(dataframe, column_name)
    # Find all unique values in the column
    unique_values = dataframe[column_name].unique()
    # For each value in the column, replace it by
    # the index of the value in the list of unique values
    # Thus all quantities in the column are replaced
    # by integers, such that two rows with the
    # same value will now have the same integer value
    dataframe[column_name] = dataframe[column_name].map(lambda s: list(unique_values).index(s))
    return dataframe
 
 
# Convert sex and embarkation point to integer labels
# Required for some machine learning algorithms
def text_to_numerical_features(dataframe):
    dataframe = column_text_to_numerical_feature(dataframe, "Sex")
    dataframe = column_text_to_numerical_feature(dataframe, "Embarked")
    return dataframe
    

# Do feature -> integer conversion on both training data, and data to predict
train = text_to_numerical_features(train)
to_predict = text_to_numerical_features(to_predict)

# This function replaces NaNs in a particular column in
# a dataframe with the mean value of the !NaN rows
def column_replace_nans_with_mean(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].fillna(
        dataframe[column_name].mean()
    )
    return dataframe


# These columns in our dataframe will
# have their NaN values replaced by means
mean_replace_nan_columns = [
    "Age",
    "Fare"
]
for column_name in mean_replace_nan_columns:
    train = column_replace_nans_with_mean(train, column_name)
    to_predict = column_replace_nans_with_mean(to_predict, column_name)

# Select features to use in model
feature_names = [
    "Age",
    "Pclass",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]
train_on = train[feature_names]
predict_on = to_predict[feature_names]
# Also get labels
train_labels = train["Survived"]

# First try a logistic regression approach
# Requires scaled features
scaler = StandardScaler()
scaler.fit(train_on)
train_on_scaled = scaler.transform(train_on)
predict_on_scaled = scaler.transform(predict_on)

model = LogisticRegression()
model.fit(train_on_scaled, train_labels)
score = model.score(train_on_scaled, train_labels)
print("Logistic regression approach, training accuracy {:.3f}".format(score))

# Try with a single decision tree
# Range of maximum tree depths to test
maximum_depths = range(1, 31)
# How many repetitions to average over for validation
VALIDATION_REPETITIONS = 100
# Track training/validation accuracy across all depths
training_scores = []
validation_scores = []
for maximum_depth in maximum_depths:
    # Split into training and validation sets, randomly
    # Average training/validation over 100 runs
    training_scores.append(0)
    validation_scores.append(0)
    for i in range(VALIDATION_REPETITIONS):
        # Add a new entry to the scores arrays, where we'll store the scores
        (
            training_set,
            validation_set,
            training_labels,
            validation_labels
        ) = train_test_split(
            train_on,
            train_labels,
            random_state=np.random.randint(1,1000000)
        )
        # Build, train decision tree
        classifier = DecisionTreeClassifier(
            max_depth=maximum_depth
        )
        classifier.fit(
            training_set,
            training_labels
        )
        training_scores[-1] += (1 / float(VALIDATION_REPETITIONS)) * classifier.score(
            training_set,
            training_labels
        )
        validation_scores[-1] += (1 / float(VALIDATION_REPETITIONS)) * classifier.score(
            validation_set, 
            validation_labels
        )
plt.plot(maximum_depths, training_scores)
plt.plot(maximum_depths, validation_scores)
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
plt.show()
# Best validation accuracy is around a maximum depth of 4
# Model is overfitting past that point

# Make predictions for test set
BEST_DEPTH = 4
classifier = DecisionTreeClassifier(
    max_depth=BEST_DEPTH
)
classifier.fit(train_on, train_labels)
column_predictions = classifier.predict(predict_on)
# Store them in a dataframe, then save to csv
predictions = pd.DataFrame(
    to_predict["PassengerId"],
    columns=[
        "PassengerId"
    ]
)
predictions["Survived"] = column_predictions

# The decision tree approach gives c. 0.7 validation accuracy at best
# Let's output logistic regression predictions to see if they are any better
# column_predictions = model.predict(predict_on_scaled)
# predictions = pd.DataFrame(
#     to_predict["PassengerId"],
#     columns=[
#         "PassengerId"
#     ]
# )
# predictions["Survived"] = column_predictions

# Save predictions to csv file 
predictions.to_csv(
    "gender_submission.csv",
    index=False
)
