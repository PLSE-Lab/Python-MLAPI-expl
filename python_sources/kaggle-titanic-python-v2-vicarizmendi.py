# In this challenge, it is asked you to complete the analysis of what sorts of people
#  were likely to survive. In particular, to apply the tools of 
# machine learning to predict which passengers survived the tragedy.

import numpy as np
import pandas as pd
import csv

#Print you can execute arbitrary python code
df_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},na_values=[""] )
df_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},na_values=[""] )

#Any files you save will be available in the output tab below
df_train.to_csv('copy_of_the_training_data.csv', index=False)

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(df_train.head())

print("\n\nNumber of observations and number of features+target of training data")
print(df_train.shape) # (891, 11)

print("\n\nData types of features and target 'Survived' of training data")
print(df_train.dtypes)

print("\n\nSummary statistics of training data")
print(df_train.describe())
 
# Fill in the rows with the most convenient values

# See all NaNs in df_train
print(df_train.isnull().sum())

# All the ages with no data -> make the median of all Ages
median_age = df_train['Age'].dropna().median()
if len(df_train.Age[ df_train.Age.isnull() ]) > 0:
    df_train.loc[ (df_train.Age.isnull()), 'Age'] = median_age

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not
# 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
print(df_train.isnull().sum())
Embarkement_mode = df_train.Embarked.dropna().mode().values
if len(df_train.Embarked[ df_train.Embarked.isnull() ]) > 0:
    df_train.loc[df_train.Embarked.isnull(),"Embarked"] = Embarkement_mode
print(df_train.isnull().sum())

# Convert strings to numeric in Embarked
Ports = list(enumerate(np.unique(df_train['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
df_train.Embarked = df_train.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


# Assign the correct type of each feature  - For Random Forest all features
# have to be numerical:
print(df_train.dtypes)

# Column "Sex" we have to change it from object to categorical - nominal
df_train["Sex"]=df_train["Sex"].map({"male":0,"female":1}).astype("int")

# Save the column passenger identification before applying the random forest 
PassengerID=df_train["PassengerId"]

# Column "Name", "Cabin" and "Ticket" could be eliminated as it is enough to 
# maintaim the index PassengerID
df_train = df_train.drop(labels=["Name","Cabin","Ticket","PassengerId"], axis=1) 

# Check all features have the right type
print(df_train.shape)
print(df_train.dtypes)
print(df_train.describe())

# Separate the features and the target
y=df_train.loc[:,"Survived"].astype("int64")
X=df_train.drop(labels=["Survived",], axis=1)

# From df_train X and y we subset internally in train and test to have them 
# to check internally the score of the model. The train is reduced in 20%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=7)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, oob_score=True)
model.fit(X_train, y_train)
print (model.oob_score_)

# Score the model with the 20% left outside the df_train dataset
score=model.score(X_test,y_test)
print ("Score: ", round(score*100, 3)) # Score:  82.123

# Predict the target with the test.csv dataset

print(df_test.shape) # (891, 11)
print(df_test.dtypes)
print(df_test.describe())

# See all NaNs in df_test
print(df_test.isnull().sum())

# All the ages with no data -> make the median of all Ages
median_age = df_test['Age'].dropna().median()
if len(df_test.Age[ df_test.Age.isnull() ]) > 0:
    df_test.loc[ (df_test.Age.isnull()), 'Age'] = median_age

# All missing Embarked -> just make them embark from most common place in train "S"
print(df_test.isnull().sum())
#Embarkement_mode = df_test.Embarked.dropna().mode().values
if len(df_test.Embarked[ df_test.Embarked.isnull() ]) > 0:
    df_test.loc[df_test.Embarked.isnull(),"Embarked"] = Embarkement_mode
print(df_test.isnull().sum())

# Convert strings to numeric in Embarked using the same dict as in df_train
df_test.Embarked = df_test.Embarked.map( lambda x: Ports_dict[x]).astype(int) 

# Assign the correct type of each feature  - For Random Forest all features
# have to be numerical:
print(df_test.dtypes)

# Column "Sex" to int
df_test["Sex"]=df_test["Sex"].map({"male":0,"female":1}).astype("int")

# Save the column passenger identification before applying the random forest 
PassengerID=df_test["PassengerId"]

# Column "Name", "Cabin" and "Ticket" could be eliminated
df_test = df_test.drop(labels=["Name","Cabin","Ticket","PassengerId"], axis=1) 

# Check all features have the right type, there are no NaNs...
print(df_test.shape)
print(df_test.dtypes)
print(df_test.describe())
print(df_test.isnull().sum())

# Substitute the NaN in Fare by the mean Fare
Fare_mean = df_test.Fare.dropna().mean()
if len(df_test.Fare[ df_test.Fare.isnull() ]) > 0:
    df_test.loc[df_test.Fare.isnull(),"Fare"] = Fare_mean
print(df_test.isnull().sum())

# Predict the Survived using the Random Forest model trained with df_train
X=df_test
y_test=model.predict(X)
Survived=y_test

#Save the passengers IDs and the Survived Prediction in a csv file
predictions_file = "myfirstforest_v2.csv"

with open(predictions_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    header = ['PassengerId', 'Survived']
    writer.writerow(header)
    writer.writerows(zip(PassengerID, Survived))
csvfile.close()