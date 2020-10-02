import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

############# HELPER METHODS ##################
def add_features(train_data):
    family_size_column = []
    title_column = []
    for i, row in train_data.iterrows():
        family_size = row["SibSp"] + row["Parch"]
        if family_size == 0:
            family_size_column.append("Single")
        elif family_size < 5:
            family_size_column.append("Small")
        else:
            family_size_column.append("Large")
    train_data["FamilySize"] = family_size_column
    train_data["Title"] = train_data["Name"].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    train_data["CabinSection"] = train_data["Cabin"].apply(lambda x: str(x)[0])
    return train_data

def preprocess(train_data, test_data):
    # fill in missing data
    #it's 3 because the only problematic missing value is "Embarcked"
    #and the most common value in the training set is S (3)
    train_data["Embarked"].fillna("S", inplace=True) # FIX THIS JANK
    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    # one hot encode the training features
    category_cols = [
        "Pclass",
        "Sex",
        "Embarked",
        "CabinSection",
        "Title",
        "FamilySize"]

    for col in category_cols:
        train_data[col] = train_data[col].apply(lambda x: str(x))
        test_data[col] = test_data[col].apply(lambda x: str(x))
        good_cols = [col+'_'+i for i in train_data[col].unique() if i in test_data[col].unique()]
        train_data = pd.concat(
            (train_data, pd.get_dummies(train_data[col], prefix = col)[good_cols]),
            axis = 1)
        test_data = pd.concat(
            (test_data, pd.get_dummies(test_data[col], prefix = col)[good_cols]),
            axis = 1)
        train_data.drop(col, 1, inplace=True)
        test_data.drop(col, 1, inplace=True)
    
    # remove unimportant features from training data
    train_data.drop("Name", 1, inplace=True)
    train_data.drop("Ticket", 1, inplace=True)
    train_data.drop("Cabin", 1, inplace=True)
    train_data.drop("PassengerId", 1, inplace=True)
    test_data.drop("Name", 1, inplace=True)
    test_data.drop("Ticket", 1, inplace=True)
    test_data.drop("Cabin", 1, inplace=True)

    train_x = train_data.drop("Survived", 1)
    train_y = train_data["Survived"]
    test_x = test_data.drop("PassengerId", 1)
    ids = test_data["PassengerId"]

    return train_x, train_y, test_x, ids

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

######### PREPROCESSING ########################
add_features(train_data)
add_features(test_data)
##print(train_data.head())
##print(train_data.describe())
train_x, train_y, test_x, ids = preprocess(train_data, test_data)
##print(train_x.head())
print(list(train_x))
print(list(test_x))

######### TRAINING ###########################
model = LogisticRegression()
model = model.fit(train_x, train_y)

######### VALIDATING #########################
print("SCORE:",model.score(train_x, train_y))

################ TEST ##################
test_y = model.predict(test_x)
result = np.dstack((ids, test_y))[0]
f = open("output.csv", 'w')
f.write("PassengerId,Survived\n")
for row in result:
    f.write(str(row[0]) + "," + str(row[1]) + "\n")