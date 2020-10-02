import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

survive = {}

def clean_data( array):
	array = pd.concat( ( array, pd.get_dummies( array[ "Sex" ])), axis=1)
	array = array.drop( 'Sex', axis=1)
	array = array.drop( 'female', axis=1)
	array = pd.concat( ( array, pd.get_dummies( array[ "Embarked" ])), axis=1)
	array = array.drop( 'Embarked', axis=1)
	array = array.drop( 'S', axis=1)
	array[ "fsize" ] = array[ "SibSp" ] + array[ "Parch" ] + 1
	return array;

def norm_array( array, col):
	tmp = array[col]
	std = tmp.std()
	mean = tmp.mean()
	tmp = (tmp - mean) / std
	array = array.drop( col, axis=1)
	array = pd.concat( (array, tmp), axis=1)
	array = norm_array( array, "Age")
	array = norm_array( array, "fsize")
	return array;

def is_title(item):
    if item in ["Mr.", "Miss.", "Mrs.", "Rev.", "Master.", "Dr.", "Major."]:
        return True
    return False

def get_family_name(name):
    flag = False
    for n in name.split(" "):
        if flag == True:
            return n
        flag = is_title(n)
    return ""

def append_fsurv(array):
    tmp=[]
    for i in range(0, array["Name"].size):
        name = array["Name"][i]
        fname = get_family_name(name)
        tmp.append(survive.get(fname, 0))
        print(i, name, fname, survive.get(fname, 0))
    fsurv = pd.DataFrame(data={"fsurv": tmp})
    return pd.concat( ( array, fsurv), axis=1)

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#
for i in range(0, 889):
    name = train["Name"][i]
    fname = get_family_name(name)
    if fname == "":
        continue
#    print(fname)
    survive[fname] = survive.get(fname, 0) + train["Survived"][i];
#print(survive)

    

train = append_fsurv(train)
test = append_fsurv(test)

print(train.isnull().sum())
print(test.isnull().sum())


# Cleaning and Formatting Training Data
train = train.dropna()
train = clean_data( train )
print("\n\nClean training data")
print( train )



# Create the target and features numpy arrays: target, features_one
#print("\n\nSurvived data")
#print( train.head )
target = train["Survived"].values
features_forest = train[["Pclass", "male", "Age", "Fare", "SibSp", "Parch", "C", "fsize", "fsurv"]].values

# Fit your first decision tree: my_tree_one
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Look at the importance and score of the included features
print(my_forest.feature_importances_)
print(my_forest.score(features_forest, target))

# Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Fare"] = test["Fare"].fillna(train["Fare"].median())

test = clean_data( test)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "male", "Age", "Fare", "SibSp", "Parch", "C", "fsize", "fsurv"]].values

# Make your prediction using the test set
my_prediction = my_forest.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

