import numpy as np
import pandas as pd
from pandas import Series,DataFrame
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#you can execute arbitrary python code
original_train = pd.read_csv("../input/train.csv")
original_test = pd.read_csv("../input/test.csv")
train = original_train.copy()
test = original_test.copy()

target = train["Survived"].values

# Embarked
# only in train, fill the two missing values with the most occurred value, which is "S".
train["Embarked"] = train["Embarked"].fillna("S")
# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.
embark_dummies_train  = pd.get_dummies(train['Embarked'])
#embark_dummies_train.drop(['S'], axis=1, inplace=True)
embark_dummies_test  = pd.get_dummies(test['Embarked'])
#embark_dummies_test.drop(['S'], axis=1, inplace=True)
train = train.join(embark_dummies_train)
test    = test.join(embark_dummies_test)
train.drop(['Embarked'], axis=1,inplace=True)
test.drop(['Embarked'], axis=1,inplace=True)

# Fare
# only for test, since there is a missing "Fare" values
test["Fare"].fillna(test["Fare"].median(), inplace=True)
# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare']    = test['Fare'].astype(int)

# Age
# get average, std, and number of NaN values in train
average_age_train   = train["Age"].mean()
std_age_train     = train["Age"].std()
count_nan_age_train = train["Age"].isnull().sum()
# get average, std, and number of NaN values in test
average_age_test   = test["Age"].mean()
std_age_test       = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()
# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
# fill NaN values in Age column with random values generated
train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2
# convert from float to int
train['Age'] = train['Age'].astype(int)
test['Age']    = test['Age'].astype(int)

#Family
# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

# drop Parch & SibSp
train = train.drop(['SibSp','Parch'], axis=1)
test    = test.drop(['SibSp','Parch'], axis=1)

# Sex
# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)
test['Person']    = test[['Age','Sex']].apply(get_person,axis=1)
# No need to use Sex column since we created Person column
train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)
# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_train  = pd.get_dummies(train['Person'])
person_dummies_train.columns = ['Child','Female','Male']
#person_dummies_train.drop(['Male'], axis=1, inplace=True)
person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child','Female','Male']
#person_dummies_test.drop(['Male'], axis=1, inplace=True)
train = train.join(person_dummies_train)
test    = test.join(person_dummies_test)
train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)

# Pclass
# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_train  = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']
#pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)
pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
#pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)
train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)
train = train.join(pclass_dummies_train)
test    = test.join(pclass_dummies_test)



#drop unnecessary columns
train = train.drop(['PassengerId','Name','Ticket','Cabin','Survived', 'Age'], axis=1)
test    = test.drop(['PassengerId','Name','Ticket','Cabin','Age'], axis=1)


# Random Forests
random_forest = RandomForestClassifier(min_samples_split=2, n_estimators=500, random_state=1)
random_forest.fit(train, target)
survived_prediction = random_forest.predict(test)
print(random_forest.score(train, target))


#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())


submission = pd.DataFrame({
        "PassengerId": original_test["PassengerId"],
        "Survived": survived_prediction
    })
submission.to_csv('titanic.csv', index=False)












