# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

pd.options.mode.chained_assignment = None  # default='warn'

# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#Extract titles
titanic_df["Title"] = titanic_df["Name"].str.replace('(.*, )|(\\..*)','')
#titanic_df.info()
rare_Title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
titanic_df["Title"][titanic_df["Title"] == "Mlle"] = "Miss"
titanic_df["Title"][titanic_df["Title"] == "Ms"] = "Miss"
titanic_df["Title"][titanic_df["Title"] == "Mme"] = "Miss"
titanic_df["Title"][titanic_df["Title"].isin(rare_Title)] = "Rare"

titanic_df_titles_Dummies = pd.get_dummies(titanic_df["Title"])
titanic_df = titanic_df.join(titanic_df_titles_Dummies)
titanic_df = titanic_df.drop("Title",axis=1)

test_df["Title"] = test_df["Name"].str.replace('(.*, )|(\\..*)','')
#titanic_df.info()

test_df["Title"][test_df["Title"] == "Mlle"] = "Miss"
test_df["Title"][test_df["Title"] == "Ms"]= "Miss"
test_df["Title"][test_df["Title"] == "Mme"]= "Miss"
test_df["Title"][test_df["Title"].isin(rare_Title)] = "Rare"
test_df_titles_Dummies = pd.get_dummies(test_df["Title"])
test_df = test_df.join(test_df_titles_Dummies)
test_df = test_df.drop("Title",axis=1)
#sns.factorplot("Title","Survived",data = titanic_df)
#sns.plt.show()
#remove un important features
titanic_df = titanic_df.drop(["Name","PassengerId","Ticket","Cabin"],axis = 1)
test_df = test_df.drop(["Name","Ticket","Cabin"],axis = 1)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']  = test_df['Age'].astype(int)

# plot new Age Values
#titanic_df['Age'].hist(bins=70, ax=axis2)
# Check effect for Pclass

#sns.factorplot("Pclass","Survived",data=titanic_df)
#plt.show()

# Embarked fill missing values

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S");

titanic_df_Embarked_Dummies = pd.get_dummies(titanic_df["Embarked"]);
titanic_df = titanic_df.join(titanic_df_Embarked_Dummies)
titanic_df = titanic_df.drop("Embarked",axis=1)
test_df_Embarked_Dummies = pd.get_dummies(test_df["Embarked"]);
test_df = test_df.join(test_df_Embarked_Dummies)
test_df = test_df.drop("Embarked",axis=1)

#Fare
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

titanic_df["Fare"] = titanic_df["Fare"].astype(int)
test_df["Fare"] = test_df["Fare"].astype(int)

#Family

def map_family_range(size):
    if size == 0: return 'Single'
    elif 0 < size <= 3 : return 'Small'
    elif 3 < size <= 5: return 'Medium'
    elif 5 < size : return 'Large'


titanic_df["Family"] = titanic_df["SibSp"] + titanic_df["Parch"]
titanic_df["Family"] = titanic_df["Family"].map(map_family_range)

titanic_df_Family_dummies = pd.get_dummies(titanic_df["Family"])
titanic_df = titanic_df.join(titanic_df_Family_dummies)
titanic_df = titanic_df.drop("Family",axis=1)

test_df["Family"] = test_df["SibSp"] + test_df["Parch"]
test_df["Family"] = test_df["Family"].map(map_family_range)

test_df_Family_dummies = pd.get_dummies(test_df["Family"])
test_df = test_df.join(test_df_Family_dummies)
test_df = test_df.drop("Family",axis=1)



# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

#Sex encode

titanic_df_Sex_Dummies = pd.get_dummies(titanic_df["Sex"])
titanic_df = titanic_df.join(titanic_df_Sex_Dummies)
titanic_df = titanic_df.drop("Sex",axis =1)

test_df_Sex_Dummies = pd.get_dummies(test_df["Sex"])
test_df = test_df.join(test_df_Sex_Dummies)
test_df = test_df.drop("Sex",axis =1)
#Train
# define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))
importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
print(X_train.head())
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('gender_submission.csv', index=False)