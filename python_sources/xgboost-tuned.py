import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
combine = [train, test]

#Method for finding substrings
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan

#Mappings
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

#Passenger Class is 3, so fill nan row with the mode
fare_mode = test[test['Pclass']==3]['Fare'].mode()
test['Fare'] = test['Fare'].fillna(fare_mode[0])

#Find the mode of embarked of passengers with the same class and similar fare
emb_mode = train[(train['Pclass']==1)&(train['Fare']<=85)&(train['Fare']>75)]['Embarked'].mode()
train['Embarked'] = train['Embarked'].fillna(emb_mode[0])

for df in combine:
    # Convert the male and female groups to integer form
    df["Sex"][df["Sex"] == "male"] = 0
    df["Sex"][df["Sex"] == "female"] = 1
    
    #Map and Create Title Feature
    df['Title'] = df['Name'].astype(str).map(lambda x: substrings_in_string(x, title_list))
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    #Map and Create Deck feature
    df['Deck'] = df['Cabin'].astype(str).map(lambda x: substrings_in_string(x, cabin_list))
    df["Deck"][df["Deck"] == "A"] = 1
    df["Deck"][df["Deck"] == "B"] = 2
    df["Deck"][df["Deck"] == "C"] = 3
    df["Deck"][df["Deck"] == "D"] = 4
    df["Deck"][df["Deck"] == "E"] = 5
    df["Deck"][df["Deck"] == "F"] = 6
    df["Deck"][df["Deck"] == "G"] = 7
    df["Deck"][df["Deck"] == "T"] = 8
    df["Deck"] = df["Deck"].fillna(0)
    
    #Create Family size, Fare per person, and isAlone features
    df['Family_size'] = df['SibSp']+df['Parch']+1
    
    #Create isAlone feature based off family size
    df['isAlone']=0
    df.loc[df['Family_size']==1, 'isAlone'] = 1
    
    # Convert the Embarked classes to integer form
    df["Embarked"][df["Embarked"] == "S"] = 0
    df["Embarked"][df["Embarked"] == "C"] = 1
    df["Embarked"][df["Embarked"] == "Q"] = 2

    #Impute Age based off random numbers in one standard deviation from the mean
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['Age'][np.isnan(df['Age'])] = age_null_random_list
    
    # Mapping Age
    df.loc[ df['Age'] <= 16, 'Age'] 					= 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']                      = 4
    
    df['Sex'] = df['Sex'].astype(int)
    df['Embarked'] = df['Embarked'].astype(int)

 
#Create target feature set
excl = ['PassengerId', 'Survived', 'Ticket', 'Cabin', 'Name', 'Parch', 'SibSp', 'Fare_Per_Person']
cols = [c for c in train.columns if c not in excl]


target = train["Survived"].values
features = train[cols].values

xgb = XGBClassifier(
 learning_rate=0.1,
 n_estimators= 2000,
 max_depth= 3,
 min_child_weight= 2,
 gamma=0.2,                        
 subsample=0.85,
 colsample_bytree=0.5,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1,
 reg_alpha=1.05,
 seed=10)
 
xgbmod = xgb.fit(features, target)

#Predict test
test_features = test[cols].values
pred = xgbmod.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])

my_solution.to_csv("xgb.csv", index_label = ["PassengerId"])