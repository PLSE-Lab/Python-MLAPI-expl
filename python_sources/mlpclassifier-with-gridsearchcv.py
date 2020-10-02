import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

predictor = [
    "Age", 
    "Fare", 
    "SibSp",
    "Parch",
    "FamilySize",
    "male", 
    "female", 
    "pclass_1", 
    "pclass_2", 
    "pclass_3",
    ]
    
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

def correct_data(titanic_data):
    #titanic_data.Sex = titanic_data.Sex.replace(['male', 'female'], [0, 1])
    titanic_data.Embarked = titanic_data.Embarked.fillna("S")
    #titanic_data.Embarked = titanic_data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
    titanic_data.Age = titanic_data.Age.fillna(titanic_data.Age.median())
    
    # Trying to add FamilySize
    titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
    
    # add Male & Female one-hot-encoding
    sex_dummy = pd.get_dummies(titanic_data.Sex)
    titanic_data['male']=sex_dummy.male
    titanic_data['female']=sex_dummy.female
    
    # add Embarked one-hot-encoding
    embarked_dummy = pd.get_dummies(titanic_data.Embarked)
    titanic_data['embarked_C'] = embarked_dummy.C
    titanic_data['embarked_S'] = embarked_dummy.S
    titanic_data['embarked_Q'] = embarked_dummy.Q
    
    # add Pclass one-hot-encoding
    pclass_dummy = pd.get_dummies(titanic_data.Pclass)
    titanic_data['pclass_1'] = pclass_dummy[1]
    titanic_data['pclass_2'] = pclass_dummy[2]
    titanic_data['pclass_3'] = pclass_dummy[3]
    
    # Cabin
    titanic_data.Cabin = titanic_data.Cabin.fillna("N")
    titanic_data['CabinDeck'] = titanic_data.Cabin.str.slice(0,1)
    cabindeck_dummy = pd.get_dummies(titanic_data.CabinDeck)

    titanic_data['CabinDeck_A'] = cabindeck_dummy.A
    titanic_data['CabinDeck_B'] = cabindeck_dummy.B
    titanic_data['CabinDeck_C'] = cabindeck_dummy.C
    titanic_data['CabinDeck_D'] = cabindeck_dummy.D
    titanic_data['CabinDeck_E'] = cabindeck_dummy.E
    titanic_data['CabinDeck_F'] = cabindeck_dummy.F
    titanic_data['CabinDeck_G'] = cabindeck_dummy.G
    titanic_data['CabinDeck_NaN'] = cabindeck_dummy.N

    titanic_data.Fare = titanic_data.Fare.fillna(titanic_data.Fare.median())
    
    return titanic_data

correct_train = correct_data(train).sample(frac=1)

trainX = correct_train[predictor].values
trainY = correct_train.Survived.values

parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

clf.fit(trainX, trainY)
print(clf.score(trainX, trainY))
print(clf.best_params_)

correct_test = correct_data(test)
testX = correct_test[predictor].values
result = clf.predict(testX)

test["Survived"] = result
result = test[["PassengerId", "Survived"]]

result.to_csv('titanic_RandomForestClassifier_FamilySize.csv', index=False)