#! /usr/bin/python
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
import re



def getTitle(fullname):
    m = p.match(fullname)
    if m == None:
      print("Warning: Fullname doesn't match: " + fullname)
      return 4

    title = m.group(2)
    if title == 'Master.':
        return 0
    elif title == 'Miss.':
        return 1
    elif title == 'Mlle.':
        return 1
    elif title == 'Ms.':
        return 1
    elif title == 'Mr.':
        return 2
    elif title == 'Mrs.':
        return 3
    elif title == 'Mme.':
        return 3
    else: 
        return 4

def addFeatures(df):
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Title'] = df['Name'].apply(lambda x: getTitle(x))
    df['EmbarkmentNum'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)   
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    return df

def removeUnusedFeatures(df):    
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Age'], axis=1) 
    df = df.drop(['PassengerId'], axis=1) 
    df = df.drop(['Embarked'], axis=1) 
    return df
    
def addAge(df):
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
    return df

if __name__ == '__main__':
    #Print you can execute arbitrary python code
    train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
    test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

    train_orig = train
    test_orig = test
    
    train['Embarked'].fillna('C', inplace=True)
    
    p = re.compile('(.*?, )(.*\.)(.*)')
    
    train = addFeatures(train)
    test  = addFeatures(test)


    print(train.head(20))
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = train[(train['Gender'] == i) & \
                                  (train['Pclass'] == j+1)]['Age'].dropna().median()
    
    train = addAge(train)
    test  = addAge(test)
    
    train = removeUnusedFeatures(train)
    test  = removeUnusedFeatures(test)
    
    X_train = train.drop("Survived",axis=1)
    Y_train = train["Survived"]
    
    
    
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(test)
    #print("Logistic Regresion")
    #print(logreg.score(X_train, Y_train))
    
    print("LogReg Feature ranking:")
    print(logreg.coef_)
    print(X_train.describe())
    #for f in range(logreg.coef_):
    #    print("%d. feature %s (%f)" % (f + 1, X_train.columns[f], logreg.coef_[f]))
    
    
#    tuning_parameters = [
#        {'n_estimators': [10, 100, 1000],
#         'criterion': ['gini', 'entropy'],
#         'min_samples_split': [2,3,5,7,11],
#         'bootstrap': ['True','False'],
#         'oob_score': ['True','False']
#        }
#    ]
    # Grid Search Cross Validation
    
#    X_trainCV, X_testCV, y_trainCV, y_testCV = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
    
#    model = RandomForestClassifier()
#    grid_best, grid_params, grid_scores = grid_searcher(X_trainCV, X_testCV, y_trainCV, y_testCV, model, tuning_parameters)
    
    
    forest = RandomForestClassifier(n_estimators = 10, bootstrap = 'False', min_samples_split = 5, criterion = 'entropy')
    forest = forest.fit(X_train, Y_train)
    output = forest.predict(test)
    
    
    
    
    
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    
    for f in range(X_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X_train.columns[f], importances[indices[f]]))
    
    
    
    # preview
    #print(coeff_df)
    
    submission = pd.DataFrame({
            "PassengerId": test_orig["PassengerId"],
            "Survived": output
    })
        
        
    submission.to_csv('titanic.csv', index=False)
