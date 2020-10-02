import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
#Print you can execute arbitrary python code
titanic_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
titanic_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

def cleanData(data):
    #print(titanic.head(5))
    #print (titanic.describe())
    
    data["Age"] = data["Age"].fillna(-1)
    
    #print (titanic["Sex"].unique())
    
    data.loc[data["Sex"] == "male","Sex"] = 0
    data.loc[data["Sex"] == "female","Sex"] = 1
    
    #print (titanic["Sex"].unique())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna("S")
    
    #print titanic["Embarked"].unique()
    #print "Unique values of class of travel:",titanic.Pclass.unique()
    data.loc[data["Embarked"] == "S","Embarked"] = 0
    data.loc[data["Embarked"] == "C","Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
    return data

if __name__ == "__main__":
    
    data_train = cleanData(titanic_train)
    predictors = ['Sex','Pclass','Age','Embarked','Fare']
    feature = data_train[predictors]
    label = data_train['Survived']
    #print "Printing Survived"
    
    """
    Spliting the training data to test the prediction
    """
    feature_train, feature_test,label_train,label_test = train_test_split(feature,label, test_size = 0.3)
    
    clf = tree.DecisionTreeClassifier()
    #feature_new = SelectKBest(chi2,k =2).fit_transform(feature_train,label_train)
    #print (feature_new)
    clf.fit(feature_train,label_train)
    
    pred = clf.predict(feature_test)
    score = accuracy_score(pred,label_test)
    print ("Accuracy value is:",score)
    #
    clfRandomForest = RandomForestClassifier(n_estimators = 100)
    clfRandomForest.fit(feature_train,label_train)
    pred = clfRandomForest.predict(feature_test)
    score = accuracy_score(pred,label_test)
    print ("Accuracy of the Random forest:",score)
    
    
    """
    Fit Decision tree to entire train data and predict the test data
     
    
    """
    data_test = cleanData(titanic_test)
    
    #clf.fit(feature,label)
    #Prediction using Random forest"
    #clfRandomForest.fit(feature,label)
    #pred = clfRandomForest.predict(data_test[predictors])
    pred = clf.predict(data_test[predictors])
    submission =  pd.DataFrame({
                     "PassengerID": titanic_test["PassengerId"],
                      "Survived": pred})
                      
    submission.to_csv('submission.csv',index = False)