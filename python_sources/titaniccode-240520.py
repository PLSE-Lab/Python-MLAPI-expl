# %% [code]
#KAGGLE TITANIC COMPITETION#
    
import pandas as pd
import numpy as np
import seaborn as sns

titanic= pd.read_csv("E://Samuel Data//DATA//Kaggle//Titanic//train.csv")
titanicOrg = pd.read_csv("E://Samuel Data//DATA//Kaggle//Titanic//train.csv") #Duplicate for reference 
titanic.head()

#EDA
titanic.info()  #5 - int64, 5 - object64 , 2 - float64
#categorical variables : [Name, sex, ticket,cabin, embarked]
titanic.isna().sum() #Age and cabin has high nan values (177,687)
Summary = titanic.describe()

#Analyzing Age Data
titanic["Age"].plot(kind ="kde", grid = True) 
titanic["Fare"].plot(kind ="hist", grid = True, bins = 20, xticks =(range(0,500,50)), yticks =(range(0,600,50))) 

#CountPlots
sns.countplot(titanic.Survived, hue = titanic.Pclass) #Survived VS Pclass
sns.countplot(titanic.Survived, hue = titanic.Sex) #Survived Vs Sex
sns.countplot(titanic.Pclass, hue = titanic.Sex) #Pclass VS Sex 
sns.countplot(titanic.Embarked,hue = titanic.Sex) #Embarked Vs Sex
sns.countplot(titanic.Embarked,hue = titanic.Survived) #Embarked Vs Sex
sns.countplot(titanic.Embarked,hue = titanic.Pclass) #Embarked Vs Pclass
 
#Boxplots
sns.boxplot(titanic.Sex,titanic.Age) #Sex Vs Age 
sns.boxplot(titanic.Survived,titanic.Age) #Survived Vs Age
sns.boxplot(titanic.Pclass,titanic.Age) #Pclass Vs Age
sns.boxplot(titanic.Embarked,titanic.Age) #EMbarked vs Age
       
#Imputing NaN's
titanic.isnull().sum()
sns.heatmap(titanic.isnull()) 
titanic.Embarked.value_counts()
titanic["Embarked"].fillna("S", inplace = True)
titanic["Age"].fillna(0, inplace =True) 

#Analyzing Age column and Imputing with different values for (Mr,Mrs,Miss,Master)
nullage = titanic[titanic.Age == 0] #subset of titanic w.r.t Null Age Values
misage =set() #36 null values for Miss (girl child) 
mastage = set() #4 null values for Master (boy child)
Mrage = set() #119 null values for Mr.(Men)
Mrsage = set() #17 null values for Mrs (women)
miscage = set() #1 null values for unknown(Dr.)

#saperating Names w.r.t(Mr,Mrs,Miss,Master)
for i in nullage.Name:
    if "Miss" in i:
        misage.add(i) 
    elif "Master" in i:
        mastage.add(i) 
    elif "Mr." in i:
        Mrage.add(i)
    elif "Mrs" in i:
        Mrsage.add(i)
    else:
        miscage.add(i)        
    
#Analyzing mean Age w.r.t Age groups
df = pd.DataFrame(columns = ["Age"], index =("Men","Women","boy","girl"))
df.Age = [26,24,5,10]

#Defining a fucntion to get the index values of particular names"""
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

getIndexes(nullage, "Moran, Mr. James")

"""Filling the Null values of Age w.r.t the values that are analyzed and confirmed
    Miss = 10, Master = 5, Mr = 26,Mrs = 24, doc = 40"""
    
for i in nullage.Name:
    if "Miss" in i:
        titanic.Age.iloc[(getIndexes(nullage, i))] = 10
    elif "Master" in i:
        titanic.Age.iloc[(getIndexes(nullage, i))] = 5
    elif "Mr." in i:
        titanic.Age.iloc[(getIndexes(nullage, i))] = 26
    elif "Mrs" in i:
        titanic.Age.iloc[(getIndexes(nullage, i))] = 24
    else:
        titanic.Age.iloc[(getIndexes(nullage, i))] = 40

sns.heatmap(titanic.isnull()) #only cabin has highest Null values, thus it is deleted

#Data cleaning
titanic.drop("Cabin", axis = 1, inplace = True) 
titanic.drop("Ticket", axis = 1, inplace = True)
titanic.drop("Name", axis = 1, inplace = True)
titanic.drop("PassengerId", axis = 1, inplace = True) 
#Cabin,Ticket,Name,PassengerId are removed

sns.heatmap(titanic.corr(), annot = True) #checking correlations

#labelEncoding Categorical variables
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
titanic['Sex']= le.fit_transform(titanic['Sex']) #1 = Male, 2 = Female 
titanic['Embarked']= le.fit_transform(titanic['Embarked']) # 0 = C,1 = Q, 2 = S

#MODEL BUILDING
from sklearn.ensemble import RandomForestClassifier as rfc
model = rfc(criterion = "entropy")

#train test splitting
train = titanic.iloc[:,1:]
test = titanic.iloc[:,0]
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(train,test,test_size = 0.3, random_state = 123)

#Fit & Prediction  
model.fit(train_x,train_y)
pred = model.predict(test_x)

#Accuracy
from sklearn.metrics import accuracy_score
print (accuracy_score(test_y,pred)) #0.805%


# =============================================================================
#  Testing data on trained model
# =============================================================================

Finaltest = pd.read_csv("E://Samuel Data//DATA//Kaggle//Titanic//test.csv")
Finaltest.isna().sum()

#Data Preprocessing
Finaltest.Embarked.value_counts()
Finaltest["Embarked"].fillna("S", inplace = True) 
Finaltest["Fare"].fillna(np.mean(Finaltest["Fare"]), inplace = True) 
Finaltest["Age"].fillna(0, inplace =True)
testnullage = Finaltest[Finaltest.Age == 0] 
testmisage =set() #36 null values for Miss (girl child) 
testmastage = set() #4 null values for Master (boy child)
testMrage = set() #119 null values for Mr.(Men)
testMrsage = set() #17 null values for Mrs (women)
testmiscage = set() #1 null values for unknown(Dr.)

for i in testnullage.Name: 
    if "Miss" in i:
        testmisage.add(i) 
    elif "Master" in i:
        testmastage.add(i) 
    elif "Mr." in i:
        testMrage.add(i)
    elif "Mrs" in i:
        testMrsage.add(i)
    else:
        testmiscage.add(i)  

"""Filling the Null values of Age w.r.t the values that are analyzed and confirmed
    Miss = 10, Master = 5, Mr = 26,Mrs = 24, doc = 40"""
    
for i in testnullage.Name:
    if "Miss" in i:
        Finaltest.Age.iloc[(getIndexes(testnullage, i))] = 10
    elif "Master" in i:
        Finaltest.Age.iloc[(getIndexes(testnullage, i))] = 5
    elif "Mr." in i:
        Finaltest.Age.iloc[(getIndexes(testnullage, i))] = 26
    elif "Mrs" in i:
        Finaltest.Age.iloc[(getIndexes(testnullage, i))] = 24
    else:
        Finaltest.Age.iloc[(getIndexes(testnullage, i))] = 40
        
#Dropping categorical Variables   
Finaltest.drop("Cabin", axis = 1, inplace = True) 
Finaltest.drop("Ticket", axis = 1, inplace = True) 
Finaltest.drop("Name", axis = 1, inplace = True) 
Finaltest.drop("PassengerId", axis = 1, inplace = True) 

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
Finaltest['Sex']= le.fit_transform(Finaltest['Sex']) #1 = Male, 2 = Female 
Finaltest['Embarked']= le.fit_transform(Finaltest['Embarked']) # 0 = C,1 = Q, 2 = S

test_pred = model.predict(Finaltest)

#import Comparision Data
test_compare = pd.read_csv("E://Samuel Data//DATA//Kaggle//Titanic//gender_submission.csv")
test_compare.isnull().sum() 
test_compare["Survived"]

#comparing for accuracy
print(accuracy_score(test_compare["Survived"],test_pred)) #82.5%

"""Final Accuracy : 82.5%"""




