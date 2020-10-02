# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:28:11 2016

@author: J2Letters

Some blocks of code are mentioned as being from : 
https://www.kaggle.com/zgo2016/titanic/titanic-logistic-regression/code, credit to its author zgo2016 (check his code for cool charts)

"""

import csv as csv
import pandas as pd
import numpy as np
import pylab as p
from scipy.optimize import fmin
from random import shuffle, seed
import re

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv("../input/train.csv", header=0)
dftest = pd.read_csv("../input/test.csv", header=0)

#logistic/sigmoid function
def g(z):
    return 1/(1+np.exp(-z))

#cost function
def J(theta,x,y,m,lam): #theta is an array cause that's how the fmin works
    unregularized = ((-1/m)*((y.mul(np.log(h(theta,x)))).add((1-y).mul(np.log(1-h(theta,x)))))).sum()
    regularization = (lam/(2*m))*(np.power(theta,2).sum())
    return unregularized+regularization

#hypothesis
def h(theta,x):
    theta = np.array(theta)
    return g(x.dot(theta)) #to do a matrix product that result in a Series with the searched values
    
#predictive function
def prediction(theta,x):
    hval = h(theta,x)
    return hval.map(lambda v : threshold(v))
    
def threshold(v):
    if v>=0.5:
        res = 1
    else:
        res = 0
    return res
    
def normalize(series):
    mean = series.mean()
    std = series.std()
    return (series-mean)/std
    
# This function returns the title from a name. function code from : https://www.kaggle.com/zgo2016/titanic/titanic-logistic-regression/code
def title(name):
    # Search for a title using a regular expression. Titles are made of capital and lowercase letters ending with a period.
    find_title = re.search(' ([A-Za-z]+)\.', name)
    # Extract and return the title If it exists. 
    if find_title:
        return find_title.group(1)
    return ""
    
def cleaningAndPreparing():
    
    #Following block of code from https://www.kaggle.com/zgo2016/titanic/titanic-logistic-regression/code   
    # Generate new feature Namelength.
    df["NameLength"] = df["Name"].apply(lambda x: len(x))    
    # Get all titles.
    titles = df["Name"].apply(title)
    # Mapping possible titles to integer values. Some titles are compressed and share the same title codes since they are rare.  
    map_title = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for i,j in map_title.items():
        titles[titles == i] = j
    # Add title values to corresponding column.
    df["Title"] = titles
    df.Title=df.Title.astype(float)
    
    
    #Cleaning the training data
    df['Constant']=1
    #cleaning Age
    df['AgeCleaned']=df['Age']
    df.loc[df.Age.isnull(),'AgeCleaned']=df['Age'].dropna().median()
    #cleaning Sex    
    df['Gender']=df['Sex'].map({'male':1,'female':0})
    #cleaning Embarked
    df['EmbarkedCleaned'] = df['Embarked']
    df['EmbarkedCleaned'] = df['Embarked'].map({'C':0,'S':1,'Q':2})
    df.loc[df.Embarked.isnull(),'EmbarkedCleaned']=df.EmbarkedCleaned.median()
    #cleaning Fare (nothing to do actually for the training set huhu)
    df['FareCleaned']=df.Fare
    #Normalizing the data (don't know yet if useful, for the sake of practice)
    df.AgeCleaned = normalize(df.AgeCleaned)
    df.EmbarkedCleaned = normalize(df.EmbarkedCleaned)
    df.Pclass = normalize(df.Pclass)
    df.Fare = normalize(df.Fare)
    df.SibSp = normalize(df.SibSp)
    df.NameLength = normalize(df.NameLength)
    df.Title = normalize(df.Title)
    
    
    #Also preparing/cleaning the test data
    # Generate new feature Namelength.
    dftest["NameLength"] = dftest["Name"].apply(lambda x: len(x))    
    # Get all titles.
    titles = dftest["Name"].apply(title)
    # Mapping possible titles to integer values. Some titles are compressed and share the same title codes since they are rare.  
    map_title = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for i,j in map_title.items():
        titles[titles == i] = j
    # Add title values to corresponding column.
    dftest["Title"] = titles   
    dftest.Title=dftest.Title.astype(float)
    
    dftest['Constant']=1
    #cleaning Age
    dftest['AgeCleaned']=dftest['Age']
    #dftest['AgeCleaned'] = dftest['AgeCleaned'].str.replace(',','.') # still containing strings though 
    #dftest['AgeCleaned'] = pd.to_numeric(dftest['AgeCleaned']) # not containing strings anymore
    dftest.loc[dftest.Age.isnull(),'AgeCleaned']=dftest['AgeCleaned'].dropna().median() 
    #cleaning Fare    
    dftest['FareCleaned']=dftest['Fare']
    #dftest['FareCleaned']= dftest['FareCleaned'].str.replace(',','.')
    #dftest['FareCleaned'] = pd.to_numeric(dftest['FareCleaned'])
    #cleaning Sex
    dftest['Gender']=dftest['Sex'].map({'male':1,'female':0})
    #cleaning Embarked
    dftest['EmbarkedCleaned'] = dftest['Embarked']
    dftest['EmbarkedCleaned'] = dftest['Embarked'].map({'C':0,'S':1,'Q':2})
    dftest.loc[dftest.Embarked.isnull(),'EmbarkedCleaned']=dftest.EmbarkedCleaned.median()
    #Normalizing
    dftest.AgeCleaned = normalize(dftest.AgeCleaned)
    dftest.EmbarkedCleaned = normalize(dftest.EmbarkedCleaned)
    dftest.Pclass = normalize(dftest.Pclass)
    dftest.FareCleaned = normalize(dftest.FareCleaned)
    dftest.SibSp = normalize(dftest.SibSp)
    dftest.NameLength = normalize(dftest.NameLength)
    dftest.Title = normalize(dftest.Title)
    
def main():
    cleaningAndPreparing()    
    
    #Dividing the train data into train (60%), cross-validation(20%) and trainTest(20%) set    
    #take the index (rangeIndex) values and convert it to a list then apply the shuffle function from random package
    listIndexes = df.index.values.tolist()
    seed(10) # set the seed so that the shuffle is always the same
    shuffle(listIndexes) #returns None
    mdf = float(df.shape[0])
    twentyPercent = int(mdf*0.2)
    firstTwentyPercent = listIndexes[0:twentyPercent]
    nextTwentyPercent = listIndexes[twentyPercent:twentyPercent*2]
    lastSixtyPercent = listIndexes[twentyPercent*2::]
    trainTest = df.loc[firstTwentyPercent]
    crossVal = df.loc[nextTwentyPercent]
    train = df.loc[lastSixtyPercent]
    
    #Uncomment the following to make a prediction on all the training set    
    train = df #warning this make the crossVal and trainTest senseless
    
    m = float(train.shape[0])
    y = train['Survived']
    predicVar = ['Constant','Pclass','AgeCleaned','FareCleaned','SibSp','Gender','EmbarkedCleaned','NameLength','Title']
    nbPredicVar = len(predicVar)
    x = train[predicVar]
    lam = 0
    
    # solves the minimization problem, finds the vector theta thanks to the train data
    # fmin is not enough if there are many dimensions
    liste0 = [0]*nbPredicVar
    thetaSol = fmin(J,liste0,(x,y,m,lam))
    
    #Now for the test data, we apply our predictive formula on the test data 
    #Extracting the variables
    #xTest = dftest[['Constant','Gender']]    
    #xTest = dftest[['Constant','Pclass','AgeCleaned','Gender']]
    xTest = dftest[predicVar]
    
    #Predicting
    predTest = prediction(thetaSol,xTest)
    
    #We need to play with predTest a bit to have a dataFrame with the values of columns PassengerId and Survived
    dftest['Survived']=predTest
    predTest = dftest[['PassengerId','Survived']]
    
    #predTestCsv = predTest.to_csv("pclassLinRegTestModel.csv")
    prediction_file = open("RegLogisticRegModel04112016.csv", "w")
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])    
    prediction_file_object.writerows(predTest.values)
    
    '''
    x = train[predicVar]
    predTrain = prediction(thetaSol,x)
    misclassError = len(train.loc[predTrain!=train.Survived,'Survived'])/m
    #print 'train misclassification error : {0}, J train : {1}'.format(misclassError, J(thetaSol,x,y,m,lam))
    
    xCrossVal = crossVal[predicVar]
    predCrossVal = prediction(thetaSol,xCrossVal)
    mCV = float(crossVal.shape[0])
    misclassErrorCV = len(crossVal.loc[predCrossVal!=crossVal.Survived,'Survived'])/mCV
    #print 'crossVal misclassification error : {0}, J cross val : {1}'.format(misclassErrorCV, J(thetaSol,xCrossVal,crossVal.Survived,mCV,lam))
    
    xTrainTest = trainTest[predicVar]
    predTrainTest = prediction(thetaSol,xTrainTest)
    mTrainTest = float(trainTest.shape[0])
    misclassErrorTrainTest = len(trainTest.loc[predTrainTest!=trainTest.Survived,'Survived'])/mTrainTest
    #print 'trainTest misclassification error : {0}, J train test : {1}'.format(misclassErrorTrainTest, J(thetaSol,xTrainTest,trainTest.Survived,mTrainTest,lam))
    '''
main()