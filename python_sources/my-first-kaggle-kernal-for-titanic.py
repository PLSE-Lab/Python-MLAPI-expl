
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import random
#This part is for training
train_data = pd.read_csv('../input/train.csv')
#converting the departure and sex data columns to dummies
dummied_train_data = pd.get_dummies(train_data, columns = ['Sex','Embarked'] )
dtd = dummied_train_data
#creation of Reverend variable
rev = []
for i in range(len(dtd)):
    name = dtd.at[i, 'Name']
    if 'Rev.' in name:
        rev.append(1)
    else:
        rev.append(0)
dtd['Reverend'] = rev
#creation of captain variable
capt = []
for i in range(len(dtd)):
    name = dtd.at[i, 'Name']
    if 'Capt' in name:
        capt.append(1)
    else:
        capt.append(0)
dtd['Captain'] = capt
#handling null values

#dropping
droppeddtd = dtd.dropna(subset = ['Age','Fare'])

#filling with mean for age
imputdtd = dtd
imputdtd.loc[(imputdtd.Age.isnull())&(imputdtd.Sex_male == 1),'Age']= imputdtd.Age[imputdtd.Sex_male == 1].mean()
imputdtd.loc[(imputdtd.Age.isnull())&(imputdtd.Sex_male == 0),'Age']= imputdtd.Age[imputdtd.Sex_male == 0].mean()
imputdtd.loc[(imputdtd.Fare.isnull())&(imputdtd.Pclass == 1), 'Fare'] =imputdtd.Fare[imputdtd.Pclass == 1].mean()
imputdtd.loc[(imputdtd.Fare.isnull())&(imputdtd.Pclass == 2), 'Fare'] =imputdtd.Fare[imputdtd.Pclass == 2].mean()
imputdtd.loc[(imputdtd.Fare.isnull())&(imputdtd.Pclass == 3), 'Fare'] =imputdtd.Fare[imputdtd.Pclass == 3].mean()
#creation of linear Regression Model
y = pd.DataFrame(droppeddtd.Survived, columns=['Survived'])
predictors = ['Age','SibSp','Parch','Fare','Sex_female','Embarked_C','Embarked_Q',
     'Reverend','Captain']
X = pd.DataFrame(droppeddtd, columns = predictors)
#Splitting the Dataset
trainX, testX, trainy, testy = train_test_split(X, y,random_state = 0)
#creation of targets with imputed ages
iy = pd.DataFrame(imputdtd.Survived, columns=['Survived'])
predictors = ['Age','SibSp','Parch','Fare','Sex_female','Embarked_C','Embarked_Q',
     'Reverend','Captain']
iX = pd.DataFrame(imputdtd, columns = predictors)
#split imputed
trainiX, testiX, trainiy, testiy = train_test_split(iX, iy,random_state = 0)
#MODELS

#Linear Regression
regmodel = linear_model.LinearRegression()
regmodel.fit(trainX,trainy)
predictions = regmodel.predict(testX)
#creation of binary predictor
binaried = []
for i in predictions:
    if i > .5:
        binaried.append(1)
    else:
        binaried.append(0)
#Percent correct
print("MAE for Linear Regression with Dropped NAN is:")
print(mean_absolute_error(testy, binaried))

#Logistic Regression
logitmodel = LogisticRegression()
trainy = np.ravel(trainy)
logitmodel.fit(trainX, trainy)
logitpredictions = logitmodel.predict(testX)
binaried = []
for i in logitpredictions:
    if i > .5:
        binaried.append(1)
    else:
        binaried.append(0)
#Percent correct
print("MAE for Logistic Regression with Dropped NAN is:")
print(mean_absolute_error(testy, binaried))

#Linear Regression with Imputed Ages and Fares
regmodel = linear_model.LinearRegression()
regmodel.fit(trainiX,trainiy)
predictions = regmodel.predict(testiX)
#creation of binary predictor
binaried = []
for i in predictions:
    if i > .5:
        binaried.append(1)
    else:
        binaried.append(0)
print("MAE for Linear Regression with imputed Ages and Fares is:")
print(mean_absolute_error(testiy, binaried))

#Logistic Regression with Imputed Ages and Fares
logitmodel = LogisticRegression()
trainiy = np.ravel(trainiy)
logitmodel.fit(trainiX, trainiy)
logitpredictions = logitmodel.predict(testiX)
binaried = []
for i in logitpredictions:
    if i > .5:
        binaried.append(1)
    else:
        binaried.append(0)
print("MAE for Logistic Regression with imputed ages and fares is:")
print(mean_absolute_error(testiy, binaried))   

