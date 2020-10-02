# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model, ensemble
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

titanicData = pd.read_csv('../input/train.csv')
titanicDataTest = pd.read_csv('../input/test.csv')
titanicData.to_html('titanic_html',na_rep='')

tmpDataFrame = titanicData[['Pclass','Sex','Fare','Age','SibSp','Parch','Embarked']].replace(['male','female','S','C','Q'],[1,2,1,2,3])
tmpDataFrameTesting = titanicDataTest[['Pclass','Sex','Fare','Age','SibSp','Parch','Embarked']].replace(['male','female','S','C','Q'],[1,2,1,2,3])

tmpVectorOutput_training = titanicData[['Survived']].as_matrix()

#I have labeled males as 1 and females as 2 in this model. I impute any unavailable data with mean of the remaining data
tmpVectorTraining = tmpDataFrame.as_matrix()
imputer = Imputer()
trans_tmpVectorTraining = imputer.fit_transform(tmpVectorTraining)

# features have been shifted to a center of mean and scaled to the max - min of data
MeanData = trans_tmpVectorTraining.mean(axis = 0)
MaxData = trans_tmpVectorTraining.max(axis=0)
MinData = trans_tmpVectorTraining.min(axis=0)
trans_tmpVectorTraining = (trans_tmpVectorTraining-MeanData) / (MaxData - MinData)

# feature scaling of the test data
tmpVectorTesting = tmpDataFrameTesting.as_matrix()
imputer = Imputer()
trans_tmpVectorTesting = imputer.fit_transform(tmpVectorTesting)
MeanDataTesting = trans_tmpVectorTesting.mean(axis = 0)
MaxDataTesting = trans_tmpVectorTesting.max(axis=0)
MinDataTesting = trans_tmpVectorTesting.min(axis=0)
trans_tmpVectorTesting = (trans_tmpVectorTesting-MeanDataTesting) / (MaxDataTesting - MinDataTesting)

c,r = tmpVectorOutput_training.shape
tmpVectorOutput_training = tmpVectorOutput_training.reshape(c,)

# apply logistic regression to get a feel for the data
regr = linear_model.LogisticRegression()
accuracyModel = cross_val_score(regr,trans_tmpVectorTraining,tmpVectorOutput_training,scoring='accuracy')
regr.fit(trans_tmpVectorTraining,tmpVectorOutput_training)
 
survivalPrediction = regr.predict(trans_tmpVectorTraining)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: ", mean_squared_error(tmpVectorOutput_training, survivalPrediction))
print("R Squared: ",r2_score(tmpVectorOutput_training,survivalPrediction))
print('The model accuracy is: ', accuracyModel.mean())

# apply random forest classifier
randomForestCLF = RandomForestClassifier(max_depth=6,random_state=0)
accuracy_randomForest = cross_val_score(randomForestCLF, trans_tmpVectorTraining, tmpVectorOutput_training, scoring='accuracy')
randomForestCLF.fit(trans_tmpVectorTraining,tmpVectorOutput_training)
survivalPrediction_randomCLF = randomForestCLF.predict(trans_tmpVectorTraining)

print('Random Forest Coefficients: \n', randomForestCLF.feature_importances_)
print("Random Forest Mean squared error: ", mean_squared_error(tmpVectorOutput_training, survivalPrediction_randomCLF))
print("Random Forest R Squared: ",r2_score(tmpVectorOutput_training,survivalPrediction_randomCLF))
print('Random Forest the model accuracy is: ', accuracy_randomForest.mean())

# apply a grid search on SVM
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='rbf',max_iter=20000,probability=True), param_grid)
grid_search.fit(trans_tmpVectorTraining,tmpVectorOutput_training)
bestEst = grid_search.best_estimator_
accuracy_svm_gridSearch = cross_val_score(bestEst, trans_tmpVectorTraining, tmpVectorOutput_training, scoring='accuracy')
survivalPrediction_svm_gridSearch = bestEst.predict(trans_tmpVectorTraining)
print('SVM Coefficients: \n', grid_search.best_params_)
print("SVM Mean squared error: ", mean_squared_error(tmpVectorOutput_training, survivalPrediction_svm_gridSearch))
print("SVM R Squared: ",r2_score(tmpVectorOutput_training,survivalPrediction_svm_gridSearch))
print('SVM the model accuracy is: ', accuracy_svm_gridSearch.mean())

# use a voting mechanism to check the accuracy of predictions
vclf = VotingClassifier(estimators=[('LR', regr), ('RF', randomForestCLF), ('SVM', bestEst)],voting='soft')
vclf.fit(trans_tmpVectorTraining,tmpVectorOutput_training)
accuracy_Voting = cross_val_score(vclf, trans_tmpVectorTraining, tmpVectorOutput_training, scoring='accuracy')
survivalPrediction_voting = vclf.predict(trans_tmpVectorTraining)

print("Voting Mean squared error: ", mean_squared_error(tmpVectorOutput_training, survivalPrediction_voting))
print("Voting R Squared: ",r2_score(tmpVectorOutput_training,survivalPrediction_voting))
print('Voting the model accuracy is: ', accuracy_Voting.mean())

# the svm and voting methods yield similar results, but I choose to use the SVM.
survivalPredictionTest_svm_gridSearch = bestEst.predict(trans_tmpVectorTesting)
testPassengerId = titanicDataTest['PassengerId'].as_matrix()
tmp = {'PassengerId':testPassengerId,'Survived':survivalPredictionTest_svm_gridSearch}
tmpDf = pd.DataFrame(data=tmp)
tmpDf.to_csv('submission.csv',index=False)