import pandas as pd
import csv
import numpy as np
dataset=pd.read_csv("../input/titanic-dataset/train (1).csv")
testset=pd.read_csv("../input/titanic-dataset/test.csv")
trainedSurvived=dataset.iloc[:,:1].values
trainedSex=dataset.iloc[:,1].values
sexForPredict=testset.iloc[:,3:4].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(trainedSurvived,trainedSex)
predictions=np.around(regressor.predict(sexForPredict))
data_to_submit = pd.DataFrame({'PassengerId':testset['PassengerId'],'Survived':predictions})
data_to_submit.to_csv('titanicSubmission.csv', index = False)

