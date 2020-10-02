#dataset --> https://www.kaggle.com/c/afsis-soil-properties/data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import time
tic=time.time()
fulldata=pd.read_csv("../input/training.csv")
fulldata=fulldata.sample(frac=1)#randomize the whole dataset
fullfeatures=fulldata.drop(["Ca","P","pH","SOC","Sand"],axis=1)
fulllabels=pd.DataFrame(fulldata[["PIDN","Ca","P","pH","SOC","Sand"]])
fullfeaturesarray=fullfeatures.values
fulllabelsarray=fulllabels.values
trainfeatures,testfeatures,trainlabels,testlabels=train_test_split(fullfeaturesarray,fulllabelsarray,train_size=957)

trainfeaturesindex=trainfeatures[...,0]
trainfeaturesdata=trainfeatures[...,1:]

testfeaturesindex=testfeatures[...,0]
testfeaturesdata=testfeatures[...,1:]

trainlabelsindex=trainlabels[...,0]
trainlabelsdata=trainlabels[...,1:]

testlabelsindex=testlabels[...,0]
testlabelsdata=testlabels[...,1:]

regression_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=10,n_estimators=50,n_jobs=-1))
regression_multirf.fit(trainfeaturesdata,trainlabelsdata)
predtrainlabelsdata=regression_multirf.predict(trainfeaturesdata)
predtestlabelsdata=regression_multirf.predict(testfeaturesdata)
error =mean_squared_error(trainlabelsdata, predtrainlabelsdata, sample_weight=None, multioutput='uniform_average')
print("Train RMSE AVG ERROR",error)
error =mean_squared_error(trainlabelsdata, predtrainlabelsdata, sample_weight=None, multioutput='raw_values')
print("Train RMSE RAW ERROR",error)
error =mean_squared_error(testlabelsdata, predtestlabelsdata, sample_weight=None, multioutput='uniform_average')
print("Test RMSE AVG ERROR",error)
error =mean_squared_error(testlabelsdata, predtestlabelsdata, sample_weight=None, multioutput='raw_values')
print("Test RMSE RAW ERROR",error)

outputfeatures=pd.read_csv("../input/sorted_test.csv")
outputfeaturesarray=outputfeatures.values
outputfeaturesarrayindex=outputfeaturesarray[...,0]
outputfeaturesarraydata=outputfeaturesarray[...,1:]
outputpredlabelsdata=regression_multirf.predict(outputfeaturesarraydata)
outptutarray=np.column_stack((outputfeaturesarrayindex,outputpredlabelsdata))
output=pd.DataFrame(outptutarray,columns=["PIDN","Ca","P","pH","SOC","Sand"])
output.set_index(keys=["PIDN"],inplace=True)
output.to_csv("output.csv",encoding='utf-8')
toc=time.time()
elapsedtime=toc-tic
print("Time Taken :",elapsedtime)
