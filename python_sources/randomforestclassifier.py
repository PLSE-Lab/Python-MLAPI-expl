import pandas, numpy  
from sklearn.ensemble import RandomForestClassifier  
  
dataset = pandas.read_csv("../input/train.csv")  
trainData = dataset.iloc[:, 1:].values  
labelData = dataset.label.values  
testDataset = pandas.read_csv("../input/test.csv")  
testData = testDataset.iloc[:, :].values  
  
rfModel = RandomForestClassifier(n_estimators=100)  
rfModel.fit(trainData, labelData)  
preds = rfModel.predict(testData)  
numpy.savetxt('res.csv', numpy.c_[range(1, len(testData) + 1), preds], delimiter=',',  
              header='ImageId,Label', comments='', fmt='%d')  