from numpy import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

print("train data shape is : ", train_data.values.shape)
print("test data shape is : ", test_data.values.shape)

trainY = ravel(train_data.values[:10000, 0])
trainX = train_data.values[:10000, 1:]
trainX[trainX > 0] = 1
print("trainX[0] is :")
print(trainX[0].reshape((28, 28)))

validY = ravel(train_data.values[40000:, 0])
validX = train_data.values[40000:, 1:]
validX[validX > 0] = 1
print("validX[0] is :")
print(validX[0].reshape((28, 28)))

testX = test_data.values
testX[testX > 0] = 1
print("testX[0] is :")
print(testX[0].reshape((28, 28)))


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(trainX, trainY)

#cross validation
predictions = clf.predict(validX)
accuracy = accuracy_score(validY, predictions)
print("the accuracy of kNN is : %f" % accuracy)

#predict test dataset
predictions = clf.predict(testX)

#save predictions
kNN_Result = pd.DataFrame(predictions, index=range(1, predictions.shape[0]+1))
kNN_Result.columns=["'Labels'"]
kNN_Result.index.name = "'ImageId'"
kNN_Result.to_csv("kNN_Result.csv")

