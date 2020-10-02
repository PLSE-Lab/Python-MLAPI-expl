import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

print(train.head())


columns = train.columns.tolist()
toRemove = ["Survived","Name","Ticket","Cabin"]
for x in toRemove:
    columns.remove(x)


trainX = train[columns]
trainX.Sex.replace(to_replace=dict(female=1.0, male=0.0), inplace=True)
trainX.Embarked.replace(to_replace={"S":0.0,"C":1.0,"Q":2.0}, inplace=True)
trainX.replace(to_replace={np.nan:0.0}, inplace=True)
trainX = trainX.as_matrix()
# trainX = np.ndarray(shape=trainX.shape,buffer=trainX)
trainY = train['Survived']
trainY = trainY.as_matrix()
l = (9.0/10.0)*len(trainX)
validationX = trainX[l:]
validationY = trainY[l:]
trainX = trainX[:l]
trainY = trainY[:l]


# trainY = np.ndarray(shape=trainY.shape,buffer=trainY)

testX = test[columns]
testX.Sex.replace(to_replace=dict(female=1.0, male=0.0), inplace=True)
testX.Embarked.replace(to_replace={"S":0.0,"C":1.0,"Q":2.0}, inplace=True)
testX.replace(to_replace={np.nan:0.0}, inplace=True)
# testX = testX.as_matrix()
# testY = test["Survived"]

# print(trainX.head(5))
# print(trainY.head(5))
# print(testX.head(500))
# print(test["Survived"])




mlp = MLPClassifier(algorithm='l-bfgs', alpha=1e-6, hidden_layer_sizes=(100), random_state=np.random.randint(0,10000),learning_rate_init=0.01,max_iter=10000,early_stopping=False)
mlp.fit(trainX, trainY)

rf = RandomForestClassifier(max_depth=6)
rf.fit(trainX,trainY)

params = {'n_estimators': 1000, 'max_depth': 3, 'subsample': 0.7,
          'learning_rate': 0.1, 'min_samples_leaf': 4, 'random_state': 3}

gb = GradientBoostingClassifier(**params)
gb.fit(trainX,trainY)

print(mlp.score(trainX,trainY))
print(rf.score(trainX,trainY))
print(gb.score(trainX,trainY))


mlpPreds = mlp.predict(testX)
rfPreds = rf.predict(testX)
gbPreds = gb.predict(testX)
# clf2 = MLPClassifier(algorithm='l-bfgs', alpha=1e-4, hidden_layer_sizes=(80,10), random_state=np.random.randint(0,1000),learning_rate_init=0.5,max_iter=5000,tol=1e-10,early_stopping=True)
# clf2.fit(trainX, trainY)
# preds2 = clf2.predict(testX)
# clf3 = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(80,10), random_state=np.random.randint(0,1000),learning_rate_init=0.5,max_iter=5000,tol=1e-10,early_stopping=True)
# clf3.fit(trainX, trainY)
# preds3 = clf3.predict(testX)
# clf4 = MLPClassifier(algorithm='l-bfgs', alpha=1e-6, hidden_layer_sizes=(80,10), random_state=np.random.randint(0,1000),learning_rate_init=0.5,max_iter=5000,tol=1e-10,early_stopping=True)
# clf4.fit(trainX, trainY)
# preds4 = clf4.predict(testX)
# clf5 = MLPClassifier(algorithm='l-bfgs', alpha=1e-7, hidden_layer_sizes=(80,10), random_state=np.random.randint(0,1000),learning_rate_init=0.5,max_iter=5000,tol=1e-10,early_stopping=True)
# clf5.fit(trainX, trainY)
# preds5 = clf5.predict(testX)



# preds = [float(((preds1[x]+preds2[x]+preds3[x]+preds4[x]+preds5[x])/5.0) >= 0.5) for x in range(len(preds1))]
# preds = preds1
# preds = [int(x) for x in gbPreds]
preds = [int(((mlpPreds[x] + rfPreds[x] + gbPreds[x])/3.0)>0.5) for x in range(len(rfPreds))]

testX["Survived"] = preds
# print(list(trainY))
print(preds)



# #Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())

# print("\n\nSummary statistics of training data")
# print(train.describe())

# #Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)
testX = testX[["PassengerId","Survived"]]
testX.to_csv('JacksTesting.csv',index=False)


