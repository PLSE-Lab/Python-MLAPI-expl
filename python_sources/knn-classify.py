import numpy as np
import pandas as pd
import operator
from sklearn import preprocessing

#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Pclass": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Pclass": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
#train = pd.read_csv("../input/train.csv",header=1,sep='\r\s+',engine='python')
def autoNorm(dataset):
    maxvals = dataset.max()
    minvals = dataset.min()
    
    ranges = maxvals - minvals
    print(maxvals)
    

train = pd.read_csv("../input/train.csv",header=0)
train['Sex'] = train['Sex'].replace({'female':2,'male':1})
train_new = train.drop(['PassengerId','Name','Cabin','Ticket','Embarked'], axis=1)
#train_new['Embarked'] = train_new['Embarked'].fillna(train['Embarked'].value_counts().argmax())

#age = train.Age
train_new['Age'] = train_new['Age'].fillna(value=train_new.Age.mean())
age=train_new.Age
autoNorm(age)

#train_new['Age'] = preprocessing.StandardScaler().fit_transform(train_new['Age'])
#print(age_max)



        
#kNN function
def kNN_classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]