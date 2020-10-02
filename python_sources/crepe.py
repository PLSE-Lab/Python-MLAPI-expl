import numpy as np
import pandas as pd

Euler=2.718281828459045235360287
#sigmoid
def sigmoid(x):
    return 1/(1+Euler**(-x))

alpha=3*(10**-5)
regulariz=0.000003

 
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
Stest = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
anstrain = train['Survived']

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Clear And prepare Data
#Deleting Columns
del train['Name']
del train['Survived']
del train['Ticket']
del train['Cabin']
del train['PassengerId']
#Normalizing values
#Modifying values
train['Sex'] = train['Sex'].replace('male',1)
train['Sex'] = train['Sex'].replace('female',0)
train['Age'] = (train['Age'].fillna(np.ceil(train['Age'].mean()))-(train['Age'].mean()))*(1/train['Age'].std())
train['Fare'] = (train['Fare']-(train['Fare'].mean()))/(train['Fare'].std())
#Turning a Class into 3
train['C'] = (train.Embarked=='C').astype(int)
train['Q'] = (train.Embarked=='Q').astype(int)
train['S'] = (train.Embarked=='S').astype(int)
train['Fix'] = (train.Embarked!='K').astype(int)

del train['Embarked']

#Start Theta
Theta = [0]*train.shape[1]

#J    
def J(x):
    part1 = (anstrain.dot(np.log(sigmoid(train.dot(x))))).sum()
    part2 = ((1-anstrain).dot(np.log(sigmoid(train.dot(x))))).sum()
    return (-1/891)*(part1 + part2)

#Process Theta
for rotation in range(1,10000):
    Pred = sigmoid(train.dot(Theta))
    Theta = Theta - alpha*(train.transpose().dot(Pred-anstrain))
    if rotation%100 == 0:
        print(rotation)
        print(J(Theta))

train['Prediction']=(Pred)
train['Pred']=(Pred>0.5).astype(int)
train['Ans']=anstrain
print(sum((train.Pred!=train.Ans).astype(int)))
#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

#Preparing submisson
del test['Name']
del test['Ticket']
del test['Cabin']
del test['PassengerId']
#Modifying values
test['Sex'] = test['Sex'].replace('male',1)
test['Sex'] = test['Sex'].replace('female',0)
test['Age'] = test['Age'].fillna(test['Age'].mean())
#Turning a Class into 3
test['C'] = (test.Embarked=='C').astype(int)
test['Q'] = (test.Embarked=='Q').astype(int)
test['S'] = (test.Embarked=='S').astype(int)
test['Fix'] = (test.Embarked!='K').astype(int)
del test['Embarked']
#Predicting values

testOut2=(sigmoid(test.dot(Theta))>0.5).astype(int)
#Organizing matrix output
testOut=pd.DataFrame([])
testOut['Survived']=testOut2
testOut['PassengerId']=Stest.PassengerId
#output
testOut.to_csv('Pred.csv',header=['Survived','PassengerId'], index=False)