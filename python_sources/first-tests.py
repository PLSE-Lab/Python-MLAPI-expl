import numpy as np
import pandas as pd

Euler=2.718281828459045235360287
#sigmoid
def sigmoid(x):
    return 1/(1+Euler**(-x))

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
#Modifying values
train['Sex'] = train['Sex'].replace('male',1)
train['Sex'] = train['Sex'].replace('female',0)
train['Age'] = train['Age'].fillna(train['Age'].mean())
agemean=train['Age'].mean()
agestd=train['Age'].std()
train['Age']=(train.Age-agemean)/agestd
faremean = train['Fare'].mean()
farestd = train ['Fare'].std()
train['Fare']=(train.Fare-faremean)/farestd
#Turning a Class into 3
train['C'] = (train.Embarked=='C').astype(int)
train['Q'] = (train.Embarked=='Q').astype(int)
train['S'] = (train.Embarked=='S').astype(int)
train['Fix'] = (train.Embarked!='K').astype(int)
del train['Embarked']
train['Age2']=train.Age*train.Age
train['Fare2']=train.Fare*train.Fare
train['FareAge']=train.Age*train.Fare
train['AgeSex']=train.Age*(train.Sex-0.5)

#Start Theta
bestTheta = Theta = pd.Series(np.random.randn(train.shape[1]),list(train))
alpha = beta = bestMatch = 0.0000000003
#Process Theta
alpha = 0.0003
beta = 0.000000003
'''
while alpha<30:
    alpha = alpha * 3
    beta = 8.1e-9
    while beta<3:
        beta = beta * 3
        Theta = pd.Series(np.random.randn(train.shape[1]),list(train))
'''
for rotation in range(1,1000):
    Pred=sigmoid(train.dot(Theta))
    ThetaChange=(anstrain-Pred).transpose().dot(train)
    ThetaChange=ThetaChange
    Theta=Theta+ThetaChange*alpha-beta*Theta*Theta
    if rotation%1000==0:
        print(rotation/100)
        '''
        rotMatch = (891-sum((Pred!=anstrain).astype(int)))/891
        if rotMatch > bestMatch:
            bestTheta = Theta
            bestMatch = rotMatch
            print(bestMatch)
        print(beta)
        print(alpha)
'''
Pred=sigmoid(train.dot(Theta))
train['Prediction']=(Pred)
train['Pred']=(Pred>0.5).astype(int)
train['Ans']=anstrain
print(sum((train.Pred!=train.Ans).astype(int)))
#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

train['Survived']=anstrain
PT=train[train.Survived==1].plot.scatter(x='AgeSex', y='Fare2', color='DarkBlue', label='Survived')
train[train.Survived==0].plot.scatter(x='AgeSex', y='Fare2', color='LightBlue', label='NSurvived',ax=PT)


print("Stat")
aux=train[train.Sex==1]
print(aux.count())
print(aux[aux.Survived==1].count())
aux=train[train.Sex==0]
print(aux.count())
print(aux[aux.Survived==1].count())






del test['Name']
del test['Ticket']
del test['Cabin']
del test['PassengerId']
#Modifying values
test['Sex'] = test['Sex'].replace('male',1)
test['Sex'] = test['Sex'].replace('female',0)
test['Age'] = test['Age'].fillna(agemean)
test['Age'] = (test.Age-agemean)/agestd
test['Fare'] = (test.Fare-faremean)/farestd
#Turning a Class into 3
test['C'] = (test.Embarked=='C').astype(int)
test['Q'] = (test.Embarked=='Q').astype(int)
test['S'] = (test.Embarked=='S').astype(int)
test['Fix'] = (test.Embarked!='K').astype(int)
del test['Embarked']
test['Age2']=test.Age*test.Age
test['Fare2']=test.Fare*test.Fare
test['FareAge']=test.Age*test.Fare
test['AgeSex']=test.Age*(test.Sex-0.5)




testOut2=(sigmoid(test.dot(Theta))>0.5).astype(int)
testOut=pd.DataFrame([])
testOut['Survived']=testOut2
testOut['PassengerId']=Stest.PassengerId
testOut.to_csv('Pred.csv',header=['Survived','PassengerId'], index=False)