import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)



train['Age']=train['Age'].fillna(train['Age'].median())
train['age']=0
#train['age'][train['Age']>24 and train['Sex']=='female'] = 1
train['age'][train['Age']>24] = 0
train['age'][train['Age']<=24] = 1
train['sex'] = 0
train['sex'][train['Sex'] =='female'] = 1
train['pclass']=1
train['pclass'][train['Pclass']==3] = 0
train['Fare'][train['Survived']==1].value_counts()


train['family']=0
for i in range(0,len(train)):
    if train['SibSp'][i]+train['Parch'][i]<=2:
        train['family'][i]=1

		
#test = pd.read_csv("test.csv")
test['Age']=test['Age'].fillna(test['Age'].median())

test['age']=0

test['age'][test['Age']<=24] = 1

test['sex'] = 0
test['sex'][test['Sex'] =='female'] = 1
test['pclass']=1

test['pclass'][test['Pclass']==3] = 0
test['family'] = 0

for i in range(0,len(test)):
    if test['SibSp'][i]+test['Parch'][i]>2:
        test['family'][i]=1

arr = test[['sex','age','pclass','family']]


features = train[['sex','age','pclass','family']].values
target = train['Survived'].values
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(n_estimators=100)
rf.fit(features,target)



rf.predict(arr)