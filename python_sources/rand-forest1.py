import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#train=pd.read_csv('train.csv')
#print(train.head())
train["Embarked"] = train["Embarked"].fillna("S")
#train['Fam']=1+train['SibSp']+train['Parch']
train['Child'] = (train['Age'] <= 10).astype(int)

train.drop(['PassengerId','Name','Cabin','Ticket','Parch','SibSp','Age'],axis=1,inplace=True)
print(train.head())
#train["Sex"][train["Sex"] == "male"] = 0
#train["Sex"][train["Sex"] == "female"] = 1

train['Sex'][train['Sex']=='male']=0
train['Sex'][train['Sex']=='female']=1

train['Embarked'][train['Embarked']=='S']=0
train['Embarked'][train['Embarked']=='C']=1
train['Embarked'][train['Embarked']=='Q']=2
#for i in range(100):
#    print i
#print(train['Parch'][train['Survived']==0].value_counts())
#print(train['Parch'][train['Survived']==1].value_counts())
#train['Parch'][train['Parch']==0]=0
#train['Parch'][train['Parch']==1]=1
#train['Parch'][train['Parch']==2]=1
#train['Parch'][train['Parch']==3]=1
#train['Parch'][train['Parch']==4]=0
#train['Parch'][train['Parch']==5]=0
#train['Parch'][train['Parch']==6]=0

#train["Age"]=train["Age"].fillna(train["Age"].median())
train["Fare"]=train["Fare"].fillna(train["Fare"].median())
y=train['Survived']
#print(y)
train.drop('Survived',axis=1,inplace=True)
X=train.as_matrix()
#print(X)

forest=RandomForestClassifier(n_estimators=200)
forest.fit(X,y)
#print(train.head())
#print(train.head())

#test=pd.read_csv('test.csv')
#test['Fam']=1+test['Parch']+test['SibSp']
test['Child'] = (test['Age'] <= 10).astype(int)

test["Embarked"] = test["Embarked"].fillna("S")
passid=test['PassengerId']
test.drop(['PassengerId','Name','Cabin','Ticket','Parch','SibSp','Age'],axis=1,inplace=True)
test['Sex'][test['Sex']=='male']=0
test['Sex'][test['Sex']=='female']=1

test['Embarked'][test['Embarked']=='S']=0
test['Embarked'][test['Embarked']=='C']=1
test['Embarked'][test['Embarked']=='Q']=2

#test['Parch'][test['Parch']==0]=0
#test['Parch'][test['Parch']==1]=1
#test['Parch'][test['Parch']==2]=1
#test['Parch'][test['Parch']==3]=1
#test['Parch'][test['Parch']==4]=0
#test['Parch'][test['Parch']==5]=0
#test['Parch'][test['Parch']==6]=0

#test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test["Pclass"]=test["Pclass"].fillna(test["Pclass"].median())

F=test.as_matrix()
prdn=forest.predict(F)
#print(prdn)
#df1=pd.DataFrame(prdn)
#df2=pd.DataFrame(passid)
#final=pd.concat([df2,df1],axis=1)
#final.columns=['PassengerId','Survived']
#print(final.head())
#final.to_csv('Titanic.csv',index_label=['PassengerId'])


arr1=np.array(passid).astype(int)
final=pd.DataFrame(prdn,arr1,columns=['Survived'])
print(final)
final.to_csv('Titanic_rf4.csv',index_label='PassengerId')






#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)