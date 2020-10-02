import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
 
 
titanic=pd.read_csv('../input/train.csv')
titanic.drop('PassengerId',axis='columns',inplace=True)

print(titanic.head())
#Understand Data
print(titanic[titanic.Survived==1].Sex.value_counts(normalize=True))
print(titanic[titanic.Survived==1].Parch.value_counts(normalize=True))
print(titanic[titanic.Survived==1].SibSp.value_counts(normalize=True))
print(titanic[titanic.Survived==1].Embarked.value_counts(normalize=True))  
print(titanic[titanic.Survived==1].Age.max())
print(titanic[titanic.Survived==1].Age.min())
print(titanic[titanic.Survived==0].Age.max())
print(titanic[titanic.Survived==0].Age.min())
print(titanic[titanic.Survived==1].Pclass.value_counts(normalize=True))

titanic[titanic.Survived==1].Age.value_counts().sort_index().plot()
plt.show()

titanic[np.logical_and(titanic.Survived==1,titanic.Pclass==1)].Age.value_counts().sort_index().plot(label='1')
titanic[np.logical_and(titanic.Survived==1,titanic.Pclass==2)].Age.value_counts().sort_index().plot(label='2')
titanic[np.logical_and(titanic.Survived==1,titanic.Pclass==3)].Age.value_counts().sort_index().plot(label='3')
plt.legend()
plt.show()
#Understand Data

fareScaler=MinMaxScaler()
embarkedEncoder=LabelEncoder()
titleEncoder=LabelEncoder()
lastNameEncoder=LabelEncoder()
ticketEncoder=LabelEncoder()

 
titanic['Sex'].replace(['male','female'],[1,0],inplace=True)
titanic['Embarked'].replace([np.nan],['C'],inplace=True)
titanic['Embarked']=embarkedEncoder.fit_transform(titanic['Embarked'])
titanic['Embarked']=titanic['Embarked']+1 
titanic['Fare']=fareScaler.fit_transform(pd.DataFrame(titanic['Fare'].values))  
titanic['Title']=titanic['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip()) 
titanic['LastName']=titanic['Name'].apply(lambda name: name.split(',')[0]) 
titanic['LastName']=lastNameEncoder.fit_transform(titanic['LastName'].values)
titanic['Title']=titleEncoder.fit_transform(titanic['Title']) 
titanic['Ticket']=ticketEncoder.fit_transform(titanic['Ticket'])
titanic['Cabin']=titanic['Cabin'].str[0]
print(titanic[titanic.Survived==1].Embarked.value_counts(normalize=True))

################################ AGE MODEL ####################################
from sklearn.ensemble import RandomForestRegressor
age=titanic.loc[titanic['Age'].notna(),:]
x=age[['SibSp','Parch','LastName']]
y=age['Age']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=4)
rfAge=RandomForestRegressor(n_estimators=10,n_jobs=-1 ,random_state=13,max_depth=6)
rfAge.fit(x_train,y_train)
y_pred=rfAge.predict(x_test)
y_pred=np.around(y_pred,0)

ageNull=titanic.loc[titanic['Age'].isnull(),:]
x=ageNull[['SibSp','Parch','LastName']]
y_pred=rfAge.predict(x)
y_pred=np.around(y_pred,0)  
titanic.loc[titanic['Age'].isnull(),'Age']=pd.Series(y_pred,dtype=int,index=ageNull.index)


################################ CABIN MODEL ####################################
cabinSurvivedRate=titanic[titanic.Survived==1].Cabin.value_counts(normalize=True) 
print(cabinSurvivedRate)
cabinSurvivedRate.update(pd.Series(cabinSurvivedRate.values*100,index=cabinSurvivedRate.index,dtype=int))
cabinSurvivedRate=cabinSurvivedRate.to_dict()


from sklearn.ensemble import RandomForestClassifier
cabin=titanic.loc[titanic['Cabin'].notna(),:]
x=cabin[['Ticket','Pclass','Title','LastName','Embarked']]
y=cabin['Cabin']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=4)
rfCabin=RandomForestClassifier(n_estimators=15,n_jobs=-1 ,random_state=13,max_depth=3)
rfCabin.fit(x_train,y_train)
y_pred=rfCabin.predict(x_test)
 

cabinNull=titanic.loc[titanic['Cabin'].isnull(),:]
x=cabinNull[['Ticket','Pclass','Title','LastName','Embarked']]
y_pred=rfCabin.predict(x)   
titanic.loc[titanic['Cabin'].isnull(),'Cabin']=pd.Series(y_pred,index=cabinNull.index)

 
titanic["Cabin"].replace(cabinSurvivedRate, inplace=True)
#titanic[titanic["Cabin"].str.isnumeric()==False]=10
titanic.loc[titanic["Cabin"].str.isnumeric()==False,'Cabin']=10

################################ SURVIVED MODEL #################################### 
survivalCols=['Age','Cabin','Parch','Sex']     
x=titanic[survivalCols]
y=titanic['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=4)
from sklearn.ensemble import RandomForestClassifier
rfSurv=RandomForestClassifier(n_estimators=15,n_jobs=-1 ,random_state=13,max_depth=3)#n_estimators=25,n_jobs=-1 ,random_state=14,max_depth=6)
rfSurv.fit(x_train,y_train)
y_pred=rfSurv.predict(x_test) 
 
     
################################ TEST THE MODEL ON TEST DATA ####################################
titanictest=pd.read_csv('../input/test.csv')

titanictest['Sex'].replace(['male','female'],[1,0],inplace=True)
titanictest['Embarked'].replace([np.nan],['C'],inplace=True)
titanictest['Embarked']=embarkedEncoder.transform(titanictest['Embarked'])
titanictest['Embarked']=titanictest['Embarked']+1
titanictest['Fare'].fillna(-1, inplace=True)
median=titanictest.Fare[(titanictest["Fare"] != -1) & (titanictest['Pclass'] == 3)].median()
titanictest['Fare']=np.where(titanictest['Fare']==-1,median,titanictest['Fare'])
titanictest['Fare']=fareScaler.transform(pd.DataFrame(titanictest['Fare'].values))  
titanictest['Title']=titanictest['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip()) 
titanictest['LastName']=titanictest['Name'].apply(lambda name: name.split(',')[0]) 
titanictest['LastName']=lastNameEncoder.fit_transform(titanictest['LastName'].values)
titanictest['Title']=titleEncoder.fit_transform(titanictest['Title']) 
titanictest['Ticket']=ticketEncoder.fit_transform(titanictest['Ticket'])
titanictest['Cabin']=titanictest['Cabin'].str[0]      
      
## Predict missing Age at Model  
age=titanictest.loc[titanictest['Age'].isnull(),:]
x=age[['SibSp','Parch','LastName']]
y_pred=rfAge.predict(x)   
titanictest.loc[titanictest['Age'].isnull(),'Age']=pd.Series(y_pred,dtype=int,index=age.index)

## Predict missing Cabin at Model  
cabin=titanictest.loc[titanictest['Cabin'].isnull(),:]
x=cabin[['Ticket','Pclass','Title','LastName','Embarked']]
y_pred=rfCabin.predict(x)  
titanictest.loc[titanictest['Cabin'].isnull(),'Cabin']=pd.Series(y_pred,index=cabin.index)
titanictest["Cabin"].replace(cabinSurvivedRate, inplace=True) 

#Run survived Model on Test 
x=titanictest.loc[:,survivalCols]
y_pred=rfSurv.predict(x)
titanictest['Prediction']=pd.Series(y_pred,index=titanictest.index)

subb=pd.read_csv('../input/gender_submission.csv')
result=pd.merge(titanictest,subb,on='PassengerId')
result['Result']= result['Survived']-result['Prediction'] 
result['Result']=result['Result'].abs()

## ACCURACY OF MODEL:
print(result['Result'].value_counts(normalize=True)) 