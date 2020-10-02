# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Importing libraries
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing dataset
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
pd.set_option('max_columns',60)

#Feature Engineering (Feature Selection and Extraction)

#Pclass
#There is a direct relationshi between a higher Pclass and more survival
#rate.We will consider this data as a categorical variable.There is a
#correlation of 0.34 between Pclass and survived which is a strong one
train['Survived'].groupby(train['Pclass']).mean()
train['Survived'].corr(train['Pclass'])

#Name
#The name doesn't seem to have any correlation with survival rate but the
#title and the length of the name may have some effect.There is a high probability
#for people with certain titles to survive.The name length has a 0.33 correlation with survival rate
#People with longer names have more chance of survival
train['Name Title']=train['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
train['Name Title'] = np.where((train['Name Title']).isin(['Lady.','Master.','Miss.','Mlle.','Mme.','Mrs.','Ms.','Sir.','the'])
                    , 'Powerful',
                                   np.where((train['Name Title']).isin(['Capt.','Col.','Don.','Dr.','Jonkheer.','Major.','Mr.','Rev.']),
                                            'Weak', 'Other'))
train['Name Length']=train['Name'].apply(lambda x:len(x))
test['Name Title']=test['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
test['Name Title'] = np.where((test['Name Title']).isin(['Lady.','Master.','Miss.','Mlle.','Mme.','Mrs.','Ms.','Sir.','the'])
                    , 'Powerful',
                                   np.where((test['Name Title']).isin(['Capt.','Col.','Don.','Dr.','Jonkheer.','Major.','Mr.','Rev.']),
                                            'Weak', 'Other'))
test['Name Length']=test['Name'].apply(lambda x:len(x))
train['Survived'].groupby(train['Name Title']).mean()
train['Survived'].groupby(pd.qcut(train['Name Length'],5)).mean()
train['Survived'].corr(train['Name Length'])
train=train.drop('Name',axis=1)
test=test.drop('Name',axis=1)

#Sex
#Don't we say women first??It seems here they got lucky
#Women have a very high chance of survival
train['Survived'].groupby(train['Sex']).mean()

#Age
#Will you let a child die in front of you?Lets see what the people on the titanic thought about it
#I can't see a correlation here.Lets see what the correlation function thinks.There seems to be a
#weak negative correlation.It seems people didn't care about children that much :(
train['Survived'].groupby(pd.qcut(train['Age'],10)).mean()
train['Survived'].corr(train['Age'])

#SibSp(No of siblings/spouse)
#I am here seeing some sort of a -ve correlation.Ofcourse you can't all family survive.You will take up the
#entire boat.And you don't want to survive alne too.There seems to be a weak -ve correlation
train['Survived'].groupby(train['SibSp']).mean()
train['Survived'].corr(train['SibSp'])

#No of parent/children
#I can't really figure out the correlation.There seems to be a weak +ve correlation
train['Survived'].groupby(train['Parch']).mean()
train['Survived'].corr(train['Parch'])

#Family Members
#Lets see if we can combine SibSp and Parch and get a strong correlation.
#There seems to be an initial +ve but later -ve correlation between no of
#family members and survival rate.We still can't really find a strong correlation
#using the corr method
train['Family Members']=train['SibSp']+train['Parch']
train=train.drop('SibSp',axis=1)
train=train.drop('Parch',axis=1)
test['Family Members']=test['SibSp']+test['Parch']
test=test.drop('SibSp',axis=1)
test=test.drop('Parch',axis=1)
train['Survived'].groupby(train['Family Members']).mean()
train['Survived'].corr(train['Family Members'])

#Ticket
#The ticket doesn't seem to say a lot but lets see if we can find something
#hidden there.Lets see if the 1st letter or the length of the ticket name 
#in any ways affects survival.The ticket name may be related to the passenger
#class.I can't really find a correlation between the ticket 1st letter and the
#survival rate.There doesn't seem to be a stong correlation between Ticket Length
#and surviva; rate either

train['Ticket Lett']=train['Ticket'].apply(lambda x:x[0])
train['Ticket Lett']=np.where((train['Ticket Lett']).isin(['1','F','P']), ['Valuable'],['Valueless'])
test['Ticket Lett']=test['Ticket'].apply(lambda x:x[0])
test['Ticket Lett']=np.where((test['Ticket Lett']).isin(['1','F','P']), 'Valuable',
                                   np.where((test['Ticket Lett']).isin(['2','3','4','5','6','7','8','9','A','C','L','S','W']),
                                            'Valueless', 'Other'))
train['Survived'].groupby(train['Ticket Lett']).mean()
train['Ticket Length']=train['Ticket'].apply(lambda x:len(x))
test['Ticket Length']=test['Ticket'].apply(lambda x:len(x))
train['Survived'].groupby(pd.qcut(train['Ticket Length'],2)).mean()
train['Survived'].corr(train['Ticket Length'])
train=train.drop('Ticket',axis=1)
test=test.drop('Ticket',axis=1)
#Fare
#The fare might have a relation with the passenger class.I cna't really
# see a correlation(Slightly biased towards +ve correlation)
#The corr function declares a strong +ve correlation.People giving higher
#fare have a higher chance of survival
train['Survived'].groupby(pd.qcut(train['Fare'],10)).mean()
train['Survived'].corr(train['Fare'])

#Cabin
#The cabin name doesn't seem to have any correlation with survival rate in
#1st look.Lets see if we can find something in the initial letter or the 
#name length.The Cabin Lett might have some correlation.There seems to be a
#weak +ve correlation between cabin length and survival rate
train['Cabin Lett']=train['Cabin'].apply(lambda x:str(x)).apply(lambda x:x[0])
train['Cabin Lett']=np.where((train['Cabin Lett']).isin(['B','C','D','E','F']), 'High Class',
                                   np.where((train['Cabin Lett']).isin(['A','G','T','n']),
                                            'Low Class', 'Other Cabin'))
test['Cabin Lett']=test['Cabin'].apply(lambda x:str(x)).apply(lambda x:x[0])
test['Cabin Lett']=np.where((test['Cabin Lett']).isin(['B','C','D','E','F']), 'High Class',
                                   np.where((test['Cabin Lett']).isin(['A','G','T','n']),
                                            'Low Class', 'Other Cabin'))
train['Survived'].groupby(train['Cabin Lett']).mean()
train['Cabin Length']=train['Cabin'].apply(lambda x:str(x)).apply(lambda x:len(x))
test['Cabin Length']=test['Cabin'].apply(lambda x:str(x)).apply(lambda x:len(x))
train['Survived'].groupby(pd.qcut(train['Cabin Length'],2)).mean()
train['Survived'].corr(train['Cabin Length'])
train=train.drop('Cabin',axis=1)
test=test.drop('Cabin',axis=1)
#Embarked
#People from Cherbourg(C) seem to have a higher survival rate
train['Survived'].groupby(train['Embarked']).mean()

#Features to be considered due to a high correlation.Lets consider the threshold to be 0.05
#so we wouldn't consider family members
Corrs=[]
for i in ['Pclass','Age','Fare','Name Length','Family Members','Ticket Length','Cabin Length']:
    Corrs.append(train['Survived'].corr(train[i]))
train=train.drop('Family Members',axis=1)
test=test.drop('Family Members',axis=1)

#Data Cleaning

#Checking for no of missing values
train.isnull().sum()
test.isnull().sum()
#Handling age.I am going to replace NaN values in age with the mean of the age values having the same Title
#Handling embarked.Replaced NaN values in Embarked with  the most common value 'S'
#Handling Fare.Replaced missing far value with the mean of fate values of people travelling in same Pclass
data_age=train['Age'].groupby(train['Name Title'])
train['Age']=data_age.transform(lambda x:x.fillna(x.mean()))
train['Embarked'].fillna('S',inplace=True)
test['Age']=data_age.transform(lambda x:x.fillna(x.mean()))
data_fare=train['Fare'].groupby(train['Pclass'])
test['Fare']=data_fare.transform(lambda x:x.fillna(x.mean()))
#Encoding
for i in ['Pclass','Sex','Embarked','Name Title','Ticket Lett','Cabin Lett']:
        train=pd.concat((train,pd.get_dummies(train[i],prefix=i)),axis=1)
        train=train.drop(i,axis=1)
        test=pd.concat((test,pd.get_dummies(test[i],prefix=i)),axis=1)
        test=test.drop(i,axis=1)
train.insert(loc=15,column='Name Title_Other',value=0)

for i in train.columns[2:]:
    if i not in test.columns:
        train=train.drop(i,axis=1)
        
"""
#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,min_samples_split=12,
            min_samples_leaf=1,max_features='auto',oob_score=True)
classifier.fit(train.iloc[:,2:].values,train.iloc[:,1].values)

#SVM
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(train.iloc[:,2:].values,train.iloc[:,1].values)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(train.iloc[:,2:].values,train.iloc[:,1].values)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
classifier.fit(train.iloc[:,2:].values,train.iloc[:,1].values)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(train.iloc[:,2:].values,train.iloc[:,1].values)
"""

#XGBoost
from xgboost import XGBClassifier
classifier=XGBClassifier(max_depth=5, n_estimators=10, learning_rate=0.1)
classifier.fit(train.iloc[:,2:].values,train.iloc[:,1].values)

#Predicting train results
y_pred_train=classifier.predict(train.iloc[:,2:].values)

#Model Selection

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(train.iloc[:,1].values,classifier.predict(train.iloc[:,2:].values))
#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=train.iloc[:,2:].values,y=train.iloc[:,1].values,cv=10)
accuracies.mean()
accuracies.std()
#Grid Search
from sklearn.model_selection import GridSearchCV
parameters=[{'n_estimators':[10,100,1000],
            'max_depth':[1,5,10],'learning_rate':[0.1,0.01,0.001]}]
grid_search=GridSearchCV(estimator=XGBClassifier(),param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(train.iloc[:,2:].values,train.iloc[:,1].values)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

#Predicting test results
y_pred_test=classifier.predict(test.iloc[:,1:].values)
prediction=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred_test})
prediction.to_csv('Predictions',index=False)

