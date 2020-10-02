#!/usr/bin/env python
# coding: utf-8

# In this tutorial I will show how to make preprocessing and machine learning of titanic data set to achieve as I did 85.03 5 accuracy by using machine learning and preprocessing.
# for me I saw alot of kaggler submit beatiful EDA better than me so I won't concentrate on EDA because I wan't to make the eyes on preprocessing and machine learning. 
# 
# 
# 
# 

# **Understanding our files before sinking**
# so we have 3 files here :
# 1 for train data , 1 for test data and another one as mirror image of our submission file that we must submit at the end. I saw alot of kernel join train with test but it is fatal mistake as there is no survival in the test thats will ulter our results and gives fake results . so to get a test set we must split our *train* file into *training* and *testing* and the test file given to us **is not the test data of our model**. 

# 

# **Note**: for computational purposes . use this code to calculate how many seconds spent to execute code

# In[ ]:


import time
start=time.time()
'''execution function'''
end=time.time()
print(end-start)


# Importing our necessary files and libraries

# In[ ]:


import numpy as np
import pandas as pd

allData =pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# **Features  processing **

# preprocessing must be done for the train (allData) and test for each feature

# Name :

# In[ ]:


#for train
allData.Name.head() 
allData['nickName']=allData.Name.str.extract('([A-Za-z]+)\.')
allData.nickName.value_counts()
allData.nickName.replace(to_replace=['Dr', 'Rev', 'Col', 'Major', 'Capt'],value='captain',inplace=True)
allData.nickName.replace(to_replace=['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'],value='noble') 
allData.nickName.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
allData.Name.head() 


# In[ ]:


#for test
test.Name.head()                           
test['nickName']=test.Name.str.extract('([A-Za-z]+)\.')
test.nickName.value_counts()
test.nickName.replace(to_replace=['Dr', 'Rev', 'Col', 'Major', 'Capt'],value='captain',inplace=True)
test.nickName.replace(to_replace=['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'],value='noble') 
test.nickName.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
test.Name.head()                           


# SibSp - Parch  :

# In[ ]:


#for train
allData['Family_size'] = allData.SibSp + allData.Parch + 1  
allData.Family_size.value_counts()
allData.Family_size.replace(to_replace = [1], value = 'single', inplace = True)
allData.Family_size.replace(to_replace = [2,3], value = 'small', inplace = True)
allData.Family_size.replace(to_replace = [4,5], value = 'medium', inplace = True)
allData.Family_size.replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)


# In[ ]:


#for test
test['Family_size'] = test.SibSp + test.Parch + 1  
test.Family_size.value_counts()
test.Family_size.replace(to_replace = [1], value = 'single', inplace = True)
test.Family_size.replace(to_replace = [2,3], value = 'small', inplace = True)
test.Family_size.replace(to_replace = [4,5], value = 'medium', inplace = True)
test.Family_size.replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)


# Imputing missing values for train and test data
# 

# In[ ]:


#train
allData.Embarked.fillna(value='S',inplace=True)
allData.Fare.fillna(value=allData.Fare.mean(),inplace=True)
allData.Age = allData.groupby(['nickName', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
allData.isnull().sum()


# In[ ]:


#test
test.Embarked.fillna(value='S',inplace=True)
test.Fare.fillna(value=test.Fare.mean(),inplace=True)
test.Age = test.groupby(['nickName', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
test.isnull().sum()


# **Data Transformation**

#    1- binning ages

# In[ ]:



age_categories=['infants','child','teen','young_adult','adult','geriatric']
age_numbers=[0,5,12,18,35,60,81]
allData['new_age']=pd.cut(allData.Age,age_numbers,labels=age_categories)
display(allData[['new_age','Age']].head())

test['new_age']=pd.cut(test.Age,age_numbers,labels=age_categories)


# 2-binning Fare
# 

# In[ ]:



fare_categories=['low','moderate','high','extreme']
fare_numbers=[-1,150,250,350,550]
allData['new_fare']=pd.cut(allData.Fare,fare_numbers,labels=fare_categories)

test['new_fare']=pd.cut(test.Fare,fare_numbers,labels=fare_categories)


# **Dropping Features**
# 
# 1- remover age :during survival , they ordered to safe first women and children
# whatever their ages.
# 2- remove name : death won't choose mr and leave sir ,will  it?
# 3- Fare : there are alot of things more important than that

# 1- drop features for train data

# In[ ]:


allData.drop(columns=['Age',
                       'Fare',
                       'PassengerId',
                       'Cabin',
                       'Name',
                       'SibSp', 
                       'Parch',
                       'Ticket'],inplace=True,axis=1)

allData.columns


# 2-drop features for test data
# **Note :** 
# before droping featues , we  extract PassengerId feature because we need it for submission

# In[ ]:


ids=test['PassengerId']
test.drop(columns=['Age',
                       'Fare',
                       'Cabin',
                       'PassengerId',
                       'Name',
                       'SibSp', 
                       'Parch',
                       'Ticket'],inplace=True,axis=1)

test.columns


# **Correcting data types of train data**

# 1- for train

# In[ ]:


allData.dtypes
allData.loc[:,['Sex','Embarked','Family_size','nickName']]=allData.loc[:,['Sex','Embarked','Family_size','nickName']].astype('category')
allData.dtypes


# 2-for test

# In[ ]:



test.dtypes
test.loc[:,['Sex','Embarked','Family_size','nickName']]=test.loc[:,['Sex','Embarked','Family_size','nickName']].astype('category')

test.dtypes


# **Encoding categorical variables** 
# 
# concatination between train and test data must be performed ; since on encoding of train and test data separately , there will be specified number of features in the train data .some features may be lost and upon  making . predict it will show you an error(if features are not equal in train and test data). I will show it at the end.

# In[ ]:


concat_data=pd.concat([allData,test],sort=False)#to prevent features loss
concat_data=pd.get_dummies(concat_data)
training=concat_data.iloc[:891,:]
x = training.drop("Survived", axis=1)
y = training["Survived"]

testing=concat_data.iloc[891:,:]
x_testing=testing.drop('Survived',axis=1)


# **splitting our train data into train and test split**

# In[ ]:


from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)


# **Modelling
# **

# **1- Accuracy of the pure model**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


logReg=LogisticRegression(solver='liblinear')
logReg.fit(x_train, y_train) 
y_pred=logReg.predict(x_test)                  
logReg_acc=metrics.accuracy_score(y_test,y_pred)
logReg_accuracy = np.round(logReg_acc*100, 2)
print(logReg_accuracy)


# In[ ]:


from sklearn import svm
Svm=svm.SVC(kernel='rbf',C=1,gamma=.1)
Svm.fit(x_train,y_train)
y_pred=Svm.predict(x_test)
svm_acc=metrics.accuracy_score(y_test,y_pred)
svm_accuracy=np.round(svm_acc*100,2)
print(svm_accuracy)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
rfc_acc=metrics.accuracy_score(y_test,y_pred)
rfc_accuracy=np.round(rfc_acc*100,2)
print(rfc_accuracy)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn_acc=metrics.accuracy_score(y_test,y_pred)
knn_accuracy=np.round(knn_acc*100,2)
print(knn_accuracy)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
gnb_acc=metrics.accuracy_score(y_test,y_pred)
gnb_accuracy=np.round(gnb_acc*100,2)
print(gnb_accuracy)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
dtc_acc=metrics.accuracy_score(y_test,y_pred)
dtc_accuracy=np.round(dtc_acc*100,2)
print(dtc_accuracy)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_pred=gbc.predict(x_test)
gbc_acc=metrics.accuracy_score(y_test,y_pred)
gbc_accuracy=np.round(gbc_acc*100,2)
print(gbc_accuracy)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc.fit(x_train,y_train)
y_pred=abc.predict(x_test)
abc_acc=metrics.accuracy_score(y_test,y_pred)
abc_accuracy=np.round(abc_acc*100,2)
print(abc_accuracy)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(x_train,y_train)
y_pred=etc.predict(x_test)
etc_acc=metrics.accuracy_score(y_test,y_pred)
etc_accuracy=np.round(etc_acc*100,2)
print(etc_accuracy)


# **2- Accuracy after cross validation**

# In[ ]:


from sklearn.model_selection import cross_val_score
def cross_val_accuracy(m):
    crossvalscore = cross_val_score(m,x_train,y_train,cv=10,scoring='accuracy').mean()
    model_accuracy=np.round(crossvalscore*100,2)
    return model_accuracy


#try it
'''cross_val_models=pd.DataFrame({'Cross_val_accuracy':[
        cross_val_accuracy(logReg),
        cross_val_accuracy(Svm),
        cross_val_accuracy(rfc),
        cross_val_accuracy(knn),
        cross_val_accuracy(gnb),
        cross_val_accuracy(dtc),
        cross_val_accuracy(gbc),
        cross_val_accuracy(abc),
        cross_val_accuracy(etc) 
        ]})
    
    


cross_val_models.index=[
           'LogisticRegression',
                      'Support vector machine',
           'RandomForestClassifier',
           'KNeighborsClassifier',
           'GaussianNB',
           'DecisionTreeClassifier',
           'GradientBoostingClassifier',
           'AdaBoostClassifier',
           'ExtraTreesClassifier',
        ]

cross_val_accuracy = cross_val_models.sort_values(by = 'Cross_val_accuracy', ascending = False)
'''


# **Hyper Parameter tuning**

# In[ ]:


#hyper parameters
logReg_params={'penalty':['l1','l2'],
               'C':np.logspace(0,4,10)}
svm_params={'C':[6,7,8,9,10,11,12],
            'kernel':['linear','rbf'],
            'gamma':[0.5,0.2,0.1,0.001,0.0001]
            }

rfc_params={'criterion':['gini','entropy'],
            'n_estimators':[10,15,20,25,30],
            'min_samples_leaf':[1,2,3],
            'min_samples_split':[3,4,5,6,7],
            'max_features':['sqrt','auto','log2']
           }
knn_params={'n_neighbors':[3,4,5,6,7,8],
            'leaf_size':[1,2,3,5],
            'weights':['uniform','distance'],
            'algorithm':['auto','ball_tree','kd_tree','brute'],
            
            
            }
dtc_params={'max_features':['auto','sqrt','log2'],

            'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'min_samples_leaf':[1,2,3]
            }
gbc_params={'learning_rate':[.01,.02,.05,.01],
            'min_samples_split':[2,3,4],
            'max_depth':[4,6,8],
            'max_features':[1.0,0.3,0.1]
            }
abc_params={'n_estimators':[1, 5, 10, 15, 20, 25, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300],
            'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]            }

etc_params={
             'n_estimators':[100,300],
            'max_depth':[None],
             'max_features':[1,3,10],
             'min_samples_split':[2,3,10],
            'min_samples_leaf':[1,3,10],
            'bootstrap':[False],
            'criterion':['gini']
            }


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


def hyperParameterTuning(m,params):
    grid=GridSearchCV(m,params,scoring='accuracy',cv=10,n_jobs=-1)
    grid.fit(x_train,y_train)
    best_score=np.round(grid.best_score_*100,2)
    return best_score


# **detecting best parameters**

# In[ ]:



def bestParameters(m,parameters):
    grid=GridSearchCV(m,parameters,scoring='accuracy',cv=10,n_jobs=-1)
    grid.fit(x_train,y_train)
    best_parameters=grid.best_params_
    return best_parameters


bestParameters(logReg,logReg_params)
bestParameters(rfc,rfc_params)
bestParameters(knn,knn_params)
bestParameters(dtc,dtc_params)
bestParameters(gbc,gbc_params)
bestParameters(abc,abc_params)
bestParameters(Svm,svm_params)


# **Organize our accuracies in tables for comparison**

# it is time consuming but you must do it by your own to compare all of your results

# In[ ]:


#now lets comapre all of our accuracies
'''pureAccuracy=pd.DataFrame({'pureModel':[
                                   etc_accuracy,
                                   abc_accuracy,
                                   gbc_accuracy,
                                   dtc_accuracy,
                                   knn_accuracy,
                                   rfc_accuracy,
                                   svm_accuracy,
                                   logReg_accuracy,
                                   gnb_accuracy
                             ]})
    
pureAccuracy.index=[  
                                   'ExtraTreesClassifier',  
                                   'AdaBoostClassifier', 
                                   'GradientBoostingClassifier',
                                   'DecisionTreeClassifier',
                                   'KNeighborsClassifier',
                                   'RandomForestClassifier',
                                   'Support vector machine',
                                   'LogisticRegression',
                                   'GaussianNB'
]
pure_accuracy = pureAccuracy.sort_values(by = 'pureModel', ascending = False)
 
CV_accuracy=pd.DataFrame({ 'crossValidation':[ 
                                   cross_val_accuracy(etc),
                                   cross_val_accuracy(abc),
                                   cross_val_accuracy(gbc),
                                   cross_val_accuracy(dtc),
                                   cross_val_accuracy(knn),
                                   cross_val_accuracy(rfc),
                                   cross_val_accuracy(Svm),
                                   cross_val_accuracy(logReg),
                                   cross_val_accuracy(gnb)
                                   ]})
CV_accuracy.index=[
                                   'ExtraTreesClassifier',  
                                   'AdaBoostClassifier', 
                                   'GradientBoostingClassifier',
                                   'DecisionTreeClassifier',
                                   'KNeighborsClassifier',
                                   'RandomForestClassifier',
                                   'Support vector machine',
                                   'LogisticRegression',
                                   'GaussianNB'] 
crossValidation_accuracy = CV_accuracy.sort_values(by = 'crossValidation', ascending = False)

HPT_accuracy=pd.DataFrame({ 'hyperparameterTuning':[
                                    hyperParameterTuning(etc,etc_params),
                                    hyperParameterTuning(abc,abc_params),
                                    hyperParameterTuning(gbc,gbc_params),
                                    hyperParameterTuning(dtc,dtc_params),
                                    hyperParameterTuning(knn,knn_params),
                                    hyperParameterTuning(rfc,rfc_params),
                                    hyperParameterTuning(Svm,svm_params),
                                    hyperParameterTuning(logReg,logReg_params),
                                    0]})#0 for gnb
                            
       
HPT_accuracy.index=[  
                                   'ExtraTreesClassifier',  
                                   'AdaBoostClassifier', 
                                   'GradientBoostingClassifier',
                                   'DecisionTreeClassifier',
                                   'KNeighborsClassifier',
                                   'RandomForestClassifier',
                                   'Support vector machine',
                                   'LogisticRegression',
                                   'GaussianNB'
]


tuning_accuracy = HPT_accuracy.sort_values(by = 'hyperparameterTuning', ascending = False)
'''


# Comparing the highiest accuracies from the threee techniques :
# 
# etc>>85.03 ,gbc ,rfc
# 
# 83.98>>>gradient boosting after cross validation
# 
# 81.17>>> logistic regression pure accuracy

# **Ensembling**

# In[ ]:


#Ensembling voting classifier
from sklearn.ensemble import VotingClassifier
dtc_voted=DecisionTreeClassifier(max_features='auto',min_samples_leaf=2,min_samples_split=9)
 
logReg_voted=LogisticRegression(penalty='l1',C=2.7825594022071245)
 
gnb_voted=GaussianNB()#no argumenets
 
svm_voted=svm.SVC(C=7,gamma=0.2,kernel='rbf',probability=True)
 
knn_voted=KNeighborsClassifier(algorithm= 'brute', leaf_size=1,n_neighbors= 8, weights= 'uniform')
votingEnsembling=VotingClassifier(estimators=[('logReg',logReg_voted),
                                                 ('gnb',gnb_voted),
                                                 ('svm',svm_voted),
                                                 ('knn',knn_voted),
                                                 ('dtc',dtc_voted)],voting='soft')
votingEnsembling.fit(x_train,y_train)
votingEnsembling.score(x_test,y_test)
crossvalscore=cross_val_score(votingEnsembling,x_train,y_train, cv = 10,scoring = "accuracy").mean()
accuracypercent=np.round(crossvalscore*100,2)


# **Submission**

# In[ ]:


#you can submit etc ,gbc ,rfc
grid_rfc=GridSearchCV(rfc,rfc_params,scoring='accuracy',cv=10,n_jobs=-1)
grid_rfc.fit(x_train,y_train)
best_score=np.round(grid_rfc.best_score_*100,2)
prediction=grid_rfc.predict(x_testing)
print(best_score)


# In[ ]:


submission=pd.DataFrame({
                        'PassengerId' :ids ,#see up codes
                        'Survived' : prediction})
submission.Survived=submission.Survived.astype(int)
submission.to_csv('prediction.csv',index=False)


# Thats all for now , I  welcome upvoting and commenting , thanks
