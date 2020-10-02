#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data overview
import numpy as np
import pandas as pd

train=pd.read_csv("../input/train.csv")

#data cleaning
train=train.drop("Ticket",axis=1)
train=train.drop("Cabin",axis=1)
train.loc[train["Sex"]=="female",'Sex']=1 #Sex:1==female
train.loc[train["Sex"]=="male",'Sex']=0 #Sex:0==male
train["Age"]=train["Age"].fillna(train["Age"].median())
train["Embarked"]=train["Embarked"].fillna("S")
train.loc[train["Embarked"]=="S",'Embarked']=0 #Embarked: 0==S 
train.loc[train["Embarked"]=="C",'Embarked']=1 #Embarked: 1==C
train.loc[train["Embarked"]=="Q",'Embarked']=2 #Embarked: 2==Q


#combining data and making new features
train["FamNum"]=train["SibSp"]+train["Parch"]
train=train.drop("SibSp",axis=1)
train=train.drop("Parch",axis=1)


#name? socaial status
train["NLength"]=train["Name"].apply(len)
train["NLengthItv"]=pd.cut(train["NLength"], 3)
print(train[["NLengthItv","Survived"]].groupby(["NLengthItv"]).mean())
train.loc[train["NLength"]<35,"NLength"]=0
train.loc[(train["NLength"]>=35) & (train["NLength"]<59), "NLength"]=1
train.loc[train["NLength"]>=59, "NLength"]=2
train=train.drop("NLengthItv", axis=1)

import re
#titles=Dr(0),Master(1),Miss(2),Mr(3),Mrs(4),Rev(5),Else(6)
train["Title"]=train["Name"].str.extract("([\w]+)\.")
train["Title"] = train["Title"].replace('Mlle', 'Miss')
train["Title"] = train["Title"].replace('Ms', 'Miss')
train["Title"] = train["Title"].replace('Mme', 'Mrs')
train["Title"]=train["Title"].replace(['Capt','Rev','Dr','Master','Col','Countess','Don','Jonkheer','Lady','Major','Sir'],'Else')
train.loc[train["Title"]=="Miss",'Title']=0
train.loc[train["Title"]=="Mr",'Title']=1 
train.loc[train["Title"]=="Mrs",'Title']=2 
train.loc[train["Title"]=="Else",'Title']=3 
# pd.crosstab(train["Title"], train["Sex"])
train=train.drop("Name", axis=1)

#separate fare into intervals
train["FareItv"]=pd.qcut(train["Fare"],5)#base on frequency
print(train[["FareItv","Survived"]].groupby(["FareItv"]).mean())
train.loc[train["Fare"]<10.5,"Fare"]=0
train.loc[(train["Fare"]>=10.5) & (train["Fare"]<=39.7),"Fare"]=1
train.loc[train["Fare"]>39.7,"Fare"]=2
train["Fare"]=train["Fare"].astype(int)
train=train.drop("FareItv",axis=1)


#separate age into intervals
train["AgeItv"]=pd.cut(train["Age"],5)
print(train[["AgeItv","Survived"]].groupby(["AgeItv"]).mean())
train.loc[train["Age"]<8,"Age"]=0
train.loc[(train["Age"]>=8) & (train["Age"]<16.5),"Age"]=1
train.loc[(train["Age"]>=16.5) & (train["Age"]<=64),"Age"]=2
train.loc[train["Age"]>64,"Age"]=3
train["Age"]=train["Age"].astype(int)
train=train.drop("AgeItv",axis=1)


#seperate family number into intervals
train["FamItv"]=pd.cut(train["FamNum"],10)#base on frequency
print(train[["FamItv","Survived"]].groupby(["FamItv"]).mean())
train.loc[train["FamNum"]<1,"FamNum"]=0 #alone
train.loc[(train["FamNum"]>=1) & (train["FamNum"]<3),"FamNum"]=1
train.loc[(train["FamNum"]>=3) & (train["FamNum"]<6),"FamNum"]=2
train.loc[train["FamNum"]>=6,"FamNum"]=3
train=train.drop("FamItv",axis=1)
train.head(10)


# In[ ]:


#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#no NAN value or categorical values for pairplot
sns.pairplot(train,vars=['Sex','Pclass','Age','NLength','FamNum','Embarked','Title'], hue="Survived",palette="pastel")


# In[ ]:


predictors = ["Pclass", "Sex", "Fare","Age","NLength", "FamNum","Embarked","Title"]

#feature selection
from sklearn.feature_selection import SelectKBest,f_classif
selector = SelectKBest(f_classif,k=7)
selector.fit(train[predictors],train["Survived"])

sns.barplot(x=predictors,y=-np.log(selector.pvalues_))


# In[ ]:



#models to use for classification of survival
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test=train_test_split(train[predictors],train["Survived"],test_size=0.3)

LRclf=LogisticRegression()
LRclf.fit(X_train,y_train)
LRpred=LRclf.predict(X_test)
LRscore=accuracy_score(LRpred,y_test)
print("Linear Rregression Accuracy: "+str(LRscore))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

KNNclf=KNeighborsClassifier()
KNNclf.fit(X_train,y_train)
KNNpred=KNNclf.predict(X_test)
KNNscore=accuracy_score(KNNpred,y_test)
print("KNN Accuracy: "+str(KNNscore))


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


SVMclf=SVC()
SVMclf.fit(X_train,y_train)
SVMpred=SVMclf.predict(X_test)
SVMscore=accuracy_score(SVMpred,y_test)
print("SVM Accuracy: "+str(SVMscore))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


treeclf=DecisionTreeClassifier()
treeclf.fit(X_train,y_train)
treepred=treeclf.predict(X_test)
treescore=accuracy_score(treepred,y_test)
print("Decision Tree Accuracy: "+str(treescore))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RFclf=RandomForestClassifier()
RFclf.fit(X_train,y_train)
RFpred=RFclf.predict(X_test)
RFscore=accuracy_score(RFpred,y_test)
print("Random Forest Accuracy: "+str(RFscore))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

GBclf=GradientBoostingClassifier()
GBclf.fit(X_train,y_train)
GBpred=GBclf.predict(X_test)
GBscore=accuracy_score(GBpred,y_test)
print("Gradient Boost Accuracy: "+str(GBscore))


# In[ ]:


#visualize all models above
models=pd.DataFrame([["Linear Regression",LRscore],
                     ["K Neighbors",KNNscore],
                     ["SVM",SVMscore],
                     ["Decision Tree",treescore],
                     ["Random Forest",RFscore],
                     ["Gradient Boost", GBscore]
                    ],columns=["model","accuracy"])
models


# In[ ]:


#vote for result
combinepred=[]
for i in range(0,len(KNNpred)):
    results=[]
    onec=0
    zeroc=0
    results.append(LRpred[i])
    results.append(KNNpred[i])
    results.append(SVMpred[i])
    results.append(treepred[i])
    results.append(RFpred[i])
    results.append(GBpred[i])
    for j in results:
        if j==1:
            onec+=1
        else:
            zeroc+=1
    if onec>zeroc:
        result=1
    else:
        result=0
    combinepred.append(result)
combine_score=accuracy_score(GBpred,y_test)
print("combined accuracy score: "+str(combine_score))


# In[ ]:


#process test data 
test=pd.read_csv("../input/test.csv")
test.head()
print(test.isnull().sum())

#data cleaning
test=test.drop("Ticket",axis=1)
test=test.drop("Cabin",axis=1)
test.loc[test["Sex"]=="female",'Sex']=1 #Sex:1==female
test.loc[test["Sex"]=="male",'Sex']=0 #Sex:0==male
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test.loc[test["Embarked"]=="S",'Embarked']=0 #Embarked: 0==S 
test.loc[test["Embarked"]=="C",'Embarked']=1 #Embarked: 1==C
test.loc[test["Embarked"]=="Q",'Embarked']=2 #Embarked: 2==Q

#combining data and making new features
test["FamNum"]=test["SibSp"]+test["Parch"]
test=test.drop("SibSp",axis=1)
test=test.drop("Parch",axis=1)

#name? socaial status
test["NLength"]=test["Name"].apply(len)
test["NLengthItv"]=pd.cut(test["NLength"], 3)
test.loc[test["NLength"]<35,"NLength"]=0
test.loc[(test["NLength"]>=35) & (test["NLength"]<59), "NLength"]=1
test.loc[test["NLength"]>=59, "NLength"]=2
test=test.drop("NLengthItv", axis=1)

import re
#titles=Dr(0),Master(1),Miss(2),Mr(3),Mrs(4),Rev(5),Else(6)
test["Title"]=test["Name"].str.extract("([\w]+)\.")
test["Title"] = test["Title"].replace('Mlle', 'Miss')
test["Title"] = test["Title"].replace('Ms', 'Miss')
test["Title"] = test["Title"].replace('Mme', 'Mrs')
test["Title"]=test["Title"].replace(['Capt','Master','Dr','Rev','Dona','Col','Countess','Don','Jonkheer','Lady','Major','Sir'],'Else')
test.loc[test["Title"]=="Miss",'Title']=0
test.loc[test["Title"]=="Mr",'Title']=1 
test.loc[test["Title"]=="Mrs",'Title']=2 
test.loc[test["Title"]=="Else",'Title']=3 
# print(pd.crosstab(test["Title"], test["Sex"]))

#separate fare into intervals
test["FareItv"]=pd.qcut(test["Fare"],5)#base on frequency
test.loc[test["Fare"]<10.5,"Fare"]=0
test.loc[(test["Fare"]>=10.5) & (test["Fare"]<=39.7),"Fare"]=1
test.loc[train["Fare"]>39.7,"Fare"]=2
test["Fare"]=test["Fare"].astype(int)
test=test.drop("FareItv",axis=1)


#separate age into intervals
test["AgeItv"]=pd.cut(test["Age"],5)
test.loc[test["Age"]<8,"Age"]=0
test.loc[(test["Age"]>=8) & (test["Age"]<16.5),"Age"]=1
test.loc[(test["Age"]>=16.5) & (test["Age"]<=64),"Age"]=2
test.loc[test["Age"]>64,"Age"]=3
test["Age"]=test["Age"].astype(int)
test=test.drop("AgeItv",axis=1)


#seperate family number into intervals
test["FamItv"]=pd.cut(test["FamNum"],10)#base on frequency
test.loc[test["FamNum"]<1,"FamNum"]=0 #alone
test.loc[(test["FamNum"]>=1) & (test["FamNum"]<3),"FamNum"]=1
test.loc[(test["FamNum"]>=3) & (test["FamNum"]<6),"FamNum"]=2
test.loc[test["FamNum"]>=6,"FamNum"]=3
test=test.drop("FamItv",axis=1)
test.head(10)


# In[ ]:


#final model
clfs=[LRclf,KNNclf,SVMclf,LRclf,treeclf,GBclf]
features=test[predictors]

def finalmodel(clfs,features):
    preds=[]
    
    for clf in clfs:
        clf.fit(train[predictors], train['Survived'])
        pred=clf.predict(features)
        preds.append(pred)
    
    predictions=[]
    for i in range(0,len(preds[0])):
        results=[]
        onec=0
        zeroc=0
        results.append(preds[0][i])
        results.append(preds[1][i])
        results.append(preds[2][i])
        results.append(preds[3][i])
        results.append(preds[4][i])
        results.append(preds[5][i])
        
        for j in results:
            if j==1:
                onec+=1
            else:
                zeroc+=1
                
        if onec>zeroc:
            result=1
        else:
            result=0
            
        predictions.append(result)
    
    return predictions
        


# In[ ]:


#submission
predictions=finalmodel(clfs,features)
submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":predictions})
submission.to_csv("submission.csv",index=False)
submission.head(10)


# In[ ]:




