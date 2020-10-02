#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Set
# 
#                                              
# ### Introduction
#                                               
# Hi friends , this is my first project in Kaggle and very special to me. Here I am presenting my project in my own way.
# Before starting the project i was having many doubts and i tried my best to get answers
# 1. Relationship between the survived and the pasengers ?
# 2. There were very few life jackets or rescue boats and On what basis passengers were rescued ?
# 3. Can budilding a model help in preditions ?
# 4. Can we consider the human emotional factors and captain priorities while preparing the model?
# 5. Is there any use of creating model?
# 
# 
# ### Project Steps followed
# 
# 1. Understanding the project data
# 2. EDA, Data Cleaning, Analyzing, identifying patterns.
# 3. Splitting the train data into train_model data and validation data and Building Model, predicting the results on validation data.
# 4. Feature engineering and creating a best suitable model (most important stage of project)
# 5. Prediting the results on test data and submiting the results

# ### Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# ### Part 1 : Understanding the project data 

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# ### Data contains 891 observations and 12 parameters (features)
# Features of the data 
# 
# PassengerId: unique id number to each passenger
# 
# Survived: passenger survive(1), didn't survive(0)
# 
# Pclass: passenger class
# 
# Name: passenger's name
# 
# Sex: passenger's gender
# 
# Age: passenger's age
# 
# SibSp: number of siblings/spouses
# 
# Parch: number of parents/children
# 
# Ticket: ticket number
# 
# Fare: amount of money spent on ticket
# 
# Cabin: cabin category
# 
# Embarked: part where passenger embarked(C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


train.dtypes


# ### Type Of Features
# 
# Categorical Features: Sex,Embarked,Survived,Parch
# 
# Ordinal Features: PClass
# 
# Continous Feature: Age
# 
# Other features: PassengerId , Name  are unique variables

# ### Part 2: Exploring the data using visualization tools

# ### Missing features in data

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.isnull().sum()


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"

# In[ ]:


test.isnull().sum()


# In[ ]:


plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='Set2')


# In[ ]:


train.Survived.value_counts()


# Survived is a dependent/target variable 
# 
# We can see that the Survival rate(1) is very less

# In[ ]:


plt.figure(figsize=(10,8))
sns.set_style('whitegrid')
sns.countplot(x='Sex',hue='Survived',data=train,palette='Set1')


# In[ ]:


Per_Sur = pd.crosstab(train["Sex"],train["Survived"])
Per_Sur["Percentage_Survived"] = round((Per_Sur[1]/(Per_Sur[0]+Per_Sur[1]))*100,2).astype(str) + "%"
Per_Sur


# Female Survival rate is 74% which is higher than Male Survival rate 18%. So this is an important variable for the model

# In[ ]:


plt.figure(figsize=(12,8))
sns.set_style('whitegrid')
sns.countplot(x='Pclass',hue='Survived',data=train,palette='rainbow')


# In[ ]:


Per_Sur=pd.crosstab(train["Pclass"],train["Survived"])
Per_Sur["Percentage_Survived"] = round((Per_Sur[1]/(Per_Sur[0]+Per_Sur[1]))*100,2).astype(str) + "%"
Per_Sur


# The passengers travelling in First class has a good survival rate 62%
# 
# Second class with 47% has average survival rate
# 
# Third class with 24% is the least
# 
# Hence this acts as an important variable to describe the model

# In[ ]:


plt.figure(figsize=(16,8))
sns.distplot(train.loc[train['Survived']==0,'Age'].dropna(),kde=False,color='red',bins=30)
sns.distplot(train.loc[train['Survived']==1,'Age'].dropna(),kde=False,color='blue',bins=30)
plt.legend(train['Survived'])


# From the above graph no proper conclusion can be made, further analysis is required 

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),annot=True)


# Age is Highly correlated(Negative) to Pclass than any other variables.
# 
# So Null values in Age have to be replaced with Pclass wise Mean values

# In[ ]:


train.loc[(train["Pclass"]==1) & (train.Age.notna() == False),"Age"]=train.loc[train["Pclass"]==1].Age.mean()

train.loc[(train["Pclass"]==2) & (train.Age.notna() == False),"Age"]=train.loc[train["Pclass"]==2].Age.mean()

train.loc[(train["Pclass"]==3) & (train.Age.notna() == False),"Age"]=train.loc[train["Pclass"]==3].Age.mean()


# In[ ]:


train.Age.isnull().sum()


# In[ ]:


test.loc[(test["Pclass"]==1) & (test.Age.notna() == False),"Age"]=test.loc[test["Pclass"]==1].Age.mean()

test.loc[(test["Pclass"]==2) & (test.Age.notna() == False),"Age"]=test.loc[test["Pclass"]==2].Age.mean()

test.loc[(test["Pclass"]==3) & (test.Age.notna() == False),"Age"]=test.loc[test["Pclass"]==3].Age.mean()


# In[ ]:


test.Age.isnull().sum()


# Age is an important factor for the model

# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(x='SibSp',hue='Survived',data=train)


# We can see that independent Passengers with "0" count of Siblings or Spouse has less Survival rate

# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(x='Parch',hue='Survived',data=train)


# We can see that independent Passengers with "0" count of Parent or Children has less Survival rate

# In[ ]:


print(train.Cabin.isnull().sum(),"\n")
train.Cabin.unique()

#Cabin has more Null values and it does not signify anything, we can drop the variable


# In[ ]:


train.drop("Cabin",axis=1,inplace=True)
train.shape


# In[ ]:


test.drop("Cabin",axis=1,inplace=True)
test.shape


# In[ ]:


print(train.Embarked.isnull().sum(),"\n")

train["Embarked"].fillna(train["Embarked"].mode()[0],inplace=True)#Imputing Null values with mode

train.Embarked.unique()


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x=train['Embarked'].dropna(),hue='Survived',data=train)


# In[ ]:


Per_Sur = pd.crosstab(train['Embarked'],train['Survived'])
Per_Sur["Percentage_Survived"] = round((Per_Sur[1]/(Per_Sur[0]+Per_Sur[1]))*100,2).astype(str)+"%"
Per_Sur


# Passengers from Southampton (S) has less Survival rate hence this also an important variable for prediction

# In[ ]:


train.loc[train["Fare"] == max(train.Fare),"Fare" ] = 263
#Imputing the outlier 500 with 263(Original max fare)


# In[ ]:


plt.figure(figsize=(16,8))
sns.distplot(train.loc[train['Survived']==0,'Fare'].dropna(),kde=False,color='blue',bins=30)
sns.distplot(train.loc[train['Survived']==1,'Fare'].dropna(),kde=False,color='green',bins=30)
plt.legend(train['Survived'])


# Passengers who bought tickets with Higher Fare(PClass = 1) has Higher Survival Rate

# In[ ]:


test.isnull().sum()


# In[ ]:


plt.figure(figsize=(16,8))
sns.distplot(test['Fare'].dropna(),kde=False,color='blue',bins=30)


# In[ ]:


test.loc[test["Fare"] == max(test.Fare),"Fare" ] = 263
#Imputing the outlier 500 with 263(Original max fare)


# In[ ]:


test.Fare.fillna(round(test.Fare.mean(),3),inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


#Dropping few variables which are not important in train data
train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
train.shape


# In[ ]:


#Dropping few variables which are not important in test data
test.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test.shape


# ### Part 3: Splitting the train data into train_model data and validation data and Building Model, predicting the results on validation data.
# 
# 1)Logistic Regression
# 
# 2)Decision Tree
# 
# 3)Random Forest
# 
# 4)Extra Tree Classifier
# 
# 5)K-Nearest Neighbours
# 

# In[ ]:


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.tree import DecisionTreeClassifier #Decision tree 
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.ensemble import ExtraTreesClassifier #Extra Tree Classifier
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# #### Preprocessing

# In[ ]:


colname=["Sex","Age",'Embarked']


# In[ ]:


#Label Encoding - converting objects to int

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for x in colname:
     train[x]=le.fit_transform(train[x])

for x in colname:
     test[x]=le.fit_transform(test[x])


# In[ ]:


#Data Splitting - train data is split into train_m and val data in 70, 30 ratio

train_m,val=train_test_split(train,test_size=0.3,random_state=10)

train_X = train_m.values[:,1:]
train_Y = train_m.values[:,0:1]

val_X = val.values[:,1:]
val_Y = val.values[:,0:1]


test_X= test.values[:,0:]


# In[ ]:


#Scaling - to bring all the data onto a common scale

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_X)

train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)


# ### Logistic Regression

# In[ ]:


log = LogisticRegression(random_state=100)
log.fit(train_X,train_Y)
pred_log=log.predict(val_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(pred_log,val_Y))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y, pred_log)
print(cfm)


# ### Decision Tree 

# In[ ]:


DT = DecisionTreeClassifier(random_state=100)
DT.fit(train_X,train_Y)
pred_DT = DT.predict(val_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(pred_DT,val_Y))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y, pred_DT)
print(cfm)


# ### Random Forests
# 

# In[ ]:


Ran_F=RandomForestClassifier(n_estimators=100,random_state=100)
Ran_F.fit(train_X,train_Y)
pred_Ran_F=Ran_F.predict(val_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(pred_Ran_F,val_Y))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y,pred_Ran_F)
print(cfm)


# ### Extra Tree Classfier

# In[ ]:


Extra_tr = ExtraTreesClassifier(random_state=100)
Extra_tr.fit(train_X,train_Y)
pred_Extra_tr = Extra_tr.predict(val_X)
print('The accuracy of the Extra Tree is',metrics.accuracy_score(pred_Extra_tr,val_Y))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y,pred_Extra_tr)
print(cfm)


# ### K-Nearest Neighbours(KNN)
# 

# In[ ]:


KNN=KNeighborsClassifier() 
KNN.fit(train_X,train_Y)
pred_KNN=KNN.predict(val_X)
print('The accuracy of the KNN is',metrics.accuracy_score(pred_KNN,val_Y))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y,pred_KNN)
print(cfm)


# In[ ]:


a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(val_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,val_Y)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=8) 
KNN.fit(train_X,train_Y)
pred_KNN=KNN.predict(val_X)
print('The accuracy of the KNN is',metrics.accuracy_score(pred_KNN,val_Y))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y,pred_KNN)
print(cfm)


# Though maximum Accuracy is 83.2% when number of neighbors = 8 for KNN  but there is no balance between FN and FP errors 

# #### Accuracies of the above models

# In[ ]:


models = [metrics.accuracy_score(pred_log,val_Y),metrics.accuracy_score(pred_DT,val_Y),metrics.accuracy_score(pred_Ran_F,val_Y),
          metrics.accuracy_score(pred_Extra_tr,val_Y),metrics.accuracy_score(pred_KNN,val_Y)]

m_names = ["Logistic","Decision Tree","Random Forest","Extra Tree","KNN"]

for x,y in zip(models,m_names):
    print("Accuracy for",y, "is :", round(x*100,2))


# ### Part 4: Feature engineering and creating a best suitable model
# Since KNN models and Random Forest are giving good Accuracy values when compared to other models
# 
# Random Forest is more preferable since :
# 
# 1. we can get important features for model tunning
# 
# 2. when compared to KNN,Random forest has a good balance between False Positive and False Negative Errors
# 

# In[ ]:


cols = train.columns[1:]
print(cols,"\n")
imp=Ran_F.feature_importances_
print(imp,"\n")

features = pd.DataFrame(cols,columns=["Features"])
features["Importance"] = imp
features = features.sort_values("Importance",ascending=False)
features


# In[ ]:


train_trail = train.copy()


# In[ ]:


train_trail.drop(["Parch","Embarked"],axis=1,inplace=True)


# In[ ]:


train_trail.drop("SibSp",axis=1,inplace=True)


# In[ ]:


test.drop(["Parch","Embarked","SibSp"],axis=1,inplace=True)


# In[ ]:


#Data Splitting - train_rev is split into train_m and val data in 70, 30 ratio

train_m,val=train_test_split(train_trail,test_size=0.3,random_state=10)

train_X = train_m.values[:,1:]
train_Y = train_m.values[:,0:1]

val_X = val.values[:,1:]
val_Y = val.values[:,0:1]


test_X= test.values[:,0:]

#Scaling - to bring all the data onto a common scale

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_X)

train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)


# In[ ]:


Ran_F=RandomForestClassifier(n_estimators=100,random_state=100)
Ran_F.fit(train_X,train_Y)
pred_Ran_F=Ran_F.predict(val_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(pred_Ran_F,val_Y))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(val_Y,pred_Ran_F)
print(cfm)
print("classification_report: ")
print(classification_report(val_Y, pred_Ran_F))
acc=accuracy_score(val_Y, pred_Ran_F)
auc = metrics.roc_auc_score(val_Y, pred_Ran_F)
print("Accuracy of the model: ", acc)
print("AUC of the model: ", auc)


# In[ ]:


from sklearn import metrics

fpr, tpr,z = metrics.roc_curve(val_Y, pred_Ran_F)# receiver operating characteristics
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)


# In[ ]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()


# After the model tunning(Feature Elimination) based on the important variables from Random Forest model
# 
# The Final Accuracy is 82.8%
# 
# Total Error is : 46 (FN = 25, FP=21) - the errors are balanced

# ### Identifying best Threshold 

# In[ ]:


y_pred_prob=Ran_F.predict_proba(val_X)


# In[ ]:


#A generic code to find values at different thresholds to pick up the best
for a in np.arange(0.4,0.6,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(val_Y, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
        cfm[1,0]," , type 1 error:", cfm[0,1])


# Model gave the best Accuracy at Threshold 0.5, no need to change the default threshold

# ### Performing Cross Validation to check for better model 

# In[ ]:


#performing kfold_cross_validation
from sklearn.model_selection import KFold

kfold_cv = KFold(n_splits=10,random_state=10)
print(kfold_cv,"\n")

from sklearn.model_selection import cross_val_score 
kfold_cv_result=cross_val_score(estimator=Ran_F,X=train_X,y=train_Y, cv=kfold_cv)
print(kfold_cv_result,"\n")
print(kfold_cv_result.mean())


# Cross Validation at any number of splits didn't give good results. So we will have to stick with the tunned model of Random Forest

# ### 5. Prediting the results on test data and submiting the results

# In[ ]:


pred_Ran_F=Ran_F.predict(test_X)


# In[ ]:


test["Survived"] = pred_Ran_F.astype(np.int)


# In[ ]:


test.head()


# In[ ]:


test.Survived.value_counts()


# In[ ]:


test.shape


# In[ ]:


test.to_excel("Titanic_Survival_predictios.xlsx",index = False,header = True)


# ### Conclusion 
# 
# 1. Final Model - Random Forest(after model tunning - feature elimination)
# 
# 2. Model Accuracy - 82.8%
# 
# 3. Type 2 Erros = 25
# 
# 4. Type 1 Errors = 21
# 
# 5. Area Under Curve(AUC) = 0.80
# 
# 5. We may improve the Accuracy by creating new variables from other features or by finding relationships between the variables but Accuracy is not the only important measure for model building, Recall, {Precision, Total Error shoul also be considerd.

# In[ ]:




