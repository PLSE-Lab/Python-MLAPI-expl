#!/usr/bin/env python
# coding: utf-8

# # Start with simplest problem
# 
# I feel like clasification is the easiest problem catogory to start with.
# We will start with simple clasification problem to predict survivals of  titanic https://www.kaggle.com/c/titanic

# # Contents
# 1. [Basic pipeline for a predictive modeling problem](#1)
# 1. [Exploratory Data Analysis (EDA)](#2)
#     * [Overall survival stats](#2_1)
#     * [Analysis features](#2_2)
#         1. [Sex](#2_2_1)
#         1. [Pclass](#2_2_2)
#         1. [Age](#2_2_3)
#         1. [Embarked](#2_2_4)
#         1. [SibSip & Parch](#2_2_5)
#         1. [Fare](#2_2_6) 
#     * [Observations Summary](#2_3)
#     * [Correlation Between The Features](#2_4)
# 1. [Feature Engineering and Data Cleaning](#4)
#     * [Converting String Values into Numeric](#4_1)
#     * [Convert Age into a categorical feature by binning](#4_2)
#     * [Convert Fare into a categorical feature by binning](#4_3)
#     * [Dropping Unwanted Features](#4_4)
# 1. [Predictive Modeling](#5)
#     * [Cross Validation](#5_1)
#     * [Confusion Matrix](#5_2)
#     * [Hyper-Parameters Tuning](#5_3)
#     * [Ensembling](#5_4)
#     * [Prediction](#5_5)
# 1. [Feature Importance](#6)
# 

# ## **Basic Pipeline for predictive modeling problem**[^](#1)<a id="1" ></a><br>
# 
# **<left><span style="color:blue">Exploratory Data Analysis</span> -> <span style="color:blue">Feature Engineering and Data Preparation</span> -> <span style="color:blue">Predictive Modeling</span></left>.**
# 
# 1. First we need to see what the data can tell us: We call this **<span style="color:blue">Exploratory Data Analysis(EDA)</span>**. Here we look at data which is hidden in rows and column format and try to visualize, summarize and interprete it looking for information.
# 1. Next we can **leverage domain knowledge** to boost machine learning model performance. We call this step, **<span style="color:blue">Feature Engineering and Data Cleaning</span>**. In this step we might add few features, Remove redundant features, Converting features into suitable form for modeling.
# 1. Then we can move on to the **<span style="color:blue">Predictive Modeling</span>**. Here we try basic ML algorthms, cross validate, ensemble and Important feature Extraction.

# ---
# 
# ## Exploratory Data Analysis (EDA)[^](#2)<a id="2" ></a><br>
# 
# With the objective in mind that this kernal aims to explain the workflow of a predictive modelling problem for begginers, I will try to use simple easy to understand visualizations in the EDA section. Kernals with more advanced EDA sections will be mentioned at the end for you to learn more.

# In[ ]:


# Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


# Read data to a pandas data frame
data=pd.read_csv('../input/train.csv')
# lets have a look on first few rows
display(data.head())
# Checking shape of our data set
print('Shape of Data : ',data.shape)


# * We have 891 data points (rows); each data point has 12 columns.

# In[ ]:


#checking for null value counts in each column
data.isnull().sum()


# * The Age, Cabin and Embarked have null values.

# ### Lets look at overall survival stats[^](#2_1)<a id="2_1" ></a><br>

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(13,5))
data['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# * Sad Story! Only 38% have survived. That is roughly 340 out of 891. 

# ---
# ### Analyse features[^](#2_2)<a id="2_2" ></a><br>

# #### Feature: Sex[^](#3_2_1)<a id="2_2_1" ></a><br>

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(18,5))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Fraction of Survival with respect to Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Survived vs Dead counts with respect to Sex')
sns.barplot(x="Sex", y="Survived", data=data,ax=ax[2])
ax[2].set_title('Survival by Gender')
plt.show()


# * While survival rate for female is around 75%, same for men is about 20%.
# * It looks like they have given priority to female passengers in the rescue.
# * **Looks like Sex is a good predictor on the survival.**

# ---
# #### Feature: Pclass[^](#2_2_2)<a id="2_2_2" ></a><br>
# **Meaning :** Ticket class : 1 = 1st, 2 = 2nd, 3 = 3rd

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(18,5))
data['Pclass'].value_counts().plot.bar(color=['#BC8F8F','#F4A460','#DAA520'],ax=ax[0])
ax[0].set_title('Number Of Passengers with respect to Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Survived vs Dead counts with respect to Pclass')
sns.barplot(x="Pclass", y="Survived", data=data,ax=ax[2])
ax[2].set_title('Survival by Pclass')
plt.show()


# * For Pclass 1 %survived is around 63%, for Pclass2 is around 48% and for Pclass2 is around 25%.
# * **So its clear that higher classes had higher priority while rescue.**
# * **Looks like Pclass is also an important feature.**

# ---
# #### Feature: Age[^](#2_2_3)<a id="2_2_3" ></a><br>
# **Meaning :** Age in years

# In[ ]:


# Plot
plt.figure(figsize=(25,6))
sns.barplot(data['Age'],data['Survived'], ci=None)
plt.xticks(rotation=90);


# * Survival rate for passenegers below Age 14(i.e children) looks to be good than others.
# * So Age seems an important feature too.
# * Rememer we had 177 null values in the Age feature. How are we gonna fill them?.

# #### Filling Age NaN
# 
# Well there are many ways to do this. One can use the mean value or median .. etc.. But can we do better?. Seems yes. [EDA To Prediction(DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic#EDA-To-Prediction-(DieTanic)) has used a wonderful method which I would use here too. There is a name feature. First lets extract the initials.
# 

# In[ ]:


data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex


# Okay so there are some misspelled Initials like Mlle or Mme that stand for Miss. Lets replace them.

# In[ ]:


data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


data.groupby('Initial')['Age'].mean() #lets check the average age by Initials


# In[ ]:


## Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46


# In[ ]:


data.Age.isnull().any() #So no null values left finally 


# ---
# #### Feature: Embarked[^](#2_2_4)<a id="2_2_4" ></a><br>
# **Meaning :** Port of Embarkation. C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,5))
sns.countplot('Embarked',data=data,ax=ax[0])
ax[0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Embarked vs Survived')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# * Majority of passengers borded from Southampton
# * Survival counts looks better at C. Why?. Could there be an influence from sex and pclass features we already studied?. Let's find out 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,5))
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0])
ax[0].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1])
ax[1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# * We guessed correctly. higher % of 1st class passegers boarding from C might be the reason.

# #### Filling Embarked NaN

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(6,5))
data['Embarked'].value_counts().plot.pie(explode=[0,0,0],autopct='%1.1f%%',ax=ax)
plt.show()


# * Since 72.5% passengers are from Southampton, So lets fill missing 2 values using S (Southampton)

# In[ ]:


data['Embarked'].fillna('S',inplace=True)


# In[ ]:


data.Embarked.isnull().any()


# ---
# #### Features: SibSip & Parch[^](#2_2_5)<a id="2_2_5" ></a><br>
# **Meaning :**  
# SibSip -> Number of siblings / spouses aboard the Titanic
# 
# Parch -> Number of parents / children aboard the Titanic
# 
# SibSip + Parch -> Family Size 

# In[ ]:


f,ax=plt.subplots(2,2,figsize=(15,10))
sns.countplot('SibSp',hue='Survived',data=data,ax=ax[0,0])
ax[0,0].set_title('SibSp vs Survived')
sns.barplot('SibSp','Survived',data=data,ax=ax[0,1])
ax[0,1].set_title('SibSp vs Survived')

sns.countplot('Parch',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Parch vs Survived')
sns.barplot('Parch','Survived',data=data,ax=ax[1,1])
ax[1,1].set_title('Parch vs Survived')

plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# * The barplot and factorplot shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. The graph roughly decreases if the number of siblings increase.

# Lets combine above and analyse family size. 

# In[ ]:


data['FamilySize'] = data['Parch'] + data['SibSp']
f,ax=plt.subplots(1,2,figsize=(15,4.5))
sns.countplot('FamilySize',hue='Survived',data=data,ax=ax[0])
ax[0].set_title('FamilySize vs Survived')
sns.barplot('FamilySize','Survived',data=data,ax=ax[1])
ax[1].set_title('FamilySize vs Survived')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# * This looks interesting! looks like family sizes of 1-3 have better survival rates than others.

# ---
# #### Fare[^](#2_2_6)<a id="2_2_6" ></a><br>
# **Meaning :** Passenger fare

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(20,5))
sns.distplot(data.Fare,ax=ax)
ax.set_title('Distribution of Fares')
plt.show()


# In[ ]:


print('Highest Fare:',data['Fare'].max(),'   Lowest Fare:',data['Fare'].min(),'    Average Fare:',data['Fare'].mean())
data['Fare_Bin']=pd.qcut(data['Fare'],6)
data.groupby(['Fare_Bin'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# * It is clear that as Fare Bins increase chances of survival increase too.

# #### Observations Summary[^](#2_3)<a id="2_3" ></a><br>

# **Sex:** Survival chance for female is better than that for male.
# 
# **Pclass:** Being a 1st class passenger gives you better chances of survival.
# 
# **Age:** Age range 5-10 years have a high chance of survival.
# 
# **Embarked:** Majority of passengers borded from Southampton.The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. All most all Passengers at Q were from Pclass3.
# 
# **Family Size:** looks like family sizes of 1-3 have better survival rates than others.
# 
# **Fare:** As Fare Bins increase chances of survival increases
# 
# 

# #### Correlation Between The Features[^](#2_4)<a id="2_4" ></a><br>

# In[ ]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# ---
# ## Feature Engineering and Data Cleaning[^](#4)<a id="4" ></a><br>
# Now what is Feature Engineering? Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work.
# 
# In this section we will be doing,
# 1. Converting String Values into Numeric
# 1. Convert Age into a categorical feature by binning
# 1. Convert Fare into a categorical feature by binning
# 1. Dropping Unwanted Features
# 

# #### Converting String Values into Numeric[^](#4_1)<a id="4_1" ></a><br>
# Since we cannot pass strings to a machine learning model, we need to convert features Sex, Embarked, etc into numeric values.

# In[ ]:


data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# #### Convert Age into a categorical feature by binning[^](#4_2)<a id="4_2" ></a><br>

# In[ ]:


print('Highest Age:',data['Age'].max(),'   Lowest Age:',data['Age'].min())


# In[ ]:


data['Age_cat']=0
data.loc[data['Age']<=16,'Age_cat']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_cat']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_cat']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_cat']=3
data.loc[data['Age']>64,'Age_cat']=4


# #### Convert Fare into a categorical feature by binning[^](#4_3)<a id="4_3" ></a><br>

# In[ ]:


data['Fare_cat']=0
data.loc[data['Fare']<=7.775,'Fare_cat']=0
data.loc[(data['Fare']>7.775)&(data['Fare']<=8.662),'Fare_cat']=1
data.loc[(data['Fare']>8.662)&(data['Fare']<=14.454),'Fare_cat']=2
data.loc[(data['Fare']>14.454)&(data['Fare']<=26.0),'Fare_cat']=3
data.loc[(data['Fare']>26.0)&(data['Fare']<=52.369),'Fare_cat']=4
data.loc[data['Fare']>52.369,'Fare_cat']=5


# #### Dropping Unwanted Features[^](#4_4)<a id="4_4" ></a><br>
# 
# Name--> We don't need name feature as it cannot be converted into any categorical value.
# 
# Age--> We have the Age_cat feature, so no need of this.
# 
# Ticket--> It is any random string that cannot be categorised.
# 
# Fare--> We have the Fare_cat feature, so unneeded
# 
# Cabin--> A lot of NaN values and also many passengers have multiple cabins. So this is a useless feature.
# 
# Fare_Bin--> We have the fare_cat feature.
# 
# PassengerId--> Cannot be categorised.
# 
# Sibsp & Parch --> We got FamilySize feature
# 

# In[ ]:


#data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
data.drop(['Name','Age','Fare','Ticket','Cabin','Fare_Bin','SibSp','Parch','PassengerId'],axis=1,inplace=True)


# In[ ]:


data.head(2)


# In[ ]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# ---
# ## Predictive Modeling[^](#5)<a id="5" ></a><br>
# 

# Now after data cleaning and feature engineering we are ready to train some classification algorithms that will make predictions for unseen data. We will first train few classification algorithms and see how they perform. Then we can look how an ensemble of classification algorithms perform on this data set.
# Following Machine Learning algorithms will be used in this kernal.
# 
# * Logistic Regression Classifier
# * Naive Bayes Classifier
# * Decision Tree Classifier
# * Random Forest Classifier
# 

# In[ ]:


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[ ]:


#Lets prepare data sets for training. 
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']


# In[ ]:


data.head(2)


# In[ ]:


# Logistic Regression
model = LogisticRegression(C=0.05,solver='liblinear')
model.fit(train_X,train_Y.values.ravel())
LR_prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression model is \t',metrics.accuracy_score(LR_prediction,test_Y))

# Naive Bayes
model=GaussianNB()
model.fit(train_X,train_Y.values.ravel())
NB_prediction=model.predict(test_X)
print('The accuracy of the NaiveBayes model is\t\t\t',metrics.accuracy_score(NB_prediction,test_Y))

# Decision Tree
model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
DT_prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is \t\t\t',metrics.accuracy_score(DT_prediction,test_Y))

# Random Forest
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y.values.ravel())
RF_prediction=model.predict(test_X)
print('The accuracy of the Random Forests model is \t\t',metrics.accuracy_score(RF_prediction,test_Y))


# ### Cross Validation[^](#5_1)<a id="5_1" ></a><br>
# 
# Accuracy we get here higlhy depends on the train & test data split of the original data set. We can use cross validation to avoid such problems arising from dataset splitting.
# I am using K-fold cross validation here. Watch this short [vedio](https://www.youtube.com/watch?v=TIgfjmp-4BA) to understand what it is.
# 

# In[ ]:


from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Logistic Regression','Decision Tree','Naive Bayes','Random Forest']
models=[LogisticRegression(solver='liblinear'),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# Now we have looked at cross validation accuracies to get an idea how those models work. There is more we can do to understand the performances of the models we tried ; let's have a look at confusion matrix for each model.

# ### Confusion Matrix[^](#5_2)<a id="5_2" ></a><br>

# A confusion matrix is a table that is often used to describe the performance of a classification model. read more [here](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

# In[ ]:


f,ax=plt.subplots(2,2,figsize=(10,8))
y_pred = cross_val_predict(LogisticRegression(C=0.05,solver='liblinear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Naive Bayes')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Random-Forests')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()


# * By looking at above matrices we can say that, if we are more concerned on making less mistakes by predicting survived as dead, then Naive Bayes model does better.
# * If we are more concerned on making less mistakes by predicting dead as survived, then Decision Tree model does better.

# ### Hyper-Parameters Tuning[^](#5_3)<a id="5_3" ></a><br>
# 
# You might have noticed there are few parameters for each model which defines how the model learns. We call these hyperparameters. These hyperparameters can be tuned to improve performance. Let's try this for Random Forest classifier.

# In[ ]:


from sklearn.model_selection import GridSearchCV
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True,cv=10)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# * Best Score for Random Forest is with n_estimators=100

# ### Ensembling[^](#5_4)<a id="5_4" ></a><br>
# 
# Ensembling is a way to increase performance of a model by combining several simple models to create a single powerful model.
# read more about ensembling [here](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/).
# Ensembling can be done in ways like: Voting Classifier, Bagging, Boosting.
# 
# I will use voting method in this kernal

# In[ ]:


from sklearn.ensemble import VotingClassifier
estimators=[('RFor',RandomForestClassifier(n_estimators=100,random_state=0)),
            ('LR',LogisticRegression(C=0.05,solver='liblinear')),
            ('DT',DecisionTreeClassifier()),
            ('NB',GaussianNB())]
ensemble=VotingClassifier(estimators=estimators,voting='soft')
ensemble.fit(train_X,train_Y.values.ravel())
print('The accuracy for ensembled model is:',ensemble.score(test_X,test_Y))
cross=cross_val_score(ensemble,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())


# ### Prediction[^](#5_5)<a id="5_5" ></a><br>
# 
# We can see that ensemble model does better than individual models. lets use that for predictions.

# In[ ]:


Ensemble_Model_For_Prediction=VotingClassifier(estimators=[
                                       ('RFor',RandomForestClassifier(n_estimators=200,random_state=0)),
                                       ('LR',LogisticRegression(C=0.05,solver='liblinear')),
                                       ('DT',DecisionTreeClassifier(random_state=0)),
                                       ('NB',GaussianNB())
                                             ], 
                       voting='soft')
Ensemble_Model_For_Prediction.fit(X,Y)


# We need to  do some preprocessing to  this test data set before we can feed that to the trained model.

# In[ ]:


test=pd.read_csv('../input/test.csv')
IDtest = test["PassengerId"]
test.head(2)


# In[ ]:


test.isnull().sum()


# In[ ]:


# Prepare Test Data set for feeding

# Construct feature Initial
test['Initial']=0
for i in test:
    test['Initial']=test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
    
test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)

# Fill Null values in Age Column
test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=33
test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=36
test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=5
test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22
test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age']=46

# Fill Null values in Fare Column
test.loc[(test.Fare.isnull()) & (test['Pclass']==3),'Fare'] = 12.45

# Construct feature Age_cat
test['Age_cat']=0
test.loc[test['Age']<=16,'Age_cat']=0
test.loc[(test['Age']>16)&(test['Age']<=32),'Age_cat']=1
test.loc[(test['Age']>32)&(test['Age']<=48),'Age_cat']=2
test.loc[(test['Age']>48)&(test['Age']<=64),'Age_cat']=3
test.loc[test['Age']>64,'Age_cat']=4

# Construct feature Fare_cat
test['Fare_cat']=0
test.loc[test['Fare']<=7.775,'Fare_cat']=0
test.loc[(test['Fare']>7.775)&(test['Fare']<=8.662),'Fare_cat']=1
test.loc[(test['Fare']>8.662)&(test['Fare']<=14.454),'Fare_cat']=2
test.loc[(test['Fare']>14.454)&(test['Fare']<=26.0),'Fare_cat']=3
test.loc[(test['Fare']>26.0)&(test['Fare']<=52.369),'Fare_cat']=4
test.loc[test['Fare']>52.369,'Fare_cat']=5

# Construct feature FamilySize
test['FamilySize'] = test['Parch'] + test['SibSp']

# Drop unwanted features
test.drop(['Name','Age','Ticket','Cabin','SibSp','Parch','Fare','PassengerId'],axis=1,inplace=True)

# Converting String Values into Numeric 
test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

test.head(2)


# In[ ]:


# Predict
test_Survived = pd.Series(ensemble.predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("predictions.csv",index=False)


# ## Feature Importance[^](#6)<a id="6" ></a><br>
# 
# Well after we have trained a model to make predictions for us, we feel curiuos on how it works. What are the features model weights more when trying to make a prediction?. As humans we seek to understand how it works. Looking at feature importances of a trained model is one way we could explain the decisions it make. Lets visualize the feature importances of the Random forest model we used inside the ensemble above.

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(6,6))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax)
ax.set_title('Feature Importance in Random Forests')
plt.show()


# **If You Like the notebook and think that it helped you, please upvote to It keep motivate me**
