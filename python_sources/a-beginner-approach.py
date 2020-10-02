#!/usr/bin/env python
# coding: utf-8

#    Hello guys,
#     
#    In this kernel I try to approach the Titanic: Machine Learning from Disaster competition, after let's say one month of training on DataScience/ML. I discovered this site about 3-4 months ago, but never committed to this field until 1 month ago. After I did some researching in these areas and reading some kernels I try to put here all the information I got and I try to explain all I do in this case maybe some kagglers with experience will correct me and give some better information/explanation. Maybe this will help new kagglers to try new things and actually have the courage to compete after seeing this.

#    First of all I will try to follow some workflow I found because I want to work in an organized way and maybe in this way I wont miss some things out.
#    
#    The workflow I found:
#        
# 1.     Question or problem definition.
# 2.     Acquire training and testing data.
# 3.     Wrangle, prepare, cleanse the data.
# 4.     Analyze, identify patterns, and explore the data.
# 5.     Model, predict and solve the problem.
# 6.     Visualize, report, and present the problem solving steps and final solution.
# 7.     Supply or submit the results.

#  1.Question or problem definition.
#  
#   In this step I think I should write down what is the main idea of what I am doing or what is the scope of this competion, all the information are written in Overview area of the competion, but I will give my opinion about this. In my opinion this competion is all about preparing youself for the next competition because it doesn't have some real value, maybe in nowdays we want to prepare a ship to not wreck into something not to see who survives if it wrecks, just my opinion. About the competition itself is about the notorious Titanic and the tragic event that happened. We some a vital information which states that 1502 out of 2224 died, so that's 0.675% without even trying to predict something. Another information that I think is a hint gave by kaggle is "some groups of people were more likely to survive than others", so this means we have to discover something to gave a prediction of them.
#   
#   Here I copy-paste what's written on competition itself:
#   
#   Competition Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# 2.Acquire training and testing data.
# 
# I think this step is clear by everyone starting out, we have to gather data, an input to have some results and you to this before anything else, I don't see how we can do this after some other operations.
# 
# I will import some libraries that I usually do, I don't know if this is a correct way to do it but I will import them all.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Now I'm new to this kernel stuff, usually I read the data that is in the same folder as my python/jupyter notebook file and I have to import them as follows: pd.read_csv('file.csv').
# But I think here in this case I have to find the kaggle's folder and import them from there.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#So after I ran the cell above it gave me some folder with 3 files and I will from them
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')


# 3.Wrangle, prepare, cleanse the data.
# 
# So after I read a lot of articles on data science all the time, like 90% it says that the most important thing is to prepare the data, so that's what I'm going to do or atleast try.

# In[ ]:


#first of all let's see what kind of data we have
titanic_train.head(10)


# Now I will try to analyze the columns to see what we have to do, I think in this case it is ok to do it because we have a small number of columns but maybe just in this case.
# 
# PassengerId -> I think this doesn't give special information so I will drop it
# 
# Survived -> this will be our y variable
# 
# Pclass -> I think this is one of the most importance feature we have
# 
# Name -> As well as the PassengerId I thought that it is kind of useless, but after I read some kernels it has some special rank in it
# Sex -> is also important because females have a higher chance of surviving
# 
# Age -> along with females children and elders have a higher chance of surviving
# 
# SibSp & Parch -> I don't know about this or what should I extract out of it, maybe people who have family related persons on the boat, usually children or females can escape with them
# 
# Ticket -> I think this is just a random number given to you when you went on this boat, what I can see at a first glance is that is somehow corellated with your class, if your ticket stars with 1 you are class 1 and so on
# 
# Fare -> this is high correlated with your class
# 
# Cabin -> I think this is where you are positionated on the boat so it should be important, but I see a lot of missing values on this
# 
# Embarked -> I don't see how your port of embarkation will escape you from this fate, the single thing I'm thinking of is that maybe some ports have more rich people in that area and we can correlate this information, but we don't see this in our database

# The next thing that I am going to do is to drop the PassengerId and Ticket columns from the test and train dataset. I must be carefull not to remove the PassengerId column in the test dataset before I save them for my final submission.

# In[ ]:


titanic_train.drop(['PassengerId'],axis=1,inplace=True)
titanic_train.drop(['Ticket'],axis=1,inplace=True)
ID_test = titanic_test['PassengerId']
titanic_test.drop(['PassengerId'],axis=1,inplace=True)
titanic_test.drop(['Ticket'],axis=1,inplace=True)
titanic_test


# Now I have to analyze some columns to see some hidden correlation and make some visualization, this part is the most unknown for me and hard to do, so most of it will be some inspiration from other kerners.
# 
# I will try to see the survival rate by Pclass, SibSp, Parch and Sex.

# In[ ]:


titanic_train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


titanic_train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Sex',ascending=False)


# In[ ]:


titanic_train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# Here I found something that bugged me for let's say 20-30 minutes because when I wanted to make the prediction for my test data set I found out that I trained a model on 28 features and my test data had 29. After I inspected my code for 20 minutes I found out that 'Parch' in train data had 7 values and in the test had 8, so when I OneHotEncoded them this put one more feature and I couldn't make a prediction. The value that was in addition in comparation with the train set was 2 values of 9 so I raplced them with the higher next value which is 6.

# In[ ]:


titanic_test['Parch']=titanic_test['Parch'].replace(9,6)
titanic_test['Parch'].value_counts()


# In[ ]:


titanic_train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


f = sns.FacetGrid(titanic_train, col='Survived')
f.map(plt.hist, 'Age', bins=10)


# In[ ]:


f = sns.FacetGrid(titanic_train, col='Survived',row='Pclass')
f.map(sns.distplot,'Age')


# In[ ]:


f = sns.FacetGrid(titanic_train, col='Survived',row='Sex')
f.map(sns.distplot,'Age',bins=20)


# We can see that in every class your rate of surviving decrease if you are am adult(+20-50) as well as you are in a lower class.
# 
# Now I will prepare the data for the ML part, I'm gonna drop all the missing lines that have no Survived values on this column.

# In[ ]:


titanic_train.dropna(axis=0,subset=['Survived'],inplace=True)


# Splitting the data in X and y.

# In[ ]:


y = titanic_train['Survived']
titanic_train.drop(['Survived'],axis=1,inplace=True)
X = titanic_train


# Before encoding the data I will impute the missing values from the train and test set, but first I have to see what columns have missing values. We can see the Cabin, Age and Embarked have missing values.

# In[ ]:


total = titanic_train.isnull().sum().sort_values(ascending=False)
percent = (titanic_train.isnull().sum()/titanic_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# As I said before the Cabin, in my opinion is not that important so I'm gonna remove it.

# In[ ]:


X.drop(['Cabin'],inplace=True,axis=1)
titanic_test.drop(['Cabin'],inplace=True,axis=1)


# Now we gonna impute the missing columns, for the Age column I am gonna use the mean value and for the Embarked one I am gonna see what is the most frequent value in the column. Also after some investigation it seems that the test data also had 2 missing values on the Fare column so we gonna impute them with the mean too.

# In[ ]:


from sklearn.preprocessing import Imputer
imputer_1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_1 = imputer_1.fit(X.loc[:,['Age']])
X.loc[:,['Age']] = imputer_1.transform(X.loc[:,['Age']])

imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)
imputer_1_test = imputer_1_test.fit(titanic_test.loc[:,['Age']])
titanic_test.loc[:,['Age']] = imputer_1_test.transform(titanic_test.loc[:,['Age']])

#we have one missing fare value
imputer_1_test = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)
imputer_1_test = imputer_1_test.fit(titanic_test.loc[:,['Fare']])
titanic_test.loc[:,['Fare']] = imputer_1_test.transform(titanic_test.loc[:,['Fare']])

freq_port = X.Embarked.dropna().mode()[0]
X['Embarked']=X['Embarked'].fillna(freq_port)

freq_port = titanic_test.Embarked.dropna().mode()[0]
titanic_test['Embarked']=titanic_test['Embarked'].fillna(freq_port)


# Verifying that there is no missing data left.

# In[ ]:


total = X.isnull().sum().sort_values(ascending=False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# Encoding all the categorical data.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder_1 = LabelEncoder()
X.loc[:,['Embarked']] = label_encoder_1.fit_transform(X.loc[:,['Embarked']])
titanic_test.loc[:,['Embarked']] = label_encoder_1.fit_transform(titanic_test.loc[:,['Embarked']])

label_encoder_2 = LabelEncoder()
X.loc[:,['Sex']] = label_encoder_2.fit_transform(X.loc[:,['Sex']])
titanic_test.loc[:,['Sex']] = label_encoder_2.fit_transform(titanic_test.loc[:,['Sex']])


# Encoding the Name value into a new column named Title.(This was not written by me)

# In[ ]:


X['Title'] =X.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
titanic_test['Title'] = titanic_test.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
pd.crosstab(X['Title'], X['Sex'])


# In[ ]:


#continuing with the copy paste :<
X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col',     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

X['Title'] = X['Title'].replace('Mlle', 'Miss')
X['Title'] = X['Title'].replace('Ms', 'Miss')
X['Title'] = X['Title'].replace('Mme', 'Mrs')

titanic_test['Title'] = titanic_test['Title'].replace(['Lady', 'Countess','Capt', 'Col',     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanic_test['Title'] =titanic_test['Title'].replace('Mlle', 'Miss')
titanic_test['Title'] = titanic_test['Title'].replace('Ms', 'Miss')
titanic_test['Title'] = titanic_test['Title'].replace('Mme', 'Mrs')


# In[ ]:


X.drop(['Name'],inplace=True,axis=1)
titanic_test.drop(['Name'],inplace=True,axis=1)


# In[ ]:


label_encoder_3 = LabelEncoder()
X.loc[:,['Title']] = label_encoder_3.fit_transform(X.loc[:,['Title']])
titanic_test.loc[:,['Title']] = label_encoder_3.fit_transform(titanic_test.loc[:,['Title']])


# After all the preparations we are left with the Age and Fare column that have some weird values if we compare them with the rest of the dataset so I thought that maybe a scalling here will be better.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_1 = StandardScaler()
X.loc[:,['Age']] = sc_1.fit_transform(X.loc[:,['Age']])
titanic_test.loc[:,['Age']] = sc_1.fit_transform(titanic_test.loc[:,['Age']])

sc_2 = StandardScaler()
X.loc[:,['Fare']] = sc_2.fit_transform(X.loc[:,['Fare']])
titanic_test.loc[:,['Fare']] = sc_2.fit_transform(titanic_test.loc[:,['Fare']])


# And the last step is to OneHotEncode the rest of the data.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0,3,4,6,7])
X = onehotencoder.fit_transform(X).toarray()
titanic_test = onehotencoder.fit_transform(titanic_test).toarray()
titanic_test


# 5. Model, predict and solve the problem.
# 
# And now the fun part, splitting the data and creating the models.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# ->RandomForest
# 
# First model that came in my head is an ensemble model, the RandomForest one, and I wanted to see how it is performing.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)


# I will use the confusion_matrix for all the models here to see how are the results.

# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)


# And making a kFoldCrossValidation to calculate the accuracies and don't get fooled.

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X = X_train, y =y_train,cv = 10)
print(accuracies.mean())
print(accuracies.std())


# Also I wanted to calculate the Accuracy,Precision and the Recall.

# In[ ]:


#calculating the accuracy,recall,precision
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)
precision = cm[0][0]/(cm[0][0]+cm[0][1])
print(precision)
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print(recall)
#I think these are very good results


# ->SupportVectorMachine model

# In[ ]:


from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)
precision = cm[0][0]/(cm[0][0]+cm[0][1])
print(precision)
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print(recall)


# ->KNN model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


accuracies = cross_val_score(estimator=classifier,X = X_train, y =y_train,cv = 10)
print(accuracies.mean())
print(accuracies.std())


# In[ ]:


accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)
precision = cm[0][0]/(cm[0][0]+cm[0][1])
print(precision)
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print(recall)


# ->XGBoost
# 
# Our favorite one.

# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


accuracies = cross_val_score(estimator=classifier,X = X_train, y =y_train,cv = 10)
print(accuracies.mean())
print(accuracies.std())


# In[ ]:


accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)
precision = cm[0][0]/(cm[0][0]+cm[0][1])
print(precision)
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print(recall)


# And I saved what is best for the final.
# 
# ->NeuralNetwork

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = 6,init='uniform',activation='relu',input_dim=28))
classifier.add(Dense(output_dim = 6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=50)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)
precision = cm[0][0]/(cm[0][0]+cm[0][1])
print(precision)
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print(recall)


# 7. Supply or submit the results.

# In[ ]:


preds_test = classifier.predict(titanic_test)
preds_test = (preds_test>0.5)
cm = confusion_matrix(y_test,y_pred)
print(cm)
ids = []
preds = []
for i in range(0,len(ID_test)):
    ids.append(ID_test[i])
for i in range(0,len(preds_test)):
    if preds_test[i] :
        preds.append(1)
    else :
        preds.append(0)
print(preds)
output = pd.DataFrame({'PassengerId': ids,
                       'Survived': preds})
output.to_csv('submission.csv', index=False)


# This is the final of my kernel, I hope you find something usefull and for the 'big guys' here to see how a beginner approach this problem after 1 month of training. To be honest I am proud that I made it happened and it actually worked. After all I couldn't respect all the workflow I mentioned earlier because I don't know what to do in those steps.
# 
# Some thanks to this kernel: https://www.kaggle.com/startupsci/titanic-data-science-solutions (it helped me with the names part because at this moment I can't handle these kind of operations)
# 
# Please if you some helpful tips and tricks don't mind to share.
# 
# Also I am sorry for the mistakes I made writing this in English.
