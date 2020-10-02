#!/usr/bin/env python
# coding: utf-8

# > # Titanic with Data Science for Beginners

# Hey,
# 
# i got into Data Science at the beginning of 2020. So i'm kind of new to this kind of Workflow.
# Nevertheless let's grab some coffee, give it a try and see what happens. :)
# 
# feel free to give feedback and if you have any suggestion please let me know
# 
# 
# # work is still in progress :)
# 
# 

# # Includes
# 
# pretty basic python stuff. If you didn't heard from seaborn already, go and check it out

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data
# load dataset in pandas, with first row as index

# In[ ]:


df =  pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)
df.head(10)


# so, how many people we got here?

# In[ ]:


len(df.index)


# And how many did survived?

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,10))

df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, ax=ax[0])
df['Survived'].loc[df['Sex']=='male'].value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, ax=ax[1])
df['Survived'].loc[df['Sex']=='female'].value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, ax=ax[2])
ax[0].set_title('Overall Survived')
ax[1].set_title('Men Survived')
ax[2].set_title('Female Survived')

ax[0].set_ylabel('')
ax[1].set_ylabel('')
ax[2].set_ylabel('')

labels = ['dead', 'survived'] 
plt.legend(labels)


# How many values are missing?

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)


# In[ ]:


df.isnull().sum(axis = 0)


# Lots of NaN's in the Cabin may we got a workaround? outherwise we need to del it

# # Map the values
# 
# i would like to map the values in some order. 
# 
# In my mind its easier to make one fixed value for survived. So we switch the 'Survived' values.
# 
# * 0 = Survived
# * 1 = not Survied
# 
# All other values get sorted by change to survive. The highst value gets the lowest number.
# 
# So for example: 
# * 1If you got by Sex, Pclasses and Age a 0 your changes are pretty good to survive.
# * If you land by 1,3,5 you may better watch out for some swimming wings.

# ## Invert the 'Survived' Value
# 

# In[ ]:


df.loc[df["Survived"] == 0,"Survived"] = -1
df.loc[df["Survived"] == 1,"Survived"] = 0
df.loc[df["Survived"] == -1,"Survived"] = 1


# In[ ]:


df.head()


# ## Gender 
# 

# get the %'s

# In[ ]:


pd.crosstab(df.Sex,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# For my idea this means:
# 
# * Female = 0
# * Male = 1
# 

# In[ ]:


df.loc[df["Sex"] == "male","Sex"] = 1
df.loc[df["Sex"] == "female","Sex"] = 0
df.head()


# ## Embarked

#  For the embarked you see 2 NaNs in the upper graph so we replace them with the most commun value

# In[ ]:


df['Embarked'].value_counts() 


# Replace NaN with 's' and map the values

# In[ ]:


df['Embarked'] = df.Embarked.fillna('S')


# lets see in witch embarked the most people surived

# In[ ]:


pd.crosstab(df.Embarked,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# * C = 0 = best
# * Q = 1
# * S = 2 = worsted 
# 

# In[ ]:


df['Embarked'] = df['Embarked'].map( {'S': 2, 'Q': 1, 'C': 0} ).astype(int)
df.head()


# In[ ]:


sns.set(style="whitegrid")

g = sns.catplot(x="Embarked", y="Survived", hue="Sex", data=df, height=6, kind="bar", palette="Set1")
g.despine(left=True)
g.set_ylabels("death probability")


# mhh its ok butt...
# 
# my order is atm just for alle geanders may i should come back later and make a second model with serperatet gernders

# # Name 
# Lets try to work with the names
# 
# map by title

# In[ ]:


#df.loc[df["Name"].str.find('Mr.') >= 0, "Type"] = 0
#df.loc[df["Name"].str.find('Mrs.') >= 0, "Type"] = 1
#df.loc[df["Name"].str.find('Miss.') >= 0, "Type"] = 2
#df.loc[df["Name"].str.find('Master.') >= 0, "Type"] = 3
#df.loc[df["Name"].str.find('Don.') >= 0, "Type"] = 4
#df.loc[df["Name"].str.find('Rev.') >= 0, "Type"] = 4
#df.loc[df["Name"].str.find('Dr.') >= 0, "Type"] = 4
#df['Type'] = df.Type.fillna(4)


# Ok lets see what we got

# In[ ]:


#df['Type'].value_counts()


# In[ ]:


#pd.crosstab(df.Type,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# i know not perfect.
# i will change it later but for today it should be okay

# In[ ]:


#del df['Type']
df.loc[df["Name"].str.find('Mr.') >= 0, "Type"] = 4
df.loc[df["Name"].str.find('Mrs.') >= 0, "Type"] = 0
df.loc[df["Name"].str.find('Miss.') >= 0, "Type"] = 1
df.loc[df["Name"].str.find('Master.') >= 0, "Type"] = 2
df.loc[df["Name"].str.find('Don.') >= 0, "Type"] = 3
df.loc[df["Name"].str.find('Rev.') >= 0, "Type"] = 3
df.loc[df["Name"].str.find('Dr.') >= 0, "Type"] = 3
df['Type'] = df.Type.fillna(3)


# In[ ]:


pd.crosstab(df.Type,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# So make some graphics by type

# In[ ]:


df['Type'] = df['Type'].astype(int)


# In[ ]:


sns.set(style="whitegrid")

g = sns.catplot(x="Type", y="Survived", hue="Sex", data=df, height=6, kind="bar", palette="Set1")
g.despine(left=True)
g.set_ylabels("death probability")


# aaaand again big diffrenc because of the geners

# In[ ]:


df.head()


# # Fare

# In[ ]:


plt.figure(figsize=(20,5))
sns.boxplot(y="Survived", x="Fare", data=df, palette="Set2",  orient="h");


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.distplot(df[df['Survived']==0].Fare,ax=ax[0])
sns.distplot(df[df['Survived']==1].Fare,ax=ax[1])


# In[ ]:


df.loc[df["Fare"] < 10,"Fare"] = 5
df.loc[(df["Fare"] >= 10) & (df["Fare"] < 25),"Fare"] = 3
df.loc[(df["Fare"] >= 25) & (df["Fare"] < 50),"Fare"] = 4
df.loc[(df["Fare"] >= 50) & (df["Fare"] < 100),"Fare"] = 2
df.loc[(df["Fare"] >= 100), 'Fare'] = 0
df.head()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
sns.distplot(df[df['Survived']==0].Fare,ax=ax[0])
sns.distplot(df[df['Survived']==1].Fare,ax=ax[1])


# In[ ]:


pd.crosstab(df.Fare,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# In[ ]:


df['Fare'] = df['Fare'].astype(int)


# # Age
# 
# first we need to get rid of all the NaN's
# 
# the easiest way is to check the mean age and fill the NaNs with that value

# In[ ]:


df['Age'].isnull().sum()


# in the future may ai will check some better way to bredigt the age 

# In[ ]:


meanAge = df['Age'].mean()
print(meanAge)


# In[ ]:


df['Age'] = df.Age.fillna(meanAge)
df['Age'] = df['Age'].astype('int64')
df.head()


# In[ ]:


df.loc[df["Age"] < 15,"Age"] = 0
df.loc[(df["Age"] >= 15) & (df["Age"] < 30),"Age"] = 3
df.loc[(df["Age"] >= 30) & (df["Age"] < 45),"Age"] = 1
df.loc[(df["Age"] >= 45) & (df["Age"] < 60),"Age"] = 2
df.loc[(df["Age"] >= 60), 'Age'] = 4
df.head()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
sns.distplot(df[df['Survived']==0].Age,ax=ax[0])
sns.distplot(df[df['Survived']==1].Age,ax=ax[1])


# In[ ]:


pd.crosstab(df.Age,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# # Cabin
# 
# https://en.wikipedia.org/wiki/First_class_facilities_of_the_RMS_Titanic#/media/File:Titanic_cutaway_diagram.png
# 
# so the decks are splittet from a(top) to g(bottom)

# In[ ]:


df.Cabin.value_counts()


# In[ ]:


df.loc[df["Cabin"].str.find('A') >= 0, "Deck"] = 5
df.loc[df["Cabin"].str.find('B') >= 0, "Deck"] = 3
df.loc[df["Cabin"].str.find('C') >= 0, "Deck"] = 4
df.loc[df["Cabin"].str.find('D') >= 0, "Deck"] = 1
df.loc[df["Cabin"].str.find('E') >= 0, "Deck"] = 2
df.loc[df["Cabin"].str.find('F') >= 0, "Deck"] = 0
df.loc[df["Cabin"].str.find('G') >= 0, "Deck"] = 6


# just for today :)

# In[ ]:


df['Deck'] = df.Deck.fillna(6)


# In[ ]:


df['Deck'] = df['Deck'].astype(int)


# In[ ]:


sns.set(style="whitegrid")

g = sns.catplot(x="Deck", y="Survived", hue="Sex", data=df, height=6, kind="bar", palette="Set1")
g.despine(left=True)
g.set_ylabels("death probability")


# In[ ]:


pd.crosstab(df.Deck,df.Survived).apply(lambda r: r/r.sum(), axis=1)


# # Trash

# In[ ]:


df.head()


# Actually i dont know if their is a reason for the tickets, parch and SibSp
# 
# SibSp = (number of siblings on board) / (number of spouse on board)
# 
# ParCh = (number of Parents on board) / (number of Children on board)
# 
# so lets drop it for the moment

# In[ ]:


del df['Name']
del df['Ticket']
del df['Parch']
del df['SibSp']
del df['Cabin']


# In[ ]:


df.head()


# # Check the Values

# In[ ]:


sns.heatmap(df.corr(), annot = True)


# In[ ]:


df.dtypes


# # let the magic happens

# first split in x and y values

# In[ ]:


y=df["Survived"]
x=df.iloc[:,1:]


# i like to split the data in a training and a test dataset just to be sure my AIs work

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=4130)


# ## RandomForestClassifier
# 
# allways try the easier stuff first

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rdclass=RandomForestClassifier()
rdclass.fit(X_train,y_train)
ypred=rdclass.predict(X_test)

accRFC = accuracy_score(y_test,ypred)
print(accRFC)


# ## SVM
# 
# svm with default values

# In[ ]:


from sklearn import svm
clf = svm.SVR()
clf.fit(X_train,y_train)

ypred = clf.predict(X_test)
accSVM_D = accuracy_score(y_test, ypred.round())
print(accSVM_D)


# ok lets make some c and epsilon tests
# 
# gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma

# In[ ]:


val = []
euc = []

for c in range(1,50):
    for eps in range(0,10):
        clf = svm.SVR(C=c, epsilon=(eps/10))
        clf.fit(X_train,y_train)

        ypred = clf.predict(X_test)
        euc.append([c,(eps/10)])
        val.append([accuracy_score(y_test, ypred.round()),c,eps])

accSVM_A = max(val)
accSVM_A = accSVM_A[0]
print(accSVM_A)


# In[ ]:


max(val)


# In[ ]:


clf = svm.SVR(C=49, epsilon=(4/10))
clf.fit(X_train,y_train)

ypred = clf.predict(X_test)
accuracy_score(y_test, ypred.round())


# In[ ]:


ypred = pd.DataFrame(ypred)
ypred = pd.Series.round(ypred)
ypred = ypred.astype('int64')
accuracy_score(y_test, ypred.round())


# In[ ]:


import numpy as np
x_ntest = np.array(X_test)
y_ntest = np.array(y_test)
x_ntrain = np.array(X_train)
y_ntrain = np.array(y_train)


# In[ ]:


print(x_ntrain.shape, y_ntrain.shape)


# ## XG Boost

# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier()

# fit the model with the training data
model.fit(x_ntrain,y_ntrain)

# predict the target on the train dataset
predict_train = model.predict(x_ntrain)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_ntrain,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(x_ntest)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_ntest,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)


# In[ ]:


"""
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# make datasets
dtrain = xgb.DMatrix(x_ntrain, label=y_train)
dtest = xgb.DMatrix(x_ntest)

# set up
param = {'max_depth':8, 'eta':0.1, 'objective':'binary:hinge' }
num_round = 10

# fit the model with the training data
bst = xgb.train(param, dtrain, num_round)

# make prediction
preds = bst.predict(dtest)
preds = preds.astype('int64')

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test,preds)
print('accuracy_score on test dataset : ', accuracy_test)
"""


# ### Deep Learning (in progress****)

# In[ ]:


from tensorflow import keras
import tensorflow as tf
from keras.utils import to_categorical


# In[ ]:


modeltf = keras.Sequential([
    keras.layers.Reshape(target_shape=(1,), input_shape=(7,)),
    keras.layers.Dense(units= 7, activation='relu'),
    keras.layers.Dense(units= 14, activation='relu'),
    keras.layers.Dense(units= 7, activation='relu'),
    keras.layers.Dense(units= 1, activation='softmax')
])

modeltf.compile(optimizer='adam', loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])


# In[ ]:


modeltf.summary()


# # Test

# In[ ]:


df =  pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)

df.loc[df["Sex"] == "male","Sex"] = 1
df.loc[df["Sex"] == "female","Sex"] = 0

df['Embarked'] = df.Embarked.fillna('S')
df['Embarked'] = df['Embarked'].map( {'S': 2, 'Q': 1, 'C': 0} ).astype(int)

df.loc[df["Name"].str.find('Mr.') >= 0, "Type"] = 4
df.loc[df["Name"].str.find('Mrs.') >= 0, "Type"] = 0
df.loc[df["Name"].str.find('Miss.') >= 0, "Type"] = 1
df.loc[df["Name"].str.find('Master.') >= 0, "Type"] = 2
df.loc[df["Name"].str.find('Don.') >= 0, "Type"] = 3
df.loc[df["Name"].str.find('Rev.') >= 0, "Type"] = 3
df.loc[df["Name"].str.find('Dr.') >= 0, "Type"] = 3
df['Type'] = df.Type.fillna(3)
df['Type'] = df['Type'].astype(int)

df.loc[df["Fare"] < 10,"Fare"] = 5
df.loc[(df["Fare"] >= 10) & (df["Fare"] < 25),"Fare"] = 3
df.loc[(df["Fare"] >= 25) & (df["Fare"] < 50),"Fare"] = 4
df.loc[(df["Fare"] >= 50) & (df["Fare"] < 100),"Fare"] = 2
df.loc[(df["Fare"] >= 100), 'Fare'] = 0
df['Fare'] = df.Type.fillna(3)
df['Fare'] = df['Fare'].astype(int)

df['Age'] = df.Age.fillna(meanAge)
df['Age'] = df['Age'].astype('int64')

df.loc[df["Age"] < 15,"Age"] = 0
df.loc[(df["Age"] >= 15) & (df["Age"] < 30),"Age"] = 3
df.loc[(df["Age"] >= 30) & (df["Age"] < 45),"Age"] = 1
df.loc[(df["Age"] >= 45) & (df["Age"] < 60),"Age"] = 2
df.loc[(df["Age"] >= 60), 'Age'] = 4

df.loc[df["Cabin"].str.find('A') >= 0, "Deck"] = 5
df.loc[df["Cabin"].str.find('B') >= 0, "Deck"] = 3
df.loc[df["Cabin"].str.find('C') >= 0, "Deck"] = 4
df.loc[df["Cabin"].str.find('D') >= 0, "Deck"] = 1
df.loc[df["Cabin"].str.find('E') >= 0, "Deck"] = 2
df.loc[df["Cabin"].str.find('F') >= 0, "Deck"] = 0
df.loc[df["Cabin"].str.find('G') >= 0, "Deck"] = 6
df['Deck'] = df.Deck.fillna(6)
df['Deck'] = df['Deck'].astype(int)

del df['Name']
del df['Ticket']
del df['Parch']
del df['SibSp']
del df['Cabin']

df.head()


# In[ ]:


data = np.array(df)


# In[ ]:


dfinal = model.predict(data)   ##XGBoost


# In[ ]:


preds = dfinal
preds = preds.astype('int64')
preds


# switch back 'survived' value

# In[ ]:


preds[preds == 1] = 2
preds[preds == 0] = 1
preds[preds == 2] = 0
preds


# In[ ]:



my_submission = pd.DataFrame({'PassengerId': df.index, 'Survived': preds})


my_submission.to_csv('submission.csv', index=False)


# In[ ]:


my_submission.head(20)


# In[ ]:




