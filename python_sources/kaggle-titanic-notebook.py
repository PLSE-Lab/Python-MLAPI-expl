#!/usr/bin/env python
# coding: utf-8

# # My Titanic Notebook / First Kaggle Project
# 
# This is my first Kaggle project, and my goal here is simply to learn the different phases of problem-solving when tackling a Data Science/ML question. Let's go through this step by step (this is openly inspired by the Notebooks available in the [Titanic Kaggle Tutorial](https://www.kaggle.com/c/titanic/overview/tutorials)):
# 
# 
# ## Context & Question
# 
# > On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 
# 
# That's a **32% survival rate**.
# 
# > One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. 
# 
# This means we could somehow **design an algorithm to predict survival based on the passenger's features** (gender, age, class, etc.). Let's see...
# 
# 
# ## What is the available data? What does it look like?
# 
# First let's load all of our data analysis/ML libraries just in case we need anything in there (we will):
# 

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# And second, the dataset provided by Kaggle for this problem:

# In[ ]:


titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic =  titanic_train.append( titanic_test , ignore_index = True, sort=True )


# So... Now what? **Let's have a quick look at the dataset: What are the available features? Do we have missing data? Which features are numerical / categorical?**
# (also described on the [Kaggle Titanic page](https://www.kaggle.com/c/titanic/data))
# 
# - Survived: Survived (1) or died (0)
# - Pclass: Passenger's class
# - Name: Passenger's name
# - Sex: Passenger's sex
# - Age: Passenger's age
# - SibSp: Number of siblings/spouses aboard
# - Parch: Number of parents/children aboard
# - Ticket: Ticket number
# - Fare: Fare
# - Cabin: Cabin
# - Embarked: Port of embarkation

# In[ ]:


titanic.describe()


# In[ ]:


titanic.count()


# In[ ]:


titanic.head(n=10)


# **Categorical features:** Survived (1 or 0 - what we're trying to predict), Sex (male / female), Embarked (S,C,Q), Pclass (1,2,3)
# 
# **Numerical features:** Age, SibSp/Parch, Fare
# 
# We can now tell that some columns have some (or even a lot of) missing values (Age, Fare, Cabin, Embarked) in both datasets. 
# We also see that in our training sample, **the average survival rate is 38% (vs 32% for the 2224 passengers)**.
# 
# ## Completing/Adding some features:
# We may want to create a few additional features like **total family size**, **age bucket** (child/adult/senior - after completing the missing ages), and **fare bucket** (although it might repeat with Pclass). I've seen a lot of people extracting the Title from the name feature: let's do that as well and see if it can useful. 
# 
# Once we have these new features, we'll be able to chart/visualize our data much better and have a first idea of correlations.

# In[ ]:




titanic_test.Fare = titanic_test.Fare.fillna(titanic.Fare.mean())

titanic_train.Embarked = titanic_train.Embarked.fillna(titanic.Embarked.value_counts().index[0])
titanic_test.Embarked = titanic_test.Embarked.fillna(titanic.Embarked.value_counts().index[0])

titanic_train['TotalFam'] = titanic_train.apply(lambda row: row.SibSp + row.Parch + 1, axis = 1) 
titanic_test['TotalFam'] = titanic_test.apply(lambda row: row.SibSp + row.Parch + 1, axis = 1) 
titanic ['TotalFam'] = titanic.apply(lambda row: row.SibSp + row.Parch + 1, axis = 1) 

titanic_train['Family'] = ['1.Loner' if x ==1 else '2.Small' if x <=3 else "3.Big" for x in titanic_train['TotalFam']] 
titanic_test['Family'] = ['1.Loner' if x ==1 else '2.Small' if x <=3 else "3.Big" for x in titanic_test['TotalFam']]

titanic_train['FareCat'] = ['4.Cheap'if x <=8 else '3.Regular' if x <=33 else '2.Expensive' if x <=150 else "1.Extra" for x in titanic_train['Fare']]
titanic_test['FareCat'] = ['4.Cheap'if x <=8 else '3.Regular' if x <=33 else '2.Expensive' if x <=150 else "1.Extra" for x in titanic_test['Fare']]


titanic_train['Title'] = titanic_train[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
titanic_test['Title'] = titanic_test[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
titanic['Title'] = titanic[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

titanic_train['Title'] = titanic_train['Title'].replace(['Lady', 'the Countess', 'Sir', 'Jonkheer', 'Master'], 'Royal')
titanic_train['Title'] = titanic_train['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Rank')
titanic_train['Title'] = titanic_train['Title'].replace('Mlle', 'Miss')
titanic_train['Title'] = titanic_train['Title'].replace('Ms', 'Miss')
titanic_train['Title'] = titanic_train['Title'].replace('Mme', 'Mrs')
titanic_train['Title'] = titanic_train['Title'].replace('Dona', 'Mrs')
titanic_train['Title'] = titanic_train['Title'].replace('Don', 'Mr')

titanic_test['Title'] = titanic_test['Title'].replace(['Lady', 'the Countess', 'Sir', 'Jonkheer', 'Master'], 'Royal')
titanic_test['Title'] = titanic_test['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Rank')
titanic_test['Title'] = titanic_test['Title'].replace('Mlle', 'Miss')
titanic_test['Title'] = titanic_test['Title'].replace('Ms', 'Miss')
titanic_test['Title'] = titanic_test['Title'].replace('Mme', 'Mrs')
titanic_test['Title'] = titanic_test['Title'].replace('Dona', 'Mrs')
titanic_test['Title'] = titanic_test['Title'].replace('Don', 'Mr')

titanic['Title'] = titanic['Title'].replace(['Lady', 'the Countess', 'Sir', 'Jonkheer', 'Master'], 'Royal')
titanic['Title'] = titanic['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Rank')
titanic['Title'] = titanic['Title'].replace('Mlle', 'Miss')
titanic['Title'] = titanic['Title'].replace('Ms', 'Miss')
titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')
titanic['Title'] = titanic['Title'].replace('Dona', 'Mrs')
titanic['Title'] = titanic['Title'].replace('Don', 'Mr')

titanic_train.Cabin = titanic_train.Cabin.fillna("UO")
titanic_test.Cabin = titanic_test.Cabin.fillna("UO")

titanic_train['Deck'] = [x[0] for x in titanic_train['Cabin']]
titanic_test['Deck'] = [x[0] for x in titanic_test['Cabin']]

titanic_train.Deck = titanic_train.Deck.fillna("U")
titanic_test.Deck = titanic_test.Deck.fillna("U")

# group by Sex, Pclass, and Title 
grouped = titanic.groupby(['Sex','Pclass', 'Title'])  

titanic_train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
titanic_test.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

titanic_train['AgeCat'] = ['4.Kid'if x <=10 else '3.Young' if x <=18 else '2.Adult' if x<=50 else '1.Senior' for x in titanic_train['Age']]
titanic_test['AgeCat'] = ['4.Kid'if x <=10 else '3.Young' if x <=18 else '2.Adult' if x<=50 else '1.Senior' for x in titanic_test['Age']]


# In[ ]:


titanic_train.head(n=20)


# **What is the distribution of the data in our Train dataset?**
# Using Matplotlib for my histograms (see this [tutorial](https://matplotlib.org/gallery/statistics/hist.html#sphx-glr-gallery-statistics-hist-py)) and Seaborn for the Swarm/Box plots (see this [tutorial](http://seaborn.pydata.org/tutorial/categorical.html#distributions-of-observations-within-categories))

# In[ ]:


facet, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
sns.despine(left=True)

sns.barplot(x="Sex", y="Survived",  data=titanic_train,  ax=axes[0,0])
sns.barplot(x="AgeCat", y="Survived", data=titanic_train,  ax=axes[0,1])
sns.barplot(x="FareCat", y="Survived",  data=titanic_train,  ax=axes[0,2])
sns.barplot(x="Pclass", y="Survived",  data=titanic_train,  ax=axes[1,0])
sns.barplot(x="Embarked", y="Survived",  data=titanic_train,  ax=axes[1,1])
sns.barplot(x="Family", y="Survived",  data=titanic_train,  ax=axes[1,2])
sns.barplot(x="Title", y="Survived",  data=titanic_train,  ax=axes[2,0])


# In[ ]:


sns.catplot(x="Family", y="Fare", hue="Survived", kind="swarm", col="AgeCat", data= titanic_train);


# In[ ]:


sns.catplot(x="Pclass", y="Age", hue="Survived", kind="swarm", col="Sex", data= titanic_train);


# In[ ]:


sns.catplot(x="Title", y="Age", hue="Survived", kind="swarm", col="Sex", data= titanic_train);


# **Correlation**
# At first glance, it seems there's a correlation between surviving the Titanic, and being a woman / being a kid or young / belonging to 1st class / paying extra for the ticket / belonging to a small family / as well as possibly embarking from C.
# 
# There might be slight correlations as well between the features themselves (paying extra & belonging to 1st class sounds possibly like a repeat for instance, and what about Title? Isn't title a combination of Age, Pclass, and Sex in a way?)
# 
# Let's create new versions of the datasets with numerical values only!

# In[ ]:


train = titanic_train
test = titanic_test

train['Family'] = [(x-titanic.TotalFam.mean())/titanic.TotalFam.std() for x in train['TotalFam']] 
test['Family'] = [(x-titanic.TotalFam.mean())/titanic.TotalFam.std() for x in test['TotalFam']]

train['AgeNorm'] = [(x-titanic.Age.mean())/titanic.Age.std() for x in train['Age']] 
test['AgeNorm'] = [(x-titanic.Age.mean())/titanic.Age.std() for x in test['Age']]

train['FareNorm'] = [(x-titanic.Fare.mean())/titanic.Fare.std() for x in train['Fare']] 
test['FareNorm'] = [(x-titanic.Fare.mean())/titanic.Fare.std() for x in test['Fare']]

title_mapping = {'Royal':5, 'Rank': 4, 'Mrs': 3, 'Miss': 2, 'Mr': 1} 
sex_mapping = {'female': 1, 'male': 0}
emb_mapping = {'C': 3, 'S': 2, 'Q': 1}
deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8, "T":8}
age_mapping = {"4.Kid": 1, "3.Young": 2, "2.Adult": 3, "1.Senior": 4}

train['Title'] = train['Title'].map(title_mapping).astype(int)
test['Title'] = test['Title'].map(title_mapping).astype(int)

train['Embarked'] = train['Embarked'].map(emb_mapping).astype(int)
test['Embarked'] = test['Embarked'].map(emb_mapping).astype(int)

train['Sex'] = train['Sex'].map(sex_mapping).astype(int)
test['Sex'] = test['Sex'].map(sex_mapping).astype(int)

train['Deck'] = train['Deck'].map(deck_mapping).astype(int)
test['Deck'] = test['Deck'].map(deck_mapping).astype(int)

train['AgeCat'] = train['AgeCat'].map(age_mapping).astype(int)
test['AgeCat'] = test['AgeCat'].map(age_mapping).astype(int)

train.head(n=10)


# In[ ]:


train['Age_Class']= train.apply(lambda row: row.AgeCat * row.Pclass, axis = 1)
test['Age_Class']= test.apply(lambda row: row.AgeCat * row.Pclass, axis = 1)


# In[ ]:


train = train.drop(['Name', 'PassengerId','Age', 'FareCat','AgeCat' ,'SibSp', 'Parch','Ticket','Fare','Cabin','TotalFam'], axis=1)
test = test.drop(['Name', 'Age', 'AgeCat', 'FareCat', 'SibSp', 'Parch','Ticket','Fare','Cabin','TotalFam'], axis=1)
train.head(n=10)


# In[ ]:


corr = train.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[ ]:


test.head(n=10)


# ## Creating a Model and Making Predictions
# 
# Let's prep our data: Training set, Cross Validation set and our Test set, and train a Logistic Regression model, as well as a few other ones to compare...
# 

# In[ ]:


X = train.drop("Survived", axis=1)
Y = train["Survived"]
Test_X  = test.drop("PassengerId", axis=1).copy()
Train_X , Valid_X , Train_Y , Valid_Y = train_test_split( X, Y , train_size = .7 )
Train_X.shape, Valid_X.shape, Train_Y.shape, Valid_Y.shape, Test_X.shape


# In[ ]:


# Logistic Regression

modelLR = LogisticRegression()
modelLR.fit(Train_X, Train_Y)
Y_predValid = modelLR.predict(Valid_X)
AccModel = round(modelLR.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

model = modelLR
maxAcc = AccValid


# In[ ]:


# K Nearest Neighbors

modelKNN = KNeighborsClassifier(n_neighbors = 3)
modelKNN.fit(Train_X, Train_Y)
Y_predValid = modelKNN.predict(Valid_X)
AccModel = round(modelLR.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

if (AccValid > maxAcc): 
    maxAcc = AccValid
    model = modelKNN


# In[ ]:


# SVC Check Parameters
A = [0.05, 0.1, 0.3, 0.7, 1, 1.5, 2, 3]
B = [0.05, 0.1, 0.3, 0.7, 1, 1.5, 2, 3]
AccM = np.zeros((len(A), len(B)))
acc = np.zeros((len(A), len(B)))
Cideal = 0
Gamideal = 0
for i in range(len(A)):
    for j in range(len(B)):
               modelSVC = SVC(C = A[i],gamma = B[j])
               modelSVC.fit(Train_X, Train_Y)
               Y_predValid = modelSVC.predict(Valid_X)
               AccM[i,j] = round(modelSVC.score(Train_X, Train_Y) * 100, 2)
               acc[i,j] = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
               if (acc[i,j] >= np.amax(acc)): 
                    Cideal = A[i]
                    Gamideal = B[j]
        
print("Model: ", AccM, "  and Val: ", acc)
print("C ideal: ", Cideal, " and Gamma ideal: ", Gamideal)
        


# In[ ]:


# SVC

modelSVC = SVC(C = Cideal, gamma = Gamideal)
modelSVC.fit(Train_X, Train_Y)
Y_predValid = modelSVC.predict(Valid_X)
AccModel = round(modelSVC.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

if (AccValid > maxAcc): 
    maxAcc = AccValid
    model = modelSVC


# In[ ]:


# Perceptron

modelP = Perceptron()
modelP.fit(Train_X, Train_Y)
Y_predValid = modelP.predict(Valid_X)
AccModel = round(modelP.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

if (AccValid > maxAcc): 
    maxAcc = AccValid
    model = modelP


# In[ ]:


# Decision Tree

modelDT = DecisionTreeClassifier()
modelDT.fit(Train_X, Train_Y)
Y_predValid = modelDT.predict(Valid_X)
AccModel = round(modelDT.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

if (AccValid > maxAcc): 
    maxAcc = AccValid
    model = modelDT


# In[ ]:


# Random Forest

modelRF = RandomForestClassifier()
modelRF.fit(Train_X, Train_Y)
Y_predValid = modelRF.predict(Valid_X)
AccModel = round(modelRF.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

if (AccValid > maxAcc): 
    maxAcc = AccValid
    model = modelRF


# In[ ]:


# SGD Classifier

modelSGD = SGDClassifier()
modelSGD.fit(Train_X, Train_Y)
Y_predValid = modelSGD.predict(Valid_X)
AccModel = round(modelSGD.score(Train_X, Train_Y) * 100, 2)
AccValid = round((sum(Valid_Y == Y_predValid)/Valid_Y.count())*100,2)
print('Accuracy on Training is', AccModel )
print('Accuracy on Cross Val is', AccValid )

if (AccValid > maxAcc): 
    maxAcc = AccValid
    model = modelKNN


# In[ ]:


print(model)
Test_Y = model.predict( Test_X )
passenger_id = test.PassengerId
Final = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': Test_Y } )
Final.shape
Final.head(n=20)


# In[ ]:


Final.to_csv( 'titanic_pred.csv' , index = False )

