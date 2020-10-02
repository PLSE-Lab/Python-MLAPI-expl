#!/usr/bin/env python
# coding: utf-8

# # A Titanic Data Science Solutions
# 
# 
# ### Lets talk about Data Science... or is it Machine Learning?
# 
# Really, everyone just wants to attach whatever they are doing to these things. They're buzzwords. But most people really don't even understand what they mean or entail. Hopefully, by the end of this demo, you'll have a bit more of a sense for what it actually means.
#  
# At the end of the day, data science is the process of cleaning, visualizing, and analyzing data often with the goal of creating some sort of model. Machine Learning is the use of algorithms to use past data to predict future data.
# 
# ## The Steps of Data Science
# 
# The entire data science process can be roughly boiled down to 5 basic steps. There is a lot of nuance within each of these. In real use, you'll jump back and forth between the stages and repeat and iterate upon them, but it at the end of the day, this is what data scientists do.
# 
# 1. Get Data
# 2. Explore and Prepare Data
# 3. Train Model
# 4. Evaluate Model
# 5. Improve Model
# 
# We are going to take a closer look at each of these steps as we get into the demo. But, first, we need to understand our problem a bit better!
# 

# ## The Titanic and Kaggle
# 
# Kaggle is a crowd sourced online data science hub. There are thousands of datasets and sample workflows to work on and learn from all for free! Their "Hello World" dataset is looking at passenger manifests from the Titanic. We're given the below problem:
# 
# > Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.
# 
# Let's get the the data analyzing, then!
# 

# ## 1. Get Data
# 
# We've come to our first step. We want to set up our environment and get our data! After all, we can't do anything at all without data. Directly below, I've loaded up some readily available python packages for use in data analysis.
#  
# Pandas - is our primary data structure package. It allows us to use the data frame object (think an excel table) for all sorts of cleaning, visualizing, and modelling purposes.
# 
# Numpy - short for numeric python, numpy brings in many mathematic and scientific functions as well as it's versatile array object, which is the basis for many machine learning models
# 
# Seaborn - is a wrapper for the package below it, matplotlib and is used to easily visualize data within python
# 
# Sklearn - is short for Scikit-learn and is an absolutely massive package full of many machine learning algorithms.
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
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


# ### Acquire data
# The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together.
# 
# ### Training and Testing Datasets
# 
# A key concept in data science is the training and testing data sets. Effectively, we split our known data into portions to validate how well our models perform down the line. We keep them separate to ensure our model is "cheating" by using data from the test set when it is created. This helps to establish an important quality of a good model: generalizability. We want a model that isn't only good on data we currently have an answer for, but also data with unknown answers!

# In[ ]:


train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# In[ ]:


train.head(10)


# ### Let's look at what we have in here!
# 
# This is a list of all of the columns we have in our dataset. You can think of these just like you would think of a column in an excel table. You'll see we were given a list of column names as well as the number of non null records we have and type. So we can determine that we have a name for 891/891 passengers in this dataset; whereas, we only know the age of 714 of 891. The type tells us if our field is a string or a numeric.
# 
# **PassengerId** - A unique identifier per passenger
# 
# **Survived** - Quite the important one, this is a binary value that tells us if a passenger survived, or not. 1 codifies surviving and 0 codifies death
# 
# **Pclass** - What ticket class did the passenger hold? (First, second, third)
# 
# **Name** - A passenger's full name w/ title
#  
# **Sex** - Gender of the passenger
#  
# **Age** - How old was the passenger
#  
# **SibSp** - A count of the siblings + spouses that accompanied the passenger aboard
#  
# **Parch** - A count of the parents + children that accompanied the passenger aboard
#  
# **Ticket** - Passenger's ticket number
#  
# **Cabin** - Where the passenger stayed on the ship
#  
# **Embarked** - From which port the passenger boarded

# ## 2. Explore and Prepare Data
#  
# Congratulations, we've finished the first step! In the real world, this often takes much more time, but here, we can move on towards what is often the most fun part of data analysis: Exploratory Data Analysis (or EDA). This is where we try to get to know our data better by asking questions of it and trying to prove our own little theories.
#  
# What things might we be most interested in learning about passengers aboard the Titanic??
#  
# Who would you think is more likely to survive?

# In[ ]:


train.head()


# In[ ]:


#because our target variable "survived" consists of 1's and 0's we can see a few things fairly quickly!
print("Total Survivors: ",train['Survived'].sum())

percentSurvived = round(train['Survived'].mean(),2)
print("Percent Survive: ", percentSurvived)


# In[ ]:


#Did women have a better chance of survival?
cols = ['Sex','Survived']
genderSurvival = train[cols].groupby('Sex').mean().round(2)
genderSurvival


# In[ ]:


#we can show this same thing with a bar plot
ax = sns.barplot(x="Sex", y="Survived", data=train, ci=None)


# ### What about children/age?
#  
# Kids were supposed to do better than adults on the Titanic, right? But we don't have a field for kids, so we'll have to make our own! How should we do this?

# In[ ]:


train.tail(10)


# In[ ]:


#first, let's get an idea of the age breakdown of the passengers
ax = sns.kdeplot(train['Age'])


# In[ ]:


#What about children? They were supposed to do better, too, right?

#make a new column to classify a child from adult
train['ageClass'] = np.where(train['Age'] < 18, 'child','adult')

#lets make sure this worked. We can view our data using the "head" method
train.head(10)


# In[ ]:


#Okay, so how do kids do?
cols = ['ageClass','Survived']
ageSurvival = train[cols].groupby(['ageClass']).mean().round(2)
ageSurvival


# In[ ]:


#we can show this same thing with a bar plot
ax = sns.barplot(x="ageClass", y="Survived", data=train, ci=None)


# So it looks like kids **did** indeed have better rates of survival. Are you surprised by this?
# 
# While kids did have a leg up on adults, it looks like it wasn't as helpful as being female. What are some factors that may be confounding this?
# 

# In[ ]:


train.head(10)


# If you remember, we also have around 200 missing ages for our passengers. That could be throwing the rates (or it could not be, we don't know).
# 
# We made the cut at 18, but were really young children better off than young adults?

# In[ ]:


#update our ageClass column to include "young adults"
YA = (train['Age'] > 13) & (train['Age'] < 18)
train['ageClass'] = np.where(YA, 'young adult',train['ageClass'])

train.head(10)


# In[ ]:


#With young adults added, we can run the exact same code!
cols = ['ageClass','Survived']
ageSurvival = train[cols].groupby(['ageClass']).mean().round(2)

ax = sns.barplot(x="ageClass", y="Survived", data=train, ci=None)
ageSurvival


# So we learned something about our data by classifying a continuous "Age" variable into buckets to make visualization easier. You might ask why we couldn't just plot every age to see if there is a direct relation.

# In[ ]:


ax = sns.lineplot(x="Age", y="Survived", data=train)


# Unfortunately, this is a mess. Now, we could make an effort to smooth this and probably get something that resembles a wide U, but in many cases it can just be easier to categorize.
#  
# We've been looking at entirely one to one relationships right now, but we can also layer these facets on top of each other. Are girls more likely to survive than boys?

# In[ ]:


#We can layer multiple fields into our table
cols = ['Sex','ageClass','Survived']
ageSexSurvival = train[cols].groupby(['Sex','ageClass']).mean().round(2)

ax = sns.barplot(x="Sex", y="Survived", hue='ageClass', data=train, ci=None)
ageSexSurvival


# Looks like you **really** benefit from being a young male, but it actually hurts you to be a young child. It seems the sweet spot is in fact Rose and you definitely don't want to be Jack.

# ### Dealing with Missing values
# 
# So what about the 200 null values? What's happening to them?

# In[ ]:


train['ageClass'] = np.where(train['Age'].isna(), 'missing',train['ageClass'])

#With young adults added, we can run the exact same code!
cols = ['ageClass','Survived']
ageSurvival = train[cols].groupby(['ageClass']).mean().round(2)

ax = sns.barplot(x="ageClass", y="Survived", data=train, ci=None)
ageSurvival


# We can see the missing values do much worse than any of our other fields. Looking at our data, what are some ways we could handle these values?

# In[ ]:


filt = train['ageClass'] == 'missing'
train[filt].head(10)


# Well, there are a couple of things we can do!
#  
# - If we have evidence our NA values are invalid, we may want to get rid of the record
# - If the rest of the information could be useful, we could try to impute the values
# - We **cannot** just get rid of NA values because they make our analysis nicer
#  
# How should we handle these?

# In[ ]:


#Well, there are some features we could use to help deduce their age
#We could just take the overall average
avgAge = round(train['Age'].mean(),0)
print('Average Age: ', round(avgAge,0))


# That's one way to do it! But we can be a bit more clever. I'm particularly interested in the titles that exist in the 'Name' field as well as the 'Sibsp' count.
# 
# If they have more than 1 Sibsp on board, they are likely a child.
#  
# If they have the title Master (for boys) or Miss (for girls) they are also likely young or a child.
#  
# If they don't fit that criteria, we can just impute the average of all ages.

# In[ ]:


#Master is a title given to young boys back then, think "Master Wayne"
condition = (train['Name'].str.contains(pat = "Master")) & (train['Age'].isna())
train['ageClass'] = np.where(condition, 'child',train['ageClass'])

#Miss implies an unmarried girl. Probably a young adult or child, we'll just put YA for now
condition = (train['Name'].str.contains(pat = "Miss")) & (train['Age'].isna())
train['ageClass'] = np.where(condition, 'young adult',train['ageClass'])

#If they have more than one Sibling/Spouse, they are definitely travelling with a sibling. We are going to guess that those with siblings are travelling as a family and are kids
condition = (train['SibSp'] > 1) & (train['Age'].isna())
train['ageClass'] = np.where(condition, 'child',train['ageClass'])

#Assigning everyone else the average age of 30 making them adults
condition = (train['ageClass'] == 'missing')
train['ageClass'] = np.where(condition, 'adult',train['ageClass'])


# In[ ]:


#We can layer multiple fields into our table
cols = ['Sex','ageClass','Survived']
ageSexSurvival = train[cols].groupby(['Sex','ageClass']).agg({'Survived':'mean','ageClass':'count'}).round(2)

ax = sns.barplot(x="Sex", y="Survived", hue='ageClass', data=train, ci=None)
ageSexSurvival


# Just like that, no more missing values!

# Lets do a quick look at some of the other features

# In[ ]:


#Looks like the more prestigious tickets did better than than the others
cols = ['Pclass','Survived']
classSurvival = train[cols].groupby(['Pclass']).agg({'Survived':'mean'}).round(2)

ax = sns.barplot(x="Pclass", y="Survived", data=train, ci=None)
classSurvival


# In[ ]:


#And first class women did particularly well
cols = ['Pclass','Sex','Survived']
classSurvival = train[cols].groupby(['Sex','Pclass']).agg({'Survived':'mean'}).round(2)

ax = sns.barplot(x="Pclass", y="Survived", hue='Sex', data=train, ci=None)
classSurvival


# In[ ]:


#And first class women did particularly well
cols = ['Pclass','Sex','ageClass','Survived']
classSurvival = train[cols].groupby(['Sex','ageClass','Pclass']).agg({'Survived':'mean','ageClass':'count'}).round(2)

ax = sns.catplot(x="ageClass", y="Survived", hue='Sex', col='Pclass', data=train, kind='bar', ci=None)
classSurvival


# Quite a lot going on here, but we can see some stand out observations mixed in with a few low n ones.

# ## 3 Train a Model (plus a little evaluation)
# We spent a whole lot of time in the EDA phase, because that is where most of our work is done! Training a model, really, doesn't take too much effort. The far more difficult thing is understanding your data and enhancing it with new features to benefit your model. If you can do that, you will come away with much more value.

# ### Converting a categorical feature
# 
# Many of our values are presented to us as strings. For instance, gender is set to "Male" and "Female", but our algorithms prefer things to be numbers. To prepare our data to be trained on, we need to encode strings into numbers. 
# 
# How could we go about doing this?

# In[ ]:


train.head(10)


# In[ ]:


#we could just assign binary or integer values to our fields.
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()


# In[ ]:


#We could also do something called "One-hot encoding"
cols = ['PassengerId','ageClass']
train2 = train[cols]
train2 = pd.get_dummies(train2)
train2.head()


# Both of these methods convert our values into numbers. In many cases, one hot encoding is preferred simply because it can avoid the algorithm mistaking categorical fields for continuous ones. 
# 
# However, for this demo, we will simply map integers for simplicity sake
#  
# Now, quick detour to shore up the rest of our fields' missing values and encoding...

# In[ ]:


#we have 2 missing values in Embarked, I'm going to use the Fare to impute them
emCost = train.groupby(['Embarked']).agg({'Fare':'mean'})
emCost


# In[ ]:


#Hmm, I wonder where these passengers embarked from?
train[train['Embarked'].isna()]


# In[ ]:


train['Embarked'] = np.where(train['Embarked'].isna(),'C',train['Embarked'])

#I'm going to roughly order these in terms of average fare cost
train['Embarked'] = train['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
train.head()


# In[ ]:


#I'm going to roughly order age classification as well
train['ageClass'] = train['ageClass'].map( {'child': 0, 'young adult': 1, 'adult': 2} ).astype(int)
train.head()


# That encodes everything, now I need to drop some of the fields the algorithms won't be able to use. What are these?
# 
# How could we get more information out of them?

# In[ ]:


#Here are the columns I want to keep
cols = ['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked','ageClass']
train = train[cols]
train.head()


# In[ ]:


#I'm also going to 'feature engineer' one more field that identifies people who are alone and family size
train['companions'] = train.loc[:,'SibSp'] + train.loc[:,'Parch']
train['loner'] = np.where(train['companions'] < 1, 1,0)
train.head()


# Finally, we need to check for any remaining null values

# In[ ]:


train.info()


# In[ ]:


#I'm replicating all the transformations we did to our training dataset to our test dataset
test['ageClass'] = np.where(test['Age'] < 18, 'child','adult')
YA = (test['Age'] > 13) & (test['Age'] < 18)
test['ageClass'] = np.where(YA, 'young adult',test['ageClass'])
test['ageClass'] = np.where(test['Age'].isna(), 'missing',test['ageClass'])

condition = (test['Name'].str.contains(pat = "Master")) & (test['Age'].isna())
test['ageClass'] = np.where(condition, 'child',test['ageClass'])

condition = (test['Name'].str.contains(pat = "Miss")) & (test['Age'].isna())
test['ageClass'] = np.where(condition, 'young adult',test['ageClass'])

condition = (test['SibSp'] > 1) & (test['Age'].isna())
test['ageClass'] = np.where(condition, 'child',test['ageClass'])

condition = (test['ageClass'] == 'missing')
test['ageClass'] = np.where(condition, 'adult',test['ageClass'])

test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test['Embarked'] = np.where(test['Embarked'].isna(),'C',test['Embarked'])
test['Embarked'] = test['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

test['ageClass'] = test['ageClass'].map( {'child': 0, 'young adult': 1, 'adult': 2} ).astype(int)

pID = test["PassengerId"].copy(deep=True)

cols = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','ageClass']
test = test[cols]

test['companions'] = test.loc[:,'SibSp'] + test.loc[:,'Parch']
test['loner'] = np.where(test['companions'] < 1, 1,0)

test.head()


# In[ ]:


test.info()


# In[ ]:


#we have one null value in our fare, lets impute that with the mean
meanFare = test['Fare'].mean()
test['Fare'] = np.where(test['Fare'].isna(),meanFare,test['Fare'])
test.info()


# ### Model time!
#  
# Time to get to modelling! We are going to use the package scikit-learn for our machine learning. It contains dozens of algorithms to choose between, but we are just going to run a handful! Typically, any data science problem includes a lot of testing many different solutions out. Since we don't know which model will be best, we make a lot and find out!
#  
# Now, there are 2 basic types of machine learning problems. Classification and Regression. We've been dealing with a classification problem. Did someone live or die? Many real world problems are regression oriented. How much money will I make next month?
#  
# We've also split our data into a training and a test set. This is called supervised machine learning. With this in mind we could grab all sorts algorithms to test out!
#  
# - Logistic Regression
# - KNN or k-Nearest Neighbors
# - Support Vector Machines
# - Naive Bayes classifier
# - Decision Tree
# - Random Forrest
# - Perceptron
# - Artificial neural network
# - RVM or Relevance Vector Machine

# We need to split our features into two different groups. Independent variables which are features that shouldn't be affected by the other values, and our dependent or target variable what we are trying to predict!

# In[ ]:


#everything besides "Survived" is an independent variable, so we are going to set them asside and denote them with an X
X_train = train.drop("Survived", axis=1)

#Survived is our dependent or target variable, we will pull that aside as well and denot it with a Y
Y_train = train["Survived"]

#We already don't have a "Survived" column in our test data set, so we are just going to grab a copy
X_test  = test.copy(deep=True)

#The two training sets hsould retain the same length
#and the two X sets should retain the same width
columns = X_train.columns
X_train.shape, Y_train.shape, X_test.shape


# Before we get started, it's important to have a baseline. Based on our analysis, it looks like the single biggest cut we could make in predicting survival is on the basis of gender. This accuracy will be the number to beat!

# In[ ]:


from sklearn.metrics import accuracy_score

naive = X_train['Sex'].copy(deep=True)
acc_naive = round(accuracy_score(Y_train, naive)*100,2)
acc_naive


# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).
#  
# Note the confidence score generated by the model based on our training dataset.

# In[ ]:


# Logistic Regression

#define the model
logreg = LogisticRegression()

#fit he model to training data
logreg.fit(X_train, Y_train)

#predict test data
logPred = logreg.predict(X_test)


acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# One thing you **always** have to be aware of in machine learning is the propensity to over fit your data. You can imagine overfitting like memorizing the answers to a math test without actually learning how to solve new problems. When you are given a new test, you can write down what you remember, but will often be completely wrong. Machines do this too. The accuracy score above is the accuracy on our *training* data set. It's what our algorithm studied. 
#  
# But we don't know the answers in the test dataset, so how can we get an idea how it will do when applied there? Well, we can try something call CV Folds. Basically, it breaks apart the training data set into chunks. It then uses those chunks as it's own mini training and test sets. So when we make a prediction on the X_train data, the algorithm has not yet seen it. Usually, this brings our numbers back down to earth.

# In[ ]:


cvAcc_log = round(cross_val_score(logreg, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_log)
print('CV Acc:    ',cvAcc_log)


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
#  
# What can we conclude from these?

# In[ ]:


coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# - Inversely as Pclass increases, probability of Survived=1 decreases
# - We see an inverse releationship with our ageClass feature, the older classification the worse off you are.
# - The number of parents/children and fare of your ticket didn't really matter

# Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of **two categories**, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).
#  
# Note that the model generates a confidence score which is higher than Logistics Regression model.

# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
svmPred = svc.predict(X_test)


acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

cvAcc_svc = round(cross_val_score(svc, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_svc)
print('CV Acc:    ',cvAcc_svc)


# In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
# 
# KNN confidence score is better than Logistics Regression but worse than SVM.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)


knnPred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


cvAcc_knn = round(cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_knn)
print('CV Acc:    ',cvAcc_knn)


# In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).
# 
# The model generated confidence score is the lowest among the models evaluated so far.

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)


gauPred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

cvAcc_gaussian = round(cross_val_score(gaussian, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_gaussian)
print('CV Acc:    ',cvAcc_gaussian)


# The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. Reference [Wikipedia](https://en.wikipedia.org/wiki/Perceptron).

# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)

perPred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

cvAcc_perceptron = round(cross_val_score(perceptron, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_perceptron)
print('CV Acc:    ',cvAcc_perceptron)


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

svcPred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

cvAcc_linear_svc = round(cross_val_score(linear_svc, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_linear_svc)
print('CV Acc:    ',cvAcc_linear_svc)


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)

sgdPred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

cvAcc_sgd = round(cross_val_score(sgd, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_sgd)
print('CV Acc:    ',cvAcc_sgd)


# This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).
# 
# The model confidence score is the highest among models evaluated so far.

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

trePred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

cvAcc_decision_tree = round(cross_val_score(decision_tree, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_decision_tree)
print('CV Acc:    ',cvAcc_decision_tree)


# The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).
# 
# The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, max_depth=4)
random_forest.fit(X_train, Y_train)

ranPred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

cvAcc_random_forest = round(cross_val_score(random_forest, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)
print('Train Acc: ',acc_random_forest)
print('CV Acc:    ',cvAcc_random_forest)


# In[ ]:


#we can get feature importance out of random forests too! What are some thoughts we can gain from this?
importances = list(random_forest.feature_importances_)
feature_importances = pd.DataFrame([(feature, round(importance, 2)) for feature, importance in zip(columns, importances)])
feature_importances.columns = ['feature','importance']
feature_importances.sort_values('importance',ascending=False)


# Finally, it is common practice to combined different models together. Hopefully, they will keep each others strengths and throw aside weaknesses. We do this through a process called ensembling. Now, there are many ways to do this. Can you think of any?

# - Model voting
# - Weighted average output
# - Machine learning?
#  
# Lots of different options. We are going to look at simple model voting here.

# In[ ]:


#scikit-learn has a voting object that acts just like any of our other models
voter = VotingClassifier(estimators=[('random_forest', random_forest), ('sgd', sgd), ('linear_svc', linear_svc),
                                    ('logreg',logreg),('svc',svc),('knn',knn),('gaussian',gaussian)], voting='hard')

scores = cross_val_score(voter, X_train, Y_train, cv=5, scoring='accuracy')

cvAcc_ensemble = round(cross_val_score(voter, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

voter.fit(X_train, Y_train)
ensPred = voter.predict(X_test)
acc_ensemble = round(voter.score(X_train, Y_train) * 100, 2)
print('Train Accuracy: ', acc_ensemble)
print("CV Accuracy:    %0.2f (+/- %0.2f)" % (round(scores.mean()*100,2), scores.std()))


# ## 4 Model evaluation
# 
# We can now rank our evaluation of all the models to choose the best one for our problem. We can see how the differnt models tack up against each other and our naive model

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree','Ensemble','Naive'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree,acc_ensemble,acc_naive],
    'ScoreCV': [cvAcc_svc, cvAcc_knn, cvAcc_log, 
                  cvAcc_random_forest, cvAcc_gaussian, cvAcc_perceptron, 
                  cvAcc_sgd, cvAcc_linear_svc, cvAcc_decision_tree,cvAcc_ensemble,acc_naive]})
models.sort_values(by='ScoreCV', ascending=False)


# ## 5 Improving the model and Results
# 
# So, all that work, and we only have a realized improvement of ~4% over our naive model. Data Science is hard work on hard problems! You are never guaranteed success, but fortunately, sometimes small gains can translate to big value! 4% is an additional 32 people correctly classified! If we want to do better, we could iterate. In my role, this usually comes down to determining what is good enough. Perfect is the enemy of good enough. Sometimes it is more time effective to settle with good model that you can implement, rather than incrementally improving forever.
# 
# That being said, it often takes many iterations of analysis to reach that stage. What did we leave on the table here? How could we go back an improve our analysis and eventual predictions?

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": pID,
        "Survived": ranPred
    })
#submission.to_csv('../output/submission.csv', index=False)

