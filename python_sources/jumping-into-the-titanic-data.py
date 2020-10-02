#!/usr/bin/env python
# coding: utf-8

# # Jumping into the Titanic Data
# 
# ---
# 
# Josh takes a stab at the Titanic data

# ## Import data and initialize primary data frames

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import seaborn as sb
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv('../input/train.csv')
testDF = pd.read_csv('../input/test.csv')
allData = [trainDF, testDF]
allDF = pd.concat([trainDF, testDF])
totalPassengers = 2224


# # Preview Data
# 
# ---
# 
# 

# In[ ]:


trainDF.head()


# # Top-Level Metadata
# 
# ---
# 
# 

# In[ ]:


trainDF.info()
print('+'+'-'*40+'+')
testDF.info()
print('+'+'-'*40+'+')
print('+'+'-'*40+'+')
print('Training data comprises %4.2f %% of all data'% (len(trainDF)/totalPassengers*100))
print('Testing data comprises %4.2f %%'%(len(testDF)/totalPassengers*100))
print('+'+'-'*40+'+')
print('+'+'-'*40+'+')


# This tells us that we probably want to complete the **Age** feature for both our training and testing data, as well as **Embarked**. We will want to complete **Fare** for the testing data as well. As we will see shortly, we may also want to correct some **Fare** data for the training and testing data.

# # Details
# 
# ---
# 
# 

# In[ ]:


# Get details about numerical features
trainDF.describe()


# In[ ]:


# Get details about categorical features
trainDF.describe(include=['O'])


# ## Numerical Feature Details
# We can see that there are at least one incomplete feature (**Age**). We can also see that **Fare** has a minimum value of $0. We will probably want to correct these later. Additionally, Embarked is missing two entries
# 
# ## Categorical (Ordinal) Feature Details
# We can see that there is some information to be gained by observing the features here. For instance, there are only 681 unique ticket numbers, telling us that many passengers have the same ticket. There are also *many* duplicated **Cabin** values, telling us that many people shared a cabin. We may be able to use this. There are two different **Sex** values, which is probably not surprising. **Embarked** has three values, denoting the boarding location.
# 
# Look at all that. Some of it is clearly useless for analysis. For instance, **Cabin** is null very often, and **Ticket** takes on several different formats. Others, such as **Age** are probably going to be useful.
# 
# Notice that the **Fare** and **Age** fields both have a minimum of 0. This may be problematic. We do not want to overlook data due to an oversight in record-keeping. Below we will do some more exploration.
# 
# ### Zero-Fare Analysis

# In[ ]:


trainDF[trainDF.Fare==0].describe(include=['O'])


# In[ ]:


trainDF[trainDF.Fare==0].describe()


# We now see that all of the zero-fare passengers are **males who embarked from S**, although they seem to come from first, second, and third class. 
# 
# ### Null-Age Analysis

# In[ ]:


trainDF[trainDF.Age.isnull()].describe()


# In[ ]:


trainDF[trainDF.Age.isnull()].describe(include=['O'])


# Nothing is immediately jumping out at me as being strongly correlated, besides the fact that 70% of the null-age passengers are males, unsurprising considering 65% of all passengers in our training data are men anyway. We'll have to do some further analysis later.
# 
# ##Null-Embarked Analysis

# In[ ]:


trainDF[trainDF.Embarked.isnull()].describe()


# In[ ]:


trainDF[trainDF.Embarked.isnull()].describe(include=['O'])


# In[ ]:


trainDF[trainDF.Embarked.isnull()].Name


# Based on the information above, we can see that the two passengers for whom we have no Embarked data are **women in first class**. It seems they shared a cabin, but they have different names, so I'm not sure we can gain any information from that fact.

# # Analysis
# 
# ----------
# 
# We want to learn some more about our data. In particular, we are concerned about the following data:
# 
# - **Age**
# - **Sex**
# - **Embarked**
# - **Pclass**
# - **Fare**
# - **SibSp**
# - **Parch**
# 
# You'll notice that we don't look too closely at the following features:
# 
# - **Passenger ID** because it is unique and will likely not have anything to do with the results
# - **Ticket** because it seems that there are many duplicates and the format is not standardized
# - **Cabin** because this field is *very* incomplete
# 
# The passenger's **Name** is sort of a special case. We'll try to extract some useful data from their titles later.
# 
# We'll start by defining a few functions that we'll use.

# In[ ]:


# Function for sorting on survival. 
# Note this is optimized for discrete variables with only a few possible values
def sortByFeature(df, groupedFeature, sortedFeature='Survived'):
    return df[[groupedFeature, sortedFeature]].groupby([groupedFeature], as_index=False).mean().sort_values(by=sortedFeature, ascending=False)
    
def survivalHistogram(df, x, b=20, size=2.2):
    from seaborn import FacetGrid as grid
    from matplotlib import pyplot as plot
    g = grid(df, col='Survived', size=size,aspect=1.6)
    g.map(plot.hist, x, bins=b)
    
def compoundHistograms(df, numerical, ordinal, b=20, size=2.2):
    from seaborn import FacetGrid as grid
    from matplotlib import pyplot as plot
    g = grid(df, col='Survived', row=ordinal, size=size, aspect=1.6)
    g.map(plot.hist, numerical, bins=b)
    

def infoHist(df, numerical, ordinal, b=20, size=2.2):
    from seaborn import FacetGrid as grid
    from matplotlib import pyplot as plot
    g = grid(df, row=ordinal, size=size, aspect=1.6)
    g.map(plot.hist, numerical,bins=b)
    
def tripleBox(df, numerical, ordinal1, ordinal2, dims=(12,12)):
    from seaborn import boxplot as graph
    from matplotlib import pyplot as plot
    dummy, ax = plot.subplots(figsize=dims)
    graph(x=ordinal1, y=numerical, hue=ordinal2, data=df, ax=ax)
    
def boxPlot(df, ordinal, numerical, survival=False, dims=(12,16)):
    from seaborn import boxplot as graph
    from matplotlib import pyplot as plot
    dummy, ax = plot.subplots(figsize=dims)
    if survival:
        graph(x=ordinal, y=numerical, hue='Survived', data=df, ax=ax)
    else:
        graph(x=ordinal, y=numerical, data=df, ax=ax)
        
def regPlot(df, x, y, dims=(12,12)):
    from seaborn import regplot as rp
    from matplotlib import pyplot as plot
    dummy, ax = plot.subplots(figsize=dims)
    g = rp(x=x, y=y, data=df, ax=ax)


# ## Determine Correlations
# We'll first sort on mean survival rate based on a number of different features. Some of these are correlated, some are not. It's quite easy to see trends by simply looking at average survival rates for each feature.
# 
# We'll also explore how some of these features are correlated with each other, which will help us engineer features and complete our data later.

# ### Correlation with Survival
# 
# ---
# 

# In[ ]:


# See average survival rate for each class
sortByFeature(trainDF, 'Pclass')


# This tells us that there is some correlation between **Pclass** and Survival.

# In[ ]:


# Survival rate by sex
sortByFeature(trainDF, 'Sex')


# **Sex** is correlated with survival

# In[ ]:


sortByFeature(trainDF, 'Embarked')


# In[ ]:


survivalHistogram(trainDF, 'Age', 20, 3)


# In[ ]:


compoundHistograms(trainDF, 'Age', 'Pclass')


# In[ ]:


compoundHistograms(trainDF, 'Age', 'Sex', 15)


# In[ ]:


boxPlot(trainDF, 'Survived', 'Age')


# We can see from this that infants tended to survive, and that the group that survived tended to be younger in general, however the median age for survivors and victims are the same, indicating that this may not be the best predictor of survival

# In[ ]:


survivalHistogram(trainDF, 'Fare', 20, 3)


# In[ ]:


boxPlot(trainDF, 'Survived', 'Fare')


# Here we can see that survival is correlated with fare, which may not be surprising.

# 
# ---
# The following are uncorrelated. Maybe we will engineer some features later using this data

# In[ ]:


# Survival rate by number of siblings and spouses aboard
sortByFeature(trainDF, 'SibSp')


# In[ ]:


# Survival rate by number of parents and children aboard
sortByFeature(trainDF, 'Parch')


# ### Correlation with Fare
# 
# ---
# 

# In[ ]:


# Group Embarked location by average Fare
sortByFeature(trainDF,'Embarked','Fare')


# In[ ]:


sortByFeature(trainDF, 'Embarked', 'Pclass')


# In[ ]:


# Graphical representation of above data
boxPlot(trainDF, 'Embarked', 'Fare')


# In[ ]:


#sortByFeature(trainDF, 'Pclass', 'Fare')
boxPlot(trainDF,'Pclass', 'Fare')


# In[ ]:


sortByFeature(trainDF, 'Sex', 'Fare')


# In[ ]:


trainDF[['Fare','Age']].corr()


# This tells us that we can probably take **Embarked, Pclass, and Sex**, but not Age into account when we complete **Fare**

# In[ ]:


trainDF[['Fare', 'Pclass']].corr()


# ### Correlation with Age
# 
# ---
# 

# In[ ]:


boxPlot(trainDF, 'Sex','Age')


# In[ ]:


boxPlot(trainDF,'Pclass','Age')


# In[ ]:


boxPlot(trainDF,'SibSp','Age')


# In[ ]:


boxPlot(trainDF,'Parch','Age')


# In[ ]:



pd.crosstab(trainDF.Embarked, trainDF.Pclass)


# This tells us that the women were generally younger, and the older passengers were generally in first class, so we may use this to complete the Age term, but as we saw earlier, there is not a strong correlation between age and fare.

# # Polish Data
# 
# ---
# 

# ## Gain Useful Data from Names
# I'm borrowing this straight from Manav Sehgal
# I am about to:
# 
# 1. Extract the title from the name
# 2. Replace the title with a numerical value
# 

# In[ ]:


# Extract titles
for df in allData:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#pd.crosstab(trainDF.Title, trainDF.Sex)
pd.crosstab(testDF.Title, testDF.Sex)


# We can see that almost every title is mutually exclusive with regards to sex (with the exception of Dr.). We'll now explore the survival rates for each title in each sex.

# In[ ]:



sortByFeature(trainDF[trainDF.Sex=='female'],'Title')


# In[ ]:


sortByFeature(trainDF[trainDF.Sex=='male'],'Title')


# We see the trend of women surviving again. In order to reduce noise, I seek to group titles into larger bins, according to age (and knowing that Mlle translates to Miss and Mme translates to Mrs)

# In[ ]:


sortByFeature(trainDF, 'Title', 'Age')


# In[ ]:


for df in allData:
    df.loc[df.Sex=='female','Title']=df.loc[df.Sex=='female','Title'].replace(['Lady','Countess','Mme','Dr','Dona'],'Mrs')
    df.loc[df.Sex=='female','Title']=df.loc[df.Sex=='female','Title'].replace(['Ms', 'Mlle'],'Miss')
    df.loc[df.Sex=='male','Title']  =df.loc[df.Sex=='male','Title'].replace(['Capt','Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir'],'Mr')

pd.crosstab(trainDF.Title, trainDF.Sex)


# In[ ]:


sortByFeature(trainDF, 'Title','Age')


# In[ ]:


sortByFeature(trainDF,'Title')


# # Map ordinal features to numeric

# In[ ]:


# Map Title
titleMap={'Mrs':1,'Miss':2,'Master':3,'Mr':4}
for df in allData:
    df.Title = df.Title.map(titleMap)
pd.crosstab(trainDF.Title, trainDF.Sex)

# Map Sex
sexMap={'female':0,'male':1}
for df in allData:
    df.Sex = df.Sex.map(sexMap)


# # Complete Data
# We now seek to complete the missing data for the features we will use:
# 
# - Age
#     - Using **Pclass** and **Title**
# - Embarked
#     - Using **Pclass**
# - Fare
#     - Using **Pclass** and **Embarked**

# ###Completing Age
# 
# ---
# 
# We'll train a regressor to guess the age

# In[ ]:


tripleBox(trainDF,'Age','Title','Pclass')


# This shows us that age is generally correlated with both Title and Pclass, so perhaps we should take both into account when completing the Age field. I will now replace the ordinal Title with a numerical value so we can **train a regressor to estimate an age for each Age based on Pclass and Title**. The correlation matrices below verify this intuition.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

def completeAge(df, useRegressor=False):
    if useRegressor:
        # Grab pertinent features (only complete ones besides Age)
        ageDF = df[['Age','Sex','Title','Parch','SibSp','Pclass']]
        knownAgeDF = ageDF.loc[(df.Age.notnull()) & (df.Age != 0)]
        unknownAgeDF = ageDF.loc[(df.Age.isnull()) | (df.Age == 0)]
        #print('len(knownAge)   = %d'%(len(knownAgeDF)))
        #print('len(unknownAge) = %d'%(len(unknownAgeDF)))
        # Pull values
        Y = knownAgeDF.values[:,0]
        X = knownAgeDF.values[:,1::]
        rfr = RandomForestRegressor(n_estimators=2000)
        rfr.fit(X, Y)
        predictedAges = rfr.predict(unknownAgeDF.values[:, 1::])
        df.loc[(df.Age.isnull()) | (df.Age == 0),'Age'] = predictedAges
    else:
        ageMap = np.zeros((3,4))
        for p in range(1,ageMap.shape[0]+1):
            for t in range(1,ageMap.shape[1]+1):
                ageMap[p-1][t-1] = trainDF.loc[trainDF.Age.notnull() & (trainDF.Pclass==(p)) & (trainDF.Title==(t)),'Age'].median()
                df.loc[df.Age.isnull() & (df.Pclass==p) & (df.Title==t),'Age'] = ageMap[p-1][t-1]
    return df


# In[ ]:


for df in allData:
    df = completeAge(df)


# ###Completing Embarked
# 
# ---
# 

# In[ ]:


len(trainDF[trainDF.Embarked.isnull()])


# In[ ]:


mostEmbarked = np.zeros((3,1),'str')
for i in range(0,len(mostEmbarked)):
    mostEmbarked[i] = trainDF[trainDF.Pclass==(i+1)].Embarked.dropna().mode()[0]

for df in allData:
    for p in range(1,4):
        df.loc[(df.Pclass==p) & df.Embarked.isnull(),'Embarked'] = mostEmbarked[p-1]
len(trainDF[trainDF.Embarked.isnull()])


# ###Completing Fare
# 
# ---
# 

# In[ ]:


tripleBox(trainDF,'Fare','Embarked','Pclass')


# We'll do a quick mapping onto a numerical feature, and rearrange how we look at the embark location to see a stronger correlation.

# In[ ]:


print(len(trainDF[trainDF.Embarked.isnull()]))
sortByFeature(trainDF,'Embarked')


# In[ ]:


embarkMap={'C':1,'Q':2,'S':3}
#embarkMap={1:1,2:3,3:2}
for df in allData:
    df.Embarked = df.Embarked.map(embarkMap)
#for df in allData:
#    df.Title = df.Embarked.map(embarkMap)
#pd.crosstab(trainDF.Embarked, trainDF.Sex)
trainDF[['Embarked','Fare']].corr()


# Now we'll estimate fare with a regressor, similar to age.

# In[ ]:


def completeFare(df, useRegressor=False):
    if useRegressor:
        fareDF = df[['Fare','Age','Title','Embarked','SibSp','Parch','Pclass']]
        # Get un/known values
        knownFareDF = fareDF.loc[(fareDF.Fare > 0) & (fareDF.Fare.notnull())]
        unknownFareDF = fareDF.loc[(fareDF.Fare <= 0) | (fareDF.Fare.isnull())]
        Y = knownFareDF.values[:,0]
        X = knownFareDF.values[:,1::]
        rfr = RandomForestRegressor(n_estimators=2000)
        rfr.fit(X, Y)
        predictedFares = rfr.predict(unknownFareDF.values[:,1::])
        df.loc[(df.Fare.isnull()) | (df.Fare <= 0),'Fare'] = predictedFares
    else:
        fareMap = np.zeros((3,3))
        for p in range(1,fareMap.shape[0]+1):
            for e in range(1,fareMap.shape[1]+1):
                fareMap[p-1][e-1] = trainDF.loc[(trainDF.Fare!=0) & (trainDF.Pclass==(p)) & (trainDF.Embarked==(e)),'Fare'].median()
                df.loc[( (df.Fare.isnull()) | (df.Fare==0) ) & (df.Pclass==p) & (df.Embarked==e),'Fare'] = fareMap[p-1][e-1]
            
    return df


# In[ ]:


for df in allData:
    df = completeFare(df)


# # Band Features to Reduce Noise
# 
# ---
# 

# In[ ]:


trainDF.Fare.hist()


# In[ ]:


testDF.Fare.hist()


# In[ ]:


nCuts = 4
#ageCutoffs = list(range(0,80,int(80/nCuts)))
trainDF['AgeBand'], bins = pd.qcut(trainDF.Age, nCuts, retbins=True)
sortByFeature(trainDF, 'AgeBand')
for i in range(0, len(bins)):
    for df in allData:
        if i == nCuts:
            df.loc[(df.Age >= bins[i]),'AgeGroup'] = i
        elif i == 0:
            df.loc[(df.Age < bins[i+1]),'AgeGroup'] = i
        else:
            df.loc[(df.Age >= bins[i]) & (df.Age < bins[i+1]),'AgeGroup'] = i
trainDF.AgeGroup.head()
bins
trainDF[['AgeGroup','Survived']].corr()
sortByFeature(trainDF,'AgeGroup')
survivalHistogram(trainDF, 'AgeGroup')


# In[ ]:


trainDF['FareBand'], bins = pd.qcut(trainDF.Fare, nCuts, retbins=True)
sortByFeature(trainDF, 'FareBand')
for i in range(0, len(bins)):
    for df in allData:
        if i == nCuts:
            df.loc[(df.Fare >= bins[i], 'FareGroup')] = i
        elif i == 0:
            df.loc[(df.Fare < bins[i+1], 'FareGroup')] = i
        else:
            df.loc[(df.Fare >= bins[i]) & (df.Fare < bins[i+1]), 'FareGroup'] = i
sortByFeature(trainDF,'FareGroup')


# In[ ]:


def makeFamSizeGroup(df):
    famSizeMap = {0:0,
                  1:1, 2:1, 3:1,
                  4:2, 5:2, 6:2,
                  7:3, 8:3, 9:3, 10:3}
    df['FamSize'] = trainDF.SibSp + trainDF.Parch
    df['FamSizeGroup'] = df.FamSize.map(famSizeMap)

for df in allData:
    df = makeFamSizeGroup(df)


# In[ ]:


trainDF[['FamSizeGroup','Survived']].corr()
#survivalHistogram(trainDF, 'FamSizeGroup')
#sortByFeature(trainDF,'FamSize')
#print(ceil(3.4))


# In[ ]:



#Do some PCA on this data to improve generalizability
featureList=['Pclass','AgeGroup','FareGroup','Embarked','Title','Sex','FamSizeGroup','FamSize','Age','Fare']
#featureList = ['Pclass','AgeGroup','FareGroup','Embarked','Title','Sex','FamSizeGroup']
#featureList = ['Pclass','Sex']
fullList = featureList + ['Survived']
#fullList=[list(featureList)[:],'Survived']
trainDF[fullList].corr()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
rfc = RandomForestClassifier(n_estimators = 1000, max_features='sqrt')

X_train = trainDF[featureList].values[:,:]
#X_train = trainDF.values[:,:]
Y_train = trainDF['Survived'].values
X_test = testDF[featureList].values[:,:]
rfc = rfc.fit(X_train, Y_train)

features = pd.DataFrame()
features['feature'] = trainDF[featureList].columns
features['importance'] = rfc.feature_importances_
features.sort_values(by='importance', ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh',figsize = (10,10))


# Seeing this, I think it would be interesting to see how we do with dimensionality reduction.

# In[ ]:


#from sklearn.decomposition import PCA
#pca = PCA(0.99)
#pca.fit(X_train)
#X_train = pca.fit_transform(X_train)
#X_test = pca.fit_transform(X_test)


# # Classification

# In[ ]:


from sklearn.model_selection import cross_val_score
cuts = 5


def ccr(model, X, Y):
    return round(model.score(X, Y)*100, 2)

def printStats(model, X, Y, name, cuts=cuts):
    print('Cross-Validation Scores for %s:'%(name))
    print(cross_val_score(model, X, Y, cv=cuts))
    print('\nMean Correct-Classification rate for %s:'%(name))
    print(ccr(model, X, Y))


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
Yp = logReg.predict(X_test)
ccr_logReg = ccr(logReg, X_train, Y_train)
print('Cross-Validation Scores for logReg:')
print(cross_val_score(logReg, X_train, Y_train, cv=cuts))
print('\nMean Correct-Classification rate for logReg:')
print(ccr_logReg)


# In[ ]:


# SVM
from sklearn import svm
mySVM = svm.SVC()
mySVM.fit(X_train, Y_train) 
Yp_svm = mySVM.predict(X_test)
ccr_svm = ccr(mySVM, X_train, Y_train)
print('Cross-Validation Scores for SVM:')
print(cross_val_score(mySVM, X_train, Y_train, cv=cuts))
print('\nMean Correct-Classification rate for SVM:')
print(ccr_svm)


# In[ ]:


# kNN
from sklearn.neighbors import KNeighborsClassifier
n_neighbors = 3
knn = KNeighborsClassifier(n_neighbors = n_neighbors)
knn.fit(X_train, Y_train)
Yp_knn = knn.predict(X_test)
ccr_knn = ccr(knn, X_train, Y_train)
printStats(knn, X_train, Y_train, 'KNN')


# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB()
naiveBayes.fit(X_train, Y_train)
Yp_naiveBayes = naiveBayes.predict(X_test)
printStats(naiveBayes, X_train, Y_train, 'Naive Bayes')


# In[ ]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Yp_perceptron = perceptron.predict(X_test)
printStats(perceptron, X_train, Y_train, 'Perceptron')


# In[ ]:


# Linear SVM
from sklearn.svm import LinearSVC

linearSVM = LinearSVC()
linearSVM.fit(X_train, Y_train)
Yp_linearSVM = linearSVM.predict(X_test)
printStats(linearSVM, X_train, Y_train, 'Linear SVM')


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Yp_sgd = sgd.predict(X_test)
printStats(sgd, X_train, Y_train, 'Stochastic Gradient Descent')


# In[ ]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dTree = DecisionTreeClassifier()
dTree.fit(X_train, Y_train)
Yp_dTree = dTree.predict(X_test)
printStats(dTree, X_train, Y_train, 'Decision Tree')


# In[ ]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier
n_estimators = 100
randForest = RandomForestClassifier(n_estimators = n_estimators)
#randForest = GridSearchCV(RandomForestClassifier(n_estimators = n_estimators),
#                          {"max_depth":[None, 5, 2],
#                           "min_samples_leaf":[2,3,5]})
randForest.fit(X_train, Y_train)
Yp_randForest = randForest.predict(X_test)
printStats(randForest, X_train, Y_train, 'Random Forest')


# # Model Selection
# The Random Forest Classifier produced the best results and is likely to avoid overfitting, so that shall be the one I use.

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": testDF["PassengerId"],
        "Survived": Yp_randForest
    })
submission.to_csv('submission.csv', index=False)

