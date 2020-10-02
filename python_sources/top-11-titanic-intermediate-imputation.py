#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv(r'/kaggle/input/titanic/train.csv')
test = pd.read_csv(r'/kaggle/input/titanic/test.csv')
data = pd.concat([train,test],axis=0, sort=True).reset_index(drop=True)


# Feature with the highest amount of missing values is *Cabin*. The first thing that comes in the mind is to drop it due to mostly missing values, but we will try to retrieve as much information as its possible, so for now its going to stay. Also, we have two missing *Embarked* values, as well as one *Fare* missing. In order to perform more advanced imputation later on, we're going to fill those values right now.

# In[ ]:


missing = data.isnull().sum().sort_values(ascending=False) / len(data)

plt.figure(figsize=(15,7))
sns.barplot(x=missing.index, y=missing.values)
plt.title('Amount of missing data relative to whole dataset', fontsize=15)
plt.xlabel('Feature',fontsize=11)
plt.ylabel('Amount of missing data',fontsize=11)


# I decided to imput *Fare* values as the median of the amount paid by people embarked at Southampton for third class. As the *Embarked* value I decided to go with the most popular embarkation city.

# In[ ]:


data.loc[data.Fare.isnull(),'Fare'] = data[(data.Embarked=='S') & (data.Pclass==3)].Fare.median()
data.loc[data.Embarked.isnull(),'Embarked'] = 'S'


# In[ ]:


data.Name.head(10)


# As you may see above, *Name* variable holds information about title of each person. Besides ordinary titles such as Mr, or Mrs we also have:
# 
# 1. *Miss* - title for unmarried woman
# 1. *Master* - title for young boy
# 1. *Rev* - clerical title
# 1. Higher social status titles - *Sir*, *Lady*, *the Countess*, *Don*, *Dona*, *Jonkheer*
# 1. Military titles - *Col*, *Major*, *Capt*
# 1. Equivalent titles - *Mlle*, *Ms*, *Mme*
# 
# Our next step is to retrieve those titles from *Name* feature and group together titles with lower appearance frequency.

# In[ ]:


data['Title'] = data.Name.map(lambda x: x.split(',')[1].split('.')[0])

#Because Mlle is french equivalent of Miss
data.loc[data.Title==' Mlle', 'Title'] = ' Miss'
#They were travelling alone, so I assume that they are unmarried
data.loc[data.Title==' Ms', 'Title'] = ' Miss'
#Mme is french equivalent for Mrs
data.loc[data.Title==' Mme', 'Title'] = ' Mrs'
#Royals and stuff
data.Title = data.Title.replace({' Sir':'Demigod',' Lady':'Demigod',' the Countess':'Demigod',' Don':'Demigod', ' Dona':'Demigod', ' Jonkheer':'Demigod'})
#Military
data.Title = data.Title.replace({' Col':'Military',' Major':'Military',' Capt':'Military'})

data.loc[(data.Title=='Demigod') & (data.Sex=='female'),'Title'] = ' Miss'
data.loc[(data.Title=='Demigod') & (data.Sex=='male'),'Title'] = ' Mr'


# I decided to group demigods together with titles associated with gender, because of very low frequency in dataset - only 6 observations in the whole dataset have *Demigod* title.
# 
# In the cell below, we're performing a bit of feature engineering. We do assume, that cabin letters may bring additional information about the probability of survival.

# In[ ]:


data['CabinLetter'] = data.Cabin.astype(str).map(lambda x: str(x[0]))
data['CabinLetter'] = data.CabinLetter.fillna('n')
data =  data.rename(columns={'Sex':'SexFemale'})
data['SexFemale'] = data.SexFemale.astype(str).replace({'male':0,'female':1})


# # Age imputation possible concepts
# 
# If it comes to most basic approach in filling missing data, we might want to fill it simply by dataset's median, but we might then lose a lot of information. Also, we may suspect, that people traveling in the first class have more money, which can be associated with being older. On the other hand, we may use some patterns from *Title* variable or there might be a higher dimension relationship between *Age* and multiple variables which will help us fill missing values more accurately. In order to choose wisely, we're going to compare following methods:
# 
# 1. Median of age imputation
# 1. Median in subclasses imputation
# 1. Model approach
# 
# Each of the following will be treated as a separate model. In order to evaluate which one is doing better, we're going to compare them using mean squared error metric.

# ## Median of age imputation

# In[ ]:


np.random.seed(1)

Y = data.Age[data.Age.isnull()==False]
testY = Y.loc[np.random.choice(Y.index,100, replace=False)]
median = np.median(Y.loc[np.setdiff1d(Y.index,testY.index)])
MaI = np.sqrt(mean_squared_error(testY, np.full((100,1),median)))
print('Holdout set:',MaI)


# ## Median in subclasses imputation

# In[ ]:


np.random.seed(1)

testX = pd.DataFrame(data.Title.loc[testY.index])
testX['Predicted'] = np.NaN

for i in data.Title.loc[np.setdiff1d(Y.index,testY.index)].unique():
    testX.loc[testX.Title==i, 'Predicted'] = data[data.Title==i].Age.loc[np.setdiff1d(Y.index,testY.index)].median()

MasI = np.sqrt(mean_squared_error(testY, testX.Predicted))
    
print('Holdout set:',MasI)


# ## Model approach
# In model approach, we're hoping, that independent variables in the dataset can help us to provide reasonable predictions for missing values. Before creating a model, we're going to create more features:
# 
# 1. *Family* - by summing *SibSp* and *Parch* we're providing additional information
# 1. *PeopleOnTheSameTicket* - one passenger can travel with more than one person on the same ticket
# 1. *FarePerPerson* - if multiple passengers belong to a single ticket, then fare value is multiplied by the amount of passengers - that's why there is so much difference between the values of ticket even when embarkation port and class is the same
# 1. *Wealth* - binned *FarePerPerson*. Usually, binning continuous variables might be useful for specific models. For example, when using Random Forest, instead of forcing a model to look for the best value for a split in continuous variable, we may force it to use pre-defined feature

# In[ ]:


data['Family'] = data.SibSp + data.Parch + 1
data['PeopleOnTheSameTicket'] = None

for i in data.Ticket.unique():
    summary = len(data[data.Ticket==i])
    data.loc[data.Ticket==i, 'PeopleOnTheSameTicket'] = summary

data['FarePerPerson'] = data.Fare/data.PeopleOnTheSameTicket
data['Wealth'] = pd.cut(data.FarePerPerson,7,labels=range(1,8))


# In[ ]:


from sklearn.linear_model import Lasso, Ridge
X = data[data.Age.isnull()==False]
X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'PeopleOnTheSameTicket','FarePerPerson','Family','Wealth']]
Y = X.Age
X = X.drop('Age',axis=1)
X = pd.get_dummies(X, drop_first=True)
testX = X.loc[testY.index]
X = X.drop(testY.index,axis=0)
Y = Y.drop(testY.index,axis=0)

model = XGBRegressor(learning_rate=0.001,n_estimators=3000, reg_lambda=0.1, objective='reg:squarederror')
model.fit(X, Y)

MA = np.sqrt(mean_squared_error(testY, model.predict(testX)))
print('Holdout set:',MA)


# In[ ]:


print('Median of age:',MaI,'\nMedian of age in subclasses:',MasI,'\nModel approach:',MA)


# As you may see above, the best results are provided using model approach, but the thing which you have to keep in mind is, that the results are for the specific random sample - the most optimal solution would be chosen after creating multiple random samples, or by cross-validation.

# In[ ]:


#Model age imputation

X = data[data.Age.isnull()==False]
X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'PeopleOnTheSameTicket','FarePerPerson','Family','Wealth']]
Y = X.Age
X = X.drop('Age',axis=1)
X = pd.get_dummies(X, drop_first=True).drop(['Title_ Rev','Title_Military'],axis=1)

model = XGBRegressor(learning_rate=0.001,n_estimators=3000, reg_lambda=0.1)
model.fit(X,Y)
dummydata = pd.get_dummies(data[data.Age.isnull()][['Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title',
                                                    'PeopleOnTheSameTicket','FarePerPerson','Family','Wealth']], drop_first=True)

data.loc[data.Age.isnull(), 'Age'] = model.predict(dummydata)


# As we finally filled all missing values in *Age* variable, we may now create some additional features:
# 1. *AgeInterval* - hoping to get some additional informations for model by dividing into three separate groups
# 1. *FareTAge* - paying more for ticket (thus being richer) along with being older may increase survivability
# 
# 

# In[ ]:


data['AgeInterval'] = pd.cut(data.Age, 3, labels=['Young','Mid-aged', 'Old'])
data['FareTAge'] = data.Fare * data.Age


# # Model approach for Cabin Letter variable
# I think, that one of the best possible solutions, is to fill missing values of *CabinLetter* variable by using model approach. In addition to that, we may use some other informations - according to https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic:
# 1. A,B,C - 1st class only
# 1. D,E - all classes
# 1. F, G - 2nd and 3rd class
# 
# Be aware, that according to some other sources (e.g. https://www.encyclopedia-titanica.org/cabins.html) only 3rd class was allocated in G cabin and I am going to stick to this convention.

# In[ ]:


X = data[data.CabinLetter!='n']
X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'CabinLetter']]
Y = X.CabinLetter
X = X.drop('CabinLetter',axis=1)
X = pd.get_dummies(X)
Y = Y.replace({'C':1,'E':2,'G':3,'D':4,'A':5,'B':6,'F':7,'T':8})

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.3)

model = RandomForestClassifier(n_estimators=500, random_state=1)
model.fit(trainX,trainY)

trainX = pd.concat([trainX,trainY], axis=1)
testX = pd.concat([testX, testY], axis=1)

for i in trainX.CabinLetter.unique():
    print('Train result for {} :'.format(i), accuracy_score(trainX.CabinLetter[trainX.CabinLetter==i], model.predict(trainX.drop('CabinLetter',axis=1)[trainX.CabinLetter==i])))
    
for i in testX.CabinLetter.unique():
    print('Holdout result for {} :'.format(i), accuracy_score(testX.CabinLetter[testX.CabinLetter==i], model.predict(testX.drop('CabinLetter',axis=1)[testX.CabinLetter==i])))
    
    
print('Accuracy overall:', accuracy_score(testX.CabinLetter, model.predict(testX.drop('CabinLetter',axis=1))))


# As you may see, the accuracy on holdout set is quite satisfying. Unfortunately, it decreases for classes with smaller values of samples.

# In[ ]:


#Model CabinLetter inmputation 

X = data[data.CabinLetter!='n']
X = X.drop(X[X.Age.isnull()].index,axis=0)
X = X[['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'CabinLetter']]
Y = X.CabinLetter
X = X.drop('CabinLetter',axis=1)
X = pd.get_dummies(X, drop_first=True)
X['Title_ Rev'] = 0
Y = Y.replace({'C':1,'E':2,'G':3,'D':4,'A':5,'B':6,'F':7,'T':8})

model = RandomForestClassifier(n_estimators=500, random_state=1)
model.fit(X,Y)
dummydata = pd.get_dummies(data[data.CabinLetter=='n'][['Age', 'Embarked','Fare', 'Parch','Pclass','SexFemale','SibSp','Title', 'CabinLetter']], drop_first=True)
data.loc[data.CabinLetter=='n','CabinLetter'] = model.predict(dummydata)

data.CabinLetter = data.CabinLetter.replace({1:'C',2:'E',3:'G',4:'D',5:'A',6:'B',7:'F',8:'T'})


# In[ ]:


# Because its child is in cabin F and people from 2nd class dont get G cabin
data.CabinLetter.loc[247] = 'F'
# Husband is in E and people from 3rd class dont get C cabin
data.CabinLetter.loc[85] = 'E'
# Rest is in F and people from 2nd class dont get C cabin
data.CabinLetter.loc[665] = 'F'
# There is no F cabin. Changing to A is justified by similar Fare to median of Fare for 1st class in cabin A
data.CabinLetter.loc[339] = 'A'


# # Model creation
# In order to evaluate model performance, we're going to compare XGBoost with Random Forest. Also, 10 fold cross-validation will be used.

# In[ ]:


data = data.drop(['Cabin','Name','Ticket'],axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data[data.Survived.isnull()==False].drop(['Survived', 'PassengerId'],axis=1)
Y = data.Survived[data.Survived.isnull()==False]
trainX, testX, trainY, testY = train_test_split(X,Y,test_size=0.30, random_state=2019)


# In[ ]:


def ModelCreation(model):
    cv = cross_validate(model,
                        X,
                        Y,
                        cv=10,
                        scoring='accuracy',
                        return_train_score=True)

    print('Training score average:',cv['train_score'].sum()/10)
    print('Holdout score average:',cv['test_score'].sum()/10)


# In[ ]:


model = RandomForestClassifier(n_estimators=900, max_depth=4)
ModelCreation(model)


# In[ ]:


model = XGBClassifier(learning_rate=0.001,n_estimators=4000,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27)

ModelCreation(model)


# Higher holdout score on training set comes at the cost of higher variation, thus model results are going to be less predictable on test data. We could try to decrease overfitting by introducing L2 parameter, decreasing max_depth, n_estimators, etc., but none of these resulted in receiving better results on public leaderboard. On the other hand Random Forest gives worse results on training data, but it almost ideally fits holdout data. It is also projected in test data with significantly higher accuracy score comparing to XGBoost.

# In[ ]:


model = RandomForestClassifier(n_estimators=900, max_depth=4)
model.fit(X, Y)
test = data[data.Survived.isnull()]
test['Survived'] = model.predict(test.drop(['Survived','PassengerId'],axis=1)).astype(int)
test = test.reset_index()
test[['PassengerId','Survived']].to_csv("RFClass.csv",index=False)


# # Self-promotion
# If you enjoyed reading my notebook, or at least find it useful in some way or another, I highly encourage you to check out my other creations:
# 
# 1. Top 20% Ames House Prices kernel https://www.kaggle.com/paragraph/rmlse-0-119-top-21-house-prices-regression#Final-thoughts
# 1. Data visualisation with Baseline (map chart sublibrary) https://www.kaggle.com/paragraph/airlines-data-visualisation-with-baseline
# 1. Text mining with NLTK library https://www.kaggle.com/paragraph/text-mining-nltk-and-k-means-included
