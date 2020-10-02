#!/usr/bin/env python
# coding: utf-8

# ### Titanic Dataset
# 
# The Titanic dataset regards the most famous and tragic marine disaster of history. Every row represent a Titanic passenger and every column feature some characteristic like Age, Sex, etc. We will use the data splitted in train and test provided by the beginner Kaggle competition "Titanic: Machine Learning from Disaster": the aim is to predict wether a passenger from the test set survived or not using a machine learning algorithm.

# Let's start imorting libraries and modules we will need for data exploration and cleaning, creating and combining features and using machine learning algorithms.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# Now we can import the data, splitted in training set, the one on which we will fit the model, and test set, the one on which we will make predictions. We also made the total list in order to clean and structure the data in both the two datasets at the same time.

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
total = [train, test]


# Let's take a look to the first 10 rows of the train dataset.

# In[ ]:


train.head(10)


# # Variables description and intuitions:
# * PassengerId: unique identification number of each passenger, useless for predictive purposes.
# * Survived: this is the value we want to predict in our model, it's a binary variable with value 1 if passenger survived, 0 otherwise. We can see that the survival rate in the training set is about 38%.
# * Pclass: categorical variable indicating the class of the Passenger ticket, as we can read from https://en.wikipedia.org/wiki/Passengers_of_the_RMS_Titanic, in 1st class there were the wealthiest passengers on board like businessmen, politicians or industrialists, in the 2nd class passengers were predominantly middle-class travelers like tourists and in the 3rd class passengers were primarily immigrants moving to the United States and Canada. We can imagine that the travel condition, in term of safety, of the 3rd class passengers was bad compared to the one in 1st class, so we expect a low survival rate for the 3rd class passengers.
# * Name: name and title of the passenger, we can imagine that the name is insignificant for predictive purposes, but the title could be an interesting variable to extract from the name column.
# * Sex: categorical variable assuming values male for men and female for women. In the training set, we have more men than women (577 men on 891 total passengers).
# * Age: numerical variable for the passenger age. As we can see, the average age in the training set is almost 30 years-old, the youngest passenger was almost 2 months-old, while the oldest was 80 years-old.
# * SibSp: numerical variable indicating the number of siblings and spouses the passenger had aboard.
# * Parch: numerical variable indicating the number od parents and children the passenger had aboard.
# * Ticket: categorical variable indicating the alphanumerical ticket code.
# * Fare: numerical variable for the price of the ticket.
# * Cabin: categorical variable indicating the alphanumerical cabin code.
# * Embarked: categorical variable for the emarkation port, S stands for Southampton, Q for Queenstow and C for Cherbourg. Most of passengers (644) embarked in Southampton port.

# In[ ]:


train.describe(include='all')


# ### Outliers
# 
# By looking to the features, the only one that seems to have outliers is the Fare feature because for example for the SibSp 8 familiars (brothers and partners) can be real especially in 1912; also for the Parch (Parents and children) variable we can make the same consideration.
# 
# By the boxplot below we can see many outliers, but are they real outlier? We can imagine that they are in same unit/currency, so according to Titanic informations, first class ticket could cost a maximum of 870 pounds (4350$ in 1912), so we can argue that they aren't outlier, simply there are few really expensive cabins/suites and many lower price Cabins. So we won't remove these observations. Same thing for the other two classes, we can imagine that some tickets have been sold with an higher price or some cabins had an higher price than the average. Moreover, we will transform this continuos variable into a discrete variable, so the possible outlier problem will be solved.

# In[ ]:


sns.boxplot(y='Fare',x='Pclass', data=train)


# ### Missing values
# We will have to handle some missing values in the training and in the test set. In the training set we are missing almost 20% of Age values, about 77% of Cabin values and only 2 Embarked values. In the test set we have the same issue with Age and Cabin, we don't miss any Embarked value but we miss 1 Fare value. We can manage the missing data after a short exploratory data analysis on the Titanic train dataset.

# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(12,6))
sns.heatmap(train.isnull(), cbar=False, ax=axes[0])
sns.heatmap(test.isnull(), cbar=False, ax=axes[1])


# # Features
# ### Survived
# 
# Paying attention to the Survived feature, the variable we want to predict, we can see from the plot below that in the training set, the survival rate is almost 40%, therefore about 60% of passengers didn't survive.

# In[ ]:


train['Survived'].value_counts(normalize=True).rename({0: 'Not Survived', 1: 'Survived'}).plot(kind='bar', color='#1F4E97', title='Survival rate')
plt.xticks(rotation=0)


# ### Sex
# 
# These two plots confirm how we can imagine rescue operations aboard the Titanic: women (and children) first!

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.countplot(train['Sex'], ax=axes[0])
axes[0].set_ylabel('Passengers')
axes[0].set_title('Passengers by sex')
sns.barplot(x='Sex', y='Survived', data=train, ci=None)
axes[1].set_ylabel('Survival rate')
axes[1].set_title('Survival rate by sex')


# ### Passenger Class
# 
# As we could expect, the highest survival rate is in the 1st class, so in the Titanic disaster, the more you were rich the more you had a chance to get rescued: we can imagine that 1st class cabin were in a better position and were safer than 2nd or 3rd class cabins. Also 1st class passenger were influent people like politicians or important businessmen and maybe they took precedence in rescue operations in some way.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.countplot(train['Pclass'], ax=axes[0])
axes[0].set_ylabel('People')
axes[0].set_title('Number of people by class')
sns.barplot(x='Pclass', y='Survived', data=train, ci=None)
axes[1].set_ylabel('Survival Probability')
axes[1].set_title('Survival Probability by class')


# ### Fare
# 
# Just as verified for PClass variable, most of the passengers that didn't survive bought cheaper tickets, as we can see the the blue peak in plot below: when we overcome a Fare of about 25, we can see that more people survived, until we reach high values of Fare in which, as we will see, there are few cases. 

# In[ ]:


fig, axes = plt.subplots(figsize=(6,6))
axes.set_ylabel('Passengers')
axes.set_title('Fare distribution')
sns.distplot(train[train['Survived']==1]['Fare'],kde=False,color='green')
sns.distplot(train[train['Survived']==0]['Fare'],kde=False,color='blue')
axes.legend(['Survived','Not Survived'])


# ### Age
# 
# As said before, the rescue operations statement is: (women) and children first! So we can see more survived chidren and less survived adults. Curiously 60 years-old or more passengers had an higher survival probability. So just looking at this plot, we can divide Passengers in 4 Age groups: 0 to 13, 13 to 30, 30 to 55, 55 to max.

# In[ ]:


fig, ax =plt.subplots()
a=sns.kdeplot(train[train['Survived'] == 0]['Age'],ax=ax)
b=sns.kdeplot(train[train['Survived'] == 1]['Age'],ax=ax)
ax.legend(['Not Survived','Survived'])
ax.set_xlabel('Age')
ax.set_title('Age - kernel density estimation')

axins = ax.inset_axes([1.2, 0.4, 0.4, 0.4])
x1, x2, y1, y2 = 0, 15, 0, 0.012
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

sns.kdeplot(train[train['Survived'] == 0]['Age'],ax=axins)
sns.kdeplot(train[train['Survived'] == 1]['Age'],ax=axins)
axins.get_legend().remove()
axins.set_title('Lower age zoom')
ax.indicate_inset_zoom(axins)


# ### Embarked
# 
# Southampton was the first embarkation port, then Cherbourg and finally Queenstown. We can see that Southampton has the worst survival rate, but we have to consider that more than 75% of passengers embarked there.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.countplot(train['Embarked'].map({'S':'Southampton','C':'Cherbourg','Q':'Queenstown'}), ax=axes[0])
axes[0].set_ylabel('Passengers')
axes[0].set_title('Passengers by embarkation port')
axes[0].set_xlabel('')
sns.barplot(x='Embarked', y='Survived', data=train, ci=None)
axes[1].set_xlabel('')
axes[1].set_ylabel('Survival rate')
axes[1].set_title('Survival rate by embarkation port')
axes[1].set_xticklabels(['Southampton','Cherbourg','Queenstown'])


# ### SibSp and Parch
# 
# Taking a quick look and these two variables, we can see that passengers who were alone had a lower chance to get rescued. Passengers having between 1 and 3 children or parents had an higher chance to survive, but with more than 3, the survival probability decrease. Passengers having 3 or more brothers/sisters and partner had a lower survival rate.
# 
# We can see that these two variables are similar, it can be useful to combine them in one variable (the sum of SibSp and Parch) called FamilySize, then we will understand if we can drop the two original variables.

# In[ ]:


sns.catplot(x='Parch', y='Survived', data=train, kind='bar', ci=None)

sns.catplot(x='SibSp', y='Survived', data=train, kind='bar', ci=None)


# ### PassengerId, Cabin, Ticket and Name
# 
# Let's make some considerations about the remaining variables:
# * PassengerId is clearly unexplicative of the phenomenon because we have 891 unique values (the ID of a passenger), so we can easily drop this column
# 

# In[ ]:


train['PassengerId'].nunique()


# * Cabin: we have many missing data, but in order to not lose some informations in this phase, we can try to extract the initial letter of the cabin and create some categories, we can fill di NaN values with an 'U' (Unknown) and than keep the others. Probably the first letter identifies a specific area of the Titanic, and that could be a crucial factor to get saved: we can try to understand the Pclass of passengers with Cabin data and group them.

# In[ ]:


train['Cabin'].unique()


# In[ ]:


train['Cabin'].fillna('U').apply(lambda x: x[0]).value_counts()


# * Ticket: we have 681 unique values consisting in an alphanumerical code: we can imagine that this code identifies the port of embarkation, the lot of the group of tickets and informations like that, not really relevant for the Titanic disaster considering the variables we already have. But we can notice that some passengers had the same ticket number, maybe this could be an information.

# In[ ]:


train['Ticket'].nunique()


# In[ ]:


train['Ticket'].head()


# * Name: the Name feature include the title (like Mr., Mrs., etc.) and this can be explicative for our predictive purpose; we can extract the title, the more frequent are Mr (adult men), Miss (not married woman), Mrs (married or widow women) and Master (boys/kids under 18), then we have low frequence title and we will decide how to manage them.

# In[ ]:


train['Name']


# In[ ]:


train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0].value_counts()


# ### Features Engineering
# 
# Now we are ready to play with our features in order to improve the final classification accuracy.
# 
# First of all, let's create the FamilySize feature as the Sum of SibSp and Parch: people with 1, 2 or 3 familiars aboard had an higher survival rate than alone or belonging to large families passengers.

# In[ ]:


for dataset in total:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']


# In[ ]:


sns.catplot(x='FamilySize', y='Survived', kind='bar', data=train, ci=None)
plt.xlabel('Family size')
plt.ylabel('Survival rate')
plt.title('Survival rate by family size')


# Now we can create another variable called Title: we extract the title from the name column, we assign to the most frequent categories similar titles (for example Mlle stands for Madmoiselle that in french means Miss) and to the "Other" category the rare titles; then we can drop the Name column from both the train and test sets.

# In[ ]:


for dataset in total:
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# Here we can see titles which appear less then 10 times in train and test datasets. 'Mlle' will become 'Miss', 'Ms' and 'Mme' will become 'Mrs', the remaining will be 'Other' because for example Sir and Lady are noble titles, Capt and Col seem to be military rank titles, etc.

# In[ ]:


train['Title'].append(test['Title']).value_counts()[train['Title'].append(test['Title']).value_counts()<10].index


# For example this "Ms." is 28 years-old and she is not married, so she is clearly a Miss.

# In[ ]:


train[train['Title']=='Ms']


# In[ ]:


for dataset in total:
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Sir', 'Col', 'Major', 'Don', 'Jonkheer', 'Dona', 'Capt', 'Lady', 'the Countess'],'Other')


# In[ ]:


train['Title'].value_counts()


# Then we can manage the Cabin feature using only the first letter of the cabin code and then filling the missing values with an U (unknown)

# In[ ]:


for dataset in total:
    dataset['Cabin'] = dataset['Cabin'].fillna('U').apply(lambda x: x[0])


# As we can see, some cabins are similar in term of Pclass, so we can group them: A, B, C and T were only 1st class, D and E include 2nd and 3rd class but have an high percentage of 1st class passengers, F and G were only 2nd and 3rd class.

# In[ ]:


fig, axes = plt.subplots(1,2,figsize=(12,6))
sns.barplot(x='Cabin', y='Survived', hue='Pclass', data=train, ax=axes[0], order=['A','B','C','D','E','F','G','T','U'], ci=False)
sns.countplot(x='Cabin', hue='Pclass', data=train[train['Cabin'] != 'U'], ax=axes[1], order=['A','B','C','D','E','F','G','T'])


# So we can create the CabinGroup feature, that simply groupes similar cabins.

# In[ ]:


def group_cabin(cabin):
    if cabin in ['A','B','C','T']:
        return 0
    elif cabin in ['D','E']:
        return 1
    elif cabin in ['F', 'G']:
        return 2
    elif cabin == 'U':
        return 3

for dataset in total:
    dataset['CabinGroup'] = dataset['Cabin'].apply(lambda x: group_cabin(x))


# Now that we have created these three new features, we can manage the missing values.

# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# We can fill the two missing values for the Embarked feature with the mode, the most frequent value: Southampton (S)

# In[ ]:


train['Embarked'].fillna('S', inplace=True)


# Now let's think about the Age feature: we could easily replace the missing values with median or mean, but we can do somthing better as we can use other features to impute better the Age of a passenger.

# In[ ]:


median_title_ages = dict(train.groupby('Title').median()['Age'])
median_title_ages


# In[ ]:


for dataset in total:
    for title in median_title_ages:
        dataset.loc[dataset['Title'] == title,'Age'] = dataset.loc[dataset['Title'] == title,'Age'].fillna(median_title_ages[title])


# We have one missing value in the test set for the Fare feature, we can use the Fare mean for the 3rd class.

# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


test['Fare'] = test['Fare'].fillna(train.groupby('Pclass').mean()['Fare'][3])


# Now we should not have missing values in both test and train sets

# In[ ]:


print(test.isnull().sum())
print(train.isnull().sum())


# We have two continuos variable, Age and Fare, for a classification problem, in order to remove noise and make classifiers perform better, we can discretize these two variables: we will also solve a possible outliers problem in the Fare column.

# As seen in the Age plot, Age has a significant impact in the children survival probability, so we don't need to create many categories: the most important thing is that children have their own category.

# In[ ]:


train['Age'] = pd.cut(train['Age'], bins=(-1,13,30,55,100), labels=[0,1,2,3], right=True)
test['Age'] = pd.cut(train['Age'], bins=(-1,13,30,55,100), labels=[0,1,2,3], right=True)


# For the Fare variable, we have many tickets with <15 price so, in order to make a balanced discretization, we can use these 4 custom bins.

# In[ ]:


train['Fare'] = pd.cut(train['Fare'], bins=(-1,8,15,60,800), labels=[0,1,2,3], right=True)
test['Fare'] = pd.cut(test['Fare'], bins=(-1,8,15,60,800), labels=[0,1,2,3], right=True)


# In[ ]:


train.head()


# Now let's drop columns we don't need.

# In[ ]:


for dataset in total:
    dataset.drop(['Name','Ticket','PassengerId','Cabin','Parch','SibSp'], axis=1, inplace=True)


# Lastly we need to map categorical variables into integers number, so the model can correctly read them.

# In[ ]:


for dataset in total:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'Q':1, 'C':2})
    dataset['Title'] = dataset['Title'].map({'Mr':0,'Other':1,'Master':2,'Miss':3,'Mrs':4})


# Here we have our final dataset!

# In[ ]:


train.head()


# ### Model Selection
# 
# Now that our dataset is cleaned, we need to find the best model to predict if a passenger survived or not. We will use the train dataset to find the model with the best performance, then we will submit to Kaggle the predictions from the test set.

# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[ ]:


classifiers = [['Random Forest',RandomForestClassifier()], ['AdaBoost',AdaBoostClassifier()], ['Support Vector Machine', SVC()], ['KNN',KNeighborsClassifier()], ['Naive Bayes',GaussianNB()], ['Decision Tree',DecisionTreeClassifier()]]


# We can test now 6 classification model:
# * Random Forest
# * AdaBoost
# * Support Vector Machine
# * K-Nearest Neighbors
# * Gaussian Naive Bayes
# * Decision Tree
# 
# We can evaluate them by four metrics: accuracy, recall, precision and f1 score obtained doing a cross validation with KFold, it simply divide the dataset into 10 folds that are used as validation set (the k-1 folds compose the training set) for each iteration: this is a good practice in order to avoid unbalanced or "lucky" split using, for example, the standard train_test_split.

# In[ ]:


for metric in ['accuracy','f1']:
    kfold = KFold(n_splits=10, random_state=99)
    score_mean = []
    std = []
    
    for model in classifiers:
        clf = model[1]
        cv_result = cross_val_score(clf,X,y, cv = kfold,scoring = metric)
        cv_result = cv_result
        score_mean.append(cv_result.mean())
        std.append(cv_result.std())
        
    models_evaluation = pd.DataFrame({metric: score_mean}, index=[i[0] for i in classifiers])
    print(models_evaluation.sort_values(metric, ascending=False))
    print('*'*32)


# We can see that Support Vector Machine, Random Forest Classifier and Adaboost Classifier seem to have better performances.
# Now we have to find the best hyperparameters for the three models. We can use GrideSearchCv and RandomizedSearchCv, sklearn functions that allow us to iterate the model for every (or just a part) parameter in order to maximize a statistic score (like accuracy).

# Let's start with Support Vector Classifier.

# In[ ]:


C = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
kernel = ['rbf', 'linear']
hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}
gd = GridSearchCV(estimator=SVC(), param_grid=hyper, verbose=True, scoring='accuracy')
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)


# Now that we found the best hyperparameters for our SVC model, we can predict the Survived feature for the Test set and then submit to the Kaggle Titanic competition.

# In[ ]:


pred=gd.predict(test)
test_sub = pd.read_csv('../input/titanic/test.csv')
test_sub['Survived'] = pd.Series(pred)
test_sub[['PassengerId', 'Survived']].to_csv('pred_submission.csv', index=False, encoding='utf-8')


# We can do the same with Random Forest Classifier, the only difference is that we are searching best parameters not for all possible combinations, but only for a random part of them.

# In[ ]:


params ={      'criterion': ['entropy', 'gini'],
               'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
               'max_features': ['auto', 'sqrt','log2', None],
               'min_samples_leaf': [4, 6, 8, 12],
               'min_samples_split': [5, 7, 10, 14],
               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}

clf = RandomForestClassifier()
rs = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 80, 
                               cv = 4, verbose= 5, random_state= 101, n_jobs = -1, scoring='accuracy')
rs.fit(X,y)
print(rs.best_score_)
print(rs.best_estimator_)


# In[ ]:


pred=rs.predict(test)
test_sub = pd.read_csv('../input/titanic/test.csv')
test_sub['Survived'] = pd.Series(pred)
test_sub[['PassengerId', 'Survived']].to_csv('pred_submission_1.csv', index=False, encoding='utf-8')


# Finally we can do the same for AdaBoost classifier

# In[ ]:


params ={'n_estimators': [10,50,100,200,300,400,500],'learning_rate': [0.01,0.1,0.2,0.4,0.6,0.8,1.0]}

clf = AdaBoostClassifier()
gs = GridSearchCV(estimator = clf, param_grid = params,verbose=True, scoring='accuracy')
gs.fit(X,y)
print(gs.best_score_)
print(gs.best_estimator_)


# In[ ]:


pred=gs.predict(test)
test_sub = pd.read_csv('../input/titanic/test.csv')
test_sub['Survived'] = pd.Series(pred)
test_sub[['PassengerId', 'Survived']].to_csv('pred_submission_2.csv', index=False, encoding='utf-8')

