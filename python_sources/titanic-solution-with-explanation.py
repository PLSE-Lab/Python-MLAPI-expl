#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Logistic Regression with Python in Machine Learning - Titanic Dataset
# Logistic Regression is a Classification Algorithym used to assign observations to a discrete set of classes. In linear we having continuous observations but in logistic regresssion we have 0 and 1 and True / False. logistic regression transforms it output using the logistic sigmid function to return a probability value. if probability is greater than 0.5 will predict it towards 1. Ex - rain forecasting, fraud detection, cancer detection, spam mail or not.
# 
# # TITANIC DATA INFORMATION -
# The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died.
# 
# # Lets go ahead and build a model which can predict if a passenser is gonna survive

# In[ ]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Importing train and test data.

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


# before doing any analysis we will check our data that it imported properly or not.

train_data.head()


# In[ ]:


test_data.shape


# In[ ]:


train_data.shape


# In[ ]:


# as can see train_data having 891 unique values with 12 feature. and test_data having 418 unique values with 11 feature.
# in test_data one feature is not included "survived" which is DV and we need to identify.
# for doing further few cleaning and munging i am combining both data.


# In[ ]:


titanic = [train_data, test_data]


# In[ ]:


# checking train columns

train_data.columns


# # Explanation of all features
# 
# ### PassengerId - Unique ID of the passenger
# 
# ### Survived - Survived (1) or died (0)
# 
# ### Pclass - Passenger's class (1st, 2nd, or 3rd)
# 
# ### Name- Passenger's name
# 
# ### Sex- Passenger's sex
# 
# ### Age - Passenger's age
# 
# ### SibSp - Number of siblings/spouses aboard the Titanic
# 
# ### Parch - Number of parents/children aboard the Titanic
# 
# ### Ticket- Ticket number
# 
# ### Fare - Fare paid for ticket
# 
# ### Cabin - Cabin number
# 
# ### Embarked - Where the passenger got on the ship (C - Cherbourg, S - Southampton, Q = Queenstown)

# In[ ]:


train_data.info()

print("------------------------------------------")

test_data.info()

# Information collected from info - 
#Seven features are integer or floats. Six in case of test dataset.
#Five features are strings (object).


# In[ ]:


# from above ingormation can see there is null value in data. lets check by once again by using isnull 
train_data.isnull().sum() 

# Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# Cabin > Age are incomplete in case of test dataset.


# In[ ]:


test_data.isnull().sum()


# In[ ]:


# Checking some statistics information about data

train_data.describe()


# #  Information Gained from describe -
# - Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
# - Survived is a categorical feature with 0 or 1 values.
# - Around 38% samples survived representative of the actual survival rate at 32%.
# - Most passengers (> 75%) did not travel with parents or children.
# - Nearly 30% of the passengers had siblings and/or spouse aboard.
# - Fares varied significantly with few passengers (<1%) paying as high as $512.
# - Few elderly passengers (<1%) within age range 65-80.

# In[ ]:


train_data.describe(include=['O'])


# # Understanding of describe from categorical feature
# 
# - Names are unique acroos the dataset (count = 891)
# - Sex variable as two possible values with 65% male(top = male, freq = 577 / count = 891)
# - Cabin values have several duplicate across samples. freq = 4
# - Embarked take 3 possible values. S port used by most passenger (top = s)
# -Ticket feature has high ratio(22 %) of duplicate values (unique = 681 , count = 891)

# In[ ]:


test_data.describe()


# In[ ]:


test_data.describe(include=['O'])


# # # Understanding of describe from categorical feature
# -Names are unique across the dataset (count=unique=418)
# 
# -Sex variable as two possible values with 63% male (top=male, freq=266/count=418).
# 
# -Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
# 
# -Embarked takes three possible values. S port used by most passengers (top=S)
# 
# -Ticket feature has high ratio (13%) of duplicate values (unique=363).

# # Assumptions based on Data Analysis
# ### now i would like to check coorelation of all features with survival to check is there any linear correlation or not. 
# ### is there any multicollineaity between predictors.
# ### also fill missing value in some features.
# ### do required encoding for categorical feature.
# ### if needed do feature scaling too
# ### as ticket feature contains high ratio of duplicates(22%) may be can drop this.
# ### cabin feature may be dropped as it having many null values both in training and test dataset.
# ### PassengerId not useful for giving any information so drop this.
# ### Name feature also not directly corelated to survival, so can drop this too.
# ### can create some new feature based on class, Fare Range, family which include detail of person and sibling . children.
# ### can crete new feature as "Title" and adjust all name based on the title.
# ### classify category based on female as we know already female survive mostly.
# ### need to find out range of children age who survived by classifying them.
# ### we knew also upper class passenger survived mostly, also high fair paid passenger survived mostly.
# 

# In[ ]:


# Lets create some pivot to see more detail analysis


# In[ ]:


train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

# can see from above pivot based on Pcassenger Class first class passenger survived around 62%.
# In[ ]:


train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# as shown from above pivot female are most likely to survived around(74%)of given data.


# In[ ]:


train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False ).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# passenger who is trvelling with one sibling have chances to survived (53%) compared to passenger who have more than one family member.


# In[ ]:


train_data[['Parch', 'Survived']].groupby(['Parch'], as_index = False ).mean().sort_values(by = 'Survived', ascending = False)


# # Visualizaton of Data

# In[ ]:


# Correlating numerical features 
#I will use Histogram chart for analyzing continuous numerical variable like age.


# In[ ]:


g = sns.FacetGrid(train_data, col = 'Survived')
g.map(plt.hist, 'Age', bins = 20)


# # Explanation of age histogram based on survival
# - Survived = 1 and nonsurvived = 0
# - age of 80 survived.
# - non survival passenger are mostly from age group 15-30.
# - maximum number of passenger who boarded from age group 15-35 age range.
#     

# # Correlating numerical and ordinal features
# We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values

# In[ ]:


g = sns.FacetGrid(train_data, col = 'Survived', row = 'Pclass')
g.map(plt.hist, 'Age', bins = 20)
g.add_legend()


# # Explanation of above histogram 
# - Survival number is much higher in case of PClass 1.
# - most unsurvived passenger belongs to Pclass 3.
# - mostly infant passenger belongs to Pclass 1, 2 and survived 
# - major number of passenger between age group 15-30 boarded in PClass 3 and non survived too.

# # Correlating Categorical Feature

# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
grid.add_legend()


# # Observations from above correlation
# - In this graph checked passenger board place (C - Cherbourg, S - Southampton, Q = Queenstown) correlation with survival
# - passenger who boarded from Cherbourg and part of passenger class 1have more male survival rate.
# - from Southampton more female survived with passenger class 1,2.
# - in all board place female have higher survival rate.

# # Correlating categorical and numerical features
# # Bar plot

# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', col='Survived')
grid.map(sns.barplot, 'Sex', 'Fare')
grid.add_legend()


# # Assumptions based on above graph between survival and Embarked
# - Cherbourg passenger paid higher fare and survived to. Female have higher survival than male.
# - Queenstown passenger have equal proprotion of survival and non survival.
# - Port of embarkation correlates with survival rates. and its shows correlation with survival****

# # Removing some feature
# Now after understanding data by making differnt assumptions checking through visualization and correlation we got to know which feature is non relevant with survival and can remove them.
# 
# 

# In[ ]:


train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_data, test_data]


# In[ ]:


train_data.columns


# In[ ]:


test_data.columns


# In[ ]:


combine[0].shape


# In[ ]:


combine[1].shape


# ### can see ticket and cabin removed from data

# # Creating Some new feature 

# In[ ]:


# Title feature added
# can see majorly passenger use Miss, Mrs, Mr, master for their title.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# In[ ]:


# We can replace many titles with a more common name or classify them as Rare.

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    
    
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# We can convert the categorical titles to ordinal.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[ ]:


train_data.head(10)


# In[ ]:


test_data.head(3)


# In[ ]:


# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
combine = [train_data, test_data]
train_data.shape, test_data.shape


# In[ ]:


# Encoding categorical feature into Numeric

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train_data.head()


# # Filling Missing value

# In[ ]:


train_data.corr().sort_values(by = "Survived",ascending = True)


# In[ ]:


corr = train_data.corr()
print (corr['Survived'].sort_values(ascending=False)[:10], '\n')
print (corr['Survived'].sort_values(ascending=False)[-10:])


# In[ ]:


#correlation matrix
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr,  annot=True,annot_kws={'size': 15})


# In[ ]:


s = corr.unstack()
s


# In[ ]:


grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[ ]:


# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

           
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:



train_data.head()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


# Let us create Age group and determine correlations with Survived.

train_data['AgeGroup'] = pd.cut(train_data['Age'], 5)
train_data[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True)


# In[ ]:


# Let us replace Age with ordinals based on these groups.

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']


# In[ ]:


train_data.head()


# In[ ]:


# Now removing agegroup as its not required after converting age into different groups
train_data = train_data.drop(['AgeGroup'], axis=1)
combine = [train_data, test_data]
train_data.head()


# ### Create new feature combining existing features(Parch & SibSp)
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# We can create another feature called IsAlone.

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_data, test_data]


# In[ ]:


train_data.head()


# In[ ]:


# We can also create an artificial feature combining Pclass and Age.

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[ ]:


# Completing a categorical feature
# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

freq_port = train_data.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
   dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
   
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_data.isnull().sum()
# all missing value filled


# In[ ]:


# Converting categorical feature to numeric
# We can now convert the EmbarkedFill feature by creating a new numeric Port feature.

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


# Filling missing vale in fare by median
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
test_data.head()


# In[ ]:


train_data['FareGroup'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(by='FareGroup', ascending=True)


# In[ ]:


# Convert the Fare feature to ordinal values based on the FareGroup.

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_data = train_data.drop(['FareGroup'], axis=1)
combine = [train_data, test_data]
    


# In[ ]:


train_data.head(10)


# In[ ]:


test_data.head(10)


# # Algorithym application / Model Building

# Now all data cleaning, wrangling part done. our data is ready to build model. as in this data we are trying to find out correlation of other feature with survived. and also training our model with training dataset. so its classification and regression problem.
# So I will use some model app;ication on this dataset for checking which one is working better.

# In[ ]:


X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# # Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# # Support Vector Machines
# 

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# # KNN k-Nearest Neighbors algorithm 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# # Gaussian Naive Bayes

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# # Decision Tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# # Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# # Model evaluation
# We can now rank our evaluation of all the models to choose the best one for our problem.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'KNN', 
              'Naive Bayes', 'Decision Tree','Random Forest'], 
             
    'Score': [acc_log,acc_svc, acc_knn,  acc_gaussian, 
              acc_decision_tree,acc_random_forest]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


submission.to_csv('/kaggle/working/submission.csv', index=False)  


# In[ ]:




