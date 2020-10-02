#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival - Exploration + Baseline Model
# 
# This is a simple notebook on exploration and baseline model to predict who will survive the sinking of the Titanic
# 
# ## **Contents**   
# [1. Load Data](#1)    
# [2. Data Exploration](#2)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.1 Basic Data Info](#2.1)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.2 Feature Distributions](#2.2)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.3 Feature Creation/Deletion](#2.3)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.4 Impute Missing Data](#2.4)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.5 Number Conversions](#2.5)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.6 Feature Extraction](#2.6)
# 
# &nbsp;&nbsp;&nbsp;&nbsp; [2.7 Applicants Contract Type](#2.7)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.8 Education Type and Occupation Type](#2.8)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.9 Organization Type and Occupation Type](#2.9)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.10 Walls Material, Foundation and House Type](#2.10)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.11 Amount Credit Distribution](#2.11)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.12 Amount Annuity Distribution - Distribution](#2.12)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.13 Amount Goods Price - Distribution](#2.13)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.14 Amount Region Population Relative](#2.14)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.15 Days Birth - Distribution](#2.15)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.16 Days Employed - Distribution](#2.16)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.17 Distribution of Num Days Registration](#2.17)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.18 Applicants Number of Family Members](#2.18)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.19 Applicants Number of Children](#2.19)  
# [3. Exploration - Bureau Data](#3)  
# &nbsp;&nbsp;&nbsp;&nbsp; [3.1 Snapshot - Bureau Data](#3) 
# 
# 
# 
# 
# 
# ## <a id="1">1. Load Data </a>

# In[ ]:


'''
# Load Python 3 packages and retrieve Titanic data (libraries installed are
   # defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import lightgbm as lgb

## Function to Hot-Code a categorical variable_
    # Takes as parameters 1) a dataframe 2) a string variable with the column name to recode
    # Leaves in tack the initial variable that was recoded

def HotC(dframe,col):   # Function to Hot-Code a categorical variable
        
    if not(isinstance(dframe,pd.DataFrame)):
        print('!!ERROR!! The first variable in the HotC function must be a dataframe')
        return
    if not(isinstance(col,str)):
        print('!!ERROR!! The second variable in the HotC function must be a string representing a column in the dataframe')
        return
    df2=pd.DataFrame(dframe[col].str.get_dummies())
    df3=pd.concat([dframe,df2],axis=1)

    return df3


path='../input/titanic-machine-learning-from-disaster/'

train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + 'test.csv')

# Write out Data sets for download
#train_df.to_csv('train_df_raw.csv', index = False)
#test_df.to_csv('test_df_raw.csv', index = False)

train_df.info()
test_df.info()

# Any results you write to the current directory are saved as output.


# In[ ]:





# ## <a id="2">2.  Data Exploration </a> 
# ### &nbsp;&nbsp;  <a id="2.1">2.1  Basic Data Info </a>
# 

# In[ ]:


# Set display
pd.options.display.max_columns=15
pd.options.display.max_rows=892

# Some data snapshoots
des='''DESCRIPTION OF FEATURES:
survival:    Survival 
PassengerId: Unique Id of a passenger. 
pclass:    Ticket class     
sex:    Sex     
Age:    Age in years     
sibsp:    # of siblings / spouses aboard the Titanic     
parch:    # of parents / children aboard the Titanic     
ticket:    Ticket number     
fare:    Passenger fare     
cabin:    Cabin number     
embarked:    Port of Embarkation'''
print(des)
print('\n')
print('SNAPSHOOT OF TRAIN_DF')
train_df.info()
print('\n'+'SNAPSHOT OF TEST_DF')
test_df.info()
print('\n')

print('BASIC DESCRIPTION')
print(train_df.describe())
print('\n')

print('SNAPSHOT OF FIRST 8 RECORDS')
print(train_df.head(8))
print('\n')

# List missing values
print('MISSING VALUE SUMMARY')
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))
train_df.info()
test_df.info()


# ### &nbsp;&nbsp;  <a id="2.2">2.2  Feature Distributions </a> 

# In[ ]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# In[ ]:


FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

print('Interaction of Embarked & Sex')


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train_df)
print('Further breakdown of Class')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ### &nbsp;&nbsp;  <a id="2.3">2.3  Feature Creation/Deletion </a>

# In[ ]:


'''# Create/delete some features'''
train_df.info()
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

axes = sns.factorplot('relatives','Survived', 
                      data=train_df, aspect = 2.5, )
print('Impact of Traveling with Relatives')
train_df['not_alone'].value_counts()
train_df.info()
test_df.info()

# Delete PassengerId from train_df (not there, so does not need to be deleted)
train_df = train_df.drop(['PassengerId'], axis=1)  
# Drop Passenger Name
#train_df = train_df.drop(['Name'], axis=1)   # Don't need to drop; not there


# In[ ]:


# Look at correlation between key variables

'''
print('Correlation of Pclass & relatives')
#print(train_df.corr().loc['relatives','Pclass'])
import scipy.stats
print(scipy.stats.pearsonr(train_df['Pclass'].values,train_df['relatives'].values)[0],'    --using scipy.stats pearsonr')
print(train_df.corr().loc['Pclass','relatives'],'    --using pandas pearsonr \n')

print('Correlation of relatives & Age')
print(train_df.corr().loc['relatives','Age'])
print('\n')
'''
cor_dataset=train_df[['Survived','Pclass','Age','Fare','relatives','not_alone']]
print('Correlation Matrix')

print(cor_dataset.corr())
print(sns.heatmap(cor_dataset.corr()))


# ### &nbsp;&nbsp;  <a id="2.4">2.4  Missing Data Imputations </a>

# In[ ]:


'''
# Drop PassengerId from the training set
train_df = train_df.drop(['PassengerId'], axis=1)
'''
#train_df = train_df.drop(['PassengerId'], axis=1)   # Don't need to drop; not there
# Convert deck first to alpha (A-), and then to numberic
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[ ]:


'''## Replace missing data in Age by using
   # random numbers based on the mean age value in regards to the standard deviation and is_null
'''
data = [train_df, test_df]
for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()     # check there are no null values
train_df.info()
test_df.info()


# In[ ]:


'''
## Fill the 2 embarked missing features with the most common values from embarked
'''
# Determine the most frequent value
train_df['Embarked'].describe()

# Fill missing values with 'S'
common_value = 'S'
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_df.info()
test_df.info()


# ### &nbsp;&nbsp;  <a id="2.5">2.5  Number Conversions </a>

# In[ ]:


'''
## Convert Fare from float to int64 using 'astype()'
'''
data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df.info()


# ### &nbsp;&nbsp;  <a id="2.6">2.6  Feature Extraction </a>

# In[ ]:


'''
## Use the Name feature to extract the titles from the Name to build a new feature
'''
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)




# In[ ]:


'''
# Hot-Code Sex feature
'''
train_df=HotC(train_df,'Sex')
train_df=train_df.drop(['Sex'], axis=1)

test_df=HotC(test_df,'Sex')
test_df=test_df.drop(['Sex'], axis=1)

pd.options.display.max_columns=20
print(train_df.head(10))
print(test_df.head(10))


# In[ ]:


'''
# Drop Ticket from the data set
'''
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# In[ ]:


'''
## Hot-Code Embarked and re-name columns
'''
train_df=HotC(train_df,'Embarked')
test_df=HotC(test_df,'Embarked')

#Rename Embarked Hot-codes
train_df=train_df.rename(index=str, columns={'C':'Emb_C','Q':'Emb_Q','S':'Emb_S'})
test_df=test_df.rename(index=str, columns={'C':'Emb_C','Q':'Emb_Q','S':'Emb_S'})

#Drop Embarked Column
train_df=train_df.drop(['Embarked'], axis=1)
test_df=test_df.drop(['Embarked'], axis=1)

print(train_df.head(10))
print(test_df.head(10))


# In[ ]:





# In[ ]:


'''
## Create Categories for Age Feature
'''

data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed
print('distribution of train_df')
train_df['Age'].value_counts()


# In[ ]:


'''
## Create categories for Fare
'''
data = [train_df, test_df]
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    


# In[ ]:


'''
## Create some additional variables
'''
# Age X Class
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

# Fare per Person
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# Let's take a last look at the training set, before we start training the models.
train_df.head(10)

print(train_df.head(10))
print(test_df.head(10)) 


# In[ ]:


'''## Try several algorithms to find the best'''

print("Results") # Output Title

## Fit Models to compare effectiveness
# Define testing dataframes
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

#print(X_test.head(10))

#SGD-Stochastic Gradient Descent
sgd = linear_model.SGDClassifier(max_iter=50, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#Logistic Regression:
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# K Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    
# Perceptron:
perceptron = Perceptron(max_iter=10)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear Support Vector Machine
linear_svc = LinearSVC(max_iter=2000)
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

'''
# lgb_light
# params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          #'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          #'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          #'min_split_gain':.01, 'min_child_weight':1}

# lgb_light = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
lgb_light = lgb
lgb_light.fit(x_train,Y_train)
Y_pred = lgb_light.predict(X_test)
acc_lgb_light = round(lgm_light.score(X_train, Y_train) * 100, 2)
'''
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

results = pd.DataFrame({
    'Model': ['LinearSVC','KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)

  
    


# In[ ]:


print (train_df.head(10))
print(test_df.head(10))


# In[ ]:


'''Conduct a K-Fold cross validation'''
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[ ]:


# Check feature importance of the random forest
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(15))
# Plot results
importances.plot.bar()


# In[ ]:



'''Drop the least important features (Parch, Emb_S, Emb_C,Emb_Q)'''
train_df  = train_df.drop("Emb_Q", axis=1)
test_df  = test_df.drop("Emb_Q", axis=1)

train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)

train_df  = train_df.drop("Emb_C", axis=1)
test_df  = test_df.drop("Emb_C", axis=1)

train_df  = train_df.drop("Emb_S", axis=1)
test_df  = test_df.drop("Emb_S", axis=1)

train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)

print(train_df.head(10))
print(test_df.head(10))


# In[ ]:


train_df  = train_df.drop("Sex", axis=1)
test_df  = test_df.drop("Sex", axis=1)

print(X_train.head(2))
print(Y_train.head(2))
print(X_test.head(2))


# In[ ]:


# Retrain random forest

# Define testing dataframes
# Define testing dataframes
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
print(round(acc_random_forest,2,), "%")
print(importances.head(15))
# Plot results
importances.plot.bar()


# In[ ]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")

