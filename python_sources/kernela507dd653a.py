#!/usr/bin/env python
# coding: utf-8

# # **Titanic dataset - Logistic Regression - SciKit Learn - XGBoost**

# ### 1. **Problem Statement: Predict if a person would live or die on the titanic**

# In[ ]:


#Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import regex as re
from plotly import tools

import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)


# ## **2. Data exploration**

# In[ ]:


#Import and take a look at the Data
train=pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# In[ ]:


train.describe()


# In[ ]:


#Draw up some visuals of survival correlations
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15,10))
sns.barplot(x="Pclass", y="Survived", data=train, ax=axs[0,0])
sns.barplot(x="Sex", y="Survived", data=train, ax=axs[0,1])
sns.barplot(x="SibSp", y="Survived", data=train, ax=axs[0,2])
sns.barplot(x="Parch", y="Survived", data=train, ax=axs[1,0])
sns.barplot(x="Embarked", y="Survived", data=train, ax=axs[1,1])
sns.countplot(x="Survived", data=train, ax=axs[1,2])


# Quick notes:
# * We can see that surival correlates with Class, Sex, Famlilal ties, also port "C" (most likely lots of women or high class) seems to have a larger surival rate.
# * Average age is 29, class pop leans towards 3, Average familal tie is less than 1. Survival rate 0.38%

# In[ ]:


#Lets check missing values
print(pd.isnull(train).sum())
print('Percent of missing "Cabin" records is %.2f%%' %((train['Cabin'].isnull().sum()/train.shape[0])*100))
print('Percent of missing "Age" records is %.2f%%' %((train['Age'].isnull().sum()/train.shape[0])*100))
print('Percent of missing "Embarked" records is %.2f%%' %((train['Embarked'].isnull().sum()/train.shape[0])*100))


# In[ ]:


#Lets check missing values
print(pd.isnull(test).sum())
print('Percent of missing "Cabin" records is %.2f%%' %((test['Cabin'].isnull().sum()/test.shape[0])*100))
print('Percent of missing "Age" records is %.2f%%' %((test['Age'].isnull().sum()/test.shape[0])*100))
print('Percent of missing "Embarked" records is %.2f%%' %((test['Embarked'].isnull().sum()/test.shape[0])*100))


# In[ ]:


for dataframe in [train, test]:
    dataframe.set_index('PassengerId', inplace=True)
    
train.head()


# ### **Dealing with missing records.**
# 

# **Alterations**
# * Cabin - Lets explore this a bit more. Could possibly be related to Pclass and survivability.
# * Age: -81% data should be enough to guess the remainders.
# * Ticket - There is a correlation with passenger class and possibly Cabin? 
# * Embarked - Couldnt possible matter where you came from?
# 
# 

# ## 2.1 - Cabin

# In[ ]:



#Extracting first letter to determine cbin grade 
train['Cabin'] = train['Cabin'].apply(lambda x: "other" if pd.isna(x) else x[0])


# In[ ]:



fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

sns.countplot(data=train.loc[train['Cabin'] == 'other'], hue ='Cabin', x='Pclass', ax=ax[0,0]).set_title('Unknown Cabin')
sns.countplot(data=train.loc[train['Cabin'] != 'other'], hue ='Cabin', x='Pclass', ax=ax[0,1]).set_title('Cabin')
sns.countplot(data=train.loc[train['Cabin'] == 'other'], hue ='Survived', x='Pclass',  ax=ax[1,0])
sns.countplot(data=train.loc[train['Cabin'] != 'other'], hue ='Survived', x='Pclass',  ax=ax[1,1])


#  Seems that people with documented cabins mostly comes from Pclass 1, and overall has higher survivability rate than the average, we will make cabin into a known = 1 or not know = 0 type of column.

# In[ ]:


for dataframe in [train, test]:
    dataframe['Cabin'] = dataframe['Cabin'].apply(lambda vector: 0 if vector == 'other' else 1)
    
train.head()


# ## 2.2 Embarked

# In[ ]:



fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

sns.countplot(data=train, hue ='Survived', x='Embarked', ax=ax[0]).set_title('Embarked Survived Ratio')
sns.countplot(data=train, hue ='Pclass', x='Embarked', ax=ax[1]).set_title('Embarked Pclass')


# In[ ]:


#"C" includes mostly class 1 people and surive ratio seems to be correlated with class here, dropping this column.

for dataframe in [train, test]:
    dataframe.drop('Embarked', inplace=True, axis=1)


# ## 2.3 Name - Title - Social Rank

# In[ ]:


#Exploring titles and social ranks omboard, as we know members of the english aristocracy joined the ship, is it possible to combine this and fare?.
#Structure: Surname, title. name -
for dataframe in [train, test]:
    dataframe['Title']= dataframe['Name'].apply(lambda name:re.search(r'\,(.*?)\.',name).group(0))
    #dataframe.drop('Name', inplace=True, axis= 1)


# In[ ]:




fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
sns.countplot(data=train, x='Title', hue='Survived', ax=ax[0])
sns.countplot(data=train, x='Title', hue='Pclass', ax=ax[1])


# In[ ]:


train.head()

#Keeping Title for now, these dropping Name + Ticket 
for dataframe in [train, test]:
    dataframe.drop(['Name', 'Ticket'], inplace=True, axis=1)


# In[ ]:


#transforming special Titles into something more usable

for dataframe in [train, test]:
   dataframe['Title'] = dataframe['Title'].apply(lambda vector: vector if vector in [', Mr.', ', Mrs.' , ', Miss.', ', Master.'] else "Special" )

train.head()


# ## 2.4 - Fare

# In[ ]:


#lets get a dist plot going for fare possibly we could use this either as a 0 - 1 delimer or bucketize 

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))

sns.distplot(train['Fare'], ax=ax[0,0]).set_title("Total")

sns.distplot(train.loc[train['Pclass']==1]['Fare'], ax=ax[0,1], color='Blue').set_title("Class 1")
             
sns.distplot(train.loc[train['Pclass']==2]['Fare'], ax=ax[1,0], color='Orange').set_title("Class 2")
             
sns.distplot(train.loc[train['Pclass']==3]['Fare'], ax=ax[1,1], color='Green').set_title("Class 3")


# In[ ]:



#Within class one, there is a clear corelation between higher survival ratio and fare. 
#On average the higher paying persons survied, most likely due to skewed stats (500 fare)

train.loc[train['Pclass'] != 1]

fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
sns.barplot(x="Pclass", y="Fare", hue="Survived", data=train, ax=ax[0])
sns.barplot(x="Sex", y="Fare",  hue="Survived", data=train,  ax=ax[1]);


# In[ ]:


#lets get binning,
for dataframe in [train, test]:
    dataframe.loc[ dataframe['Fare'] <= 10, 'Fare']  = 0
    dataframe.loc[(dataframe['Fare'] > 10) & (dataframe['Fare'] <= 15), 'Fare'] = 1
    dataframe.loc[(dataframe['Fare'] > 15) & (dataframe['Fare'] <= 22), 'Fare'] = 2
    dataframe.loc[(dataframe['Fare'] > 22) & (dataframe['Fare'] <= 30), 'Fare'] = 3
    dataframe.loc[(dataframe['Fare'] > 30) & (dataframe['Fare'] <= 40), 'Fare'] = 4
    dataframe.loc[(dataframe['Fare'] > 40) & (dataframe['Fare'] <= 60), 'Fare'] = 5
    dataframe.loc[(dataframe['Fare'] > 60) & (dataframe['Fare'] <= 100), 'Fare'] = 6
    dataframe.loc[(dataframe['Fare'] > 100) & (dataframe['Fare'] <= 200), 'Fare'] = 7
    dataframe.loc[ dataframe['Fare'] > 200, 'Fare'] = 8 ;


# In[ ]:


#fixing missing fare in test, set it to the avg (pclass 3 -)
test.loc[test['Fare'].isnull(), 'Fare'] = 0


# In[ ]:


sns.countplot(x='Fare', hue='Survived', data=train)
#looks good lets continue.


# In[ ]:


train.head()


# ## 2.6 Families

# In[ ]:


Families = train.loc[(train['SibSp']>0) | (train['Parch'] > 0)]
Singles = train.loc[(train['SibSp']==0) & (train['Parch'] == 0)]

print('Percent of surving when part of a "family" is %.2f%%' %(Families.loc[Families['Survived']==1]['SibSp'].count() / Families['SibSp'].count()))
print('Percent of surving when single is %.2f%%' %(Singles.loc[Singles['Survived']==1]['SibSp'].count() / Singles['SibSp'].count()))
 
print(Families.shape)
print(Singles.shape)

fig, ax =plt.subplots(1,2)
sns.countplot(x='Survived', data = Families, ax=ax[0]).set_title('Families')
sns.countplot(x='Survived', data = Singles, ax=ax[1]).set_title('Singles')


# In[ ]:


#Create another column with the number of famial ties and complete remove SibSp and Parch,

for dataframe in [train, test]:
    dataframe['FamMemb'] = dataframe.apply(lambda column: column[['SibSp']] + column['Parch'], axis=1)
    dataframe.drop(['Parch','SibSp'],axis = 1, inplace = True)
    
train.head()


# In[ ]:


sns.countplot(data=train, x='FamMemb', hue='Survived') 


# In[ ]:


#Lets get ready to bucketize the last data
#overall Age distribution 

group_labels = ['distplot']
fig = ff.create_distplot([train.loc[train['Age'].notnull()]['Age'].values], ['Age'])
py.iplot(fig, filename='Basic Distplot')


# ## 2.7 - Age

# In[ ]:



fig, ax = plt.subplots(ncols=3 ,  figsize=(20, 5))
              
sns.barplot(x="Sex", y="Age", hue="Title", data=train, ax=ax[0])
sns.barplot(x="Sex", y="Age", hue="Fare", data=train,  ax=ax[1]);
sns.barplot(x="Sex", y="Age", hue="Pclass", data=train, ax=ax[2])


# In[ ]:


#Fixing Special case title in test.
test.loc[test.index == 980, 'Age'] = 22

#Define function to return a random age number based on class, title and sex. 
def f(row, dataframe):
    return np.random.choice(dataframe.loc[(dataframe['Sex'] == row['Sex']) & (dataframe['Pclass'] == row['Pclass']) & (dataframe['Title'] == row['Title']) & (dataframe['Age'].notnull())]['Age'], 1)[0]


for dataframe in [train, test]:
    dataframe.loc[dataframe['Age'].isnull(), 'Age'] = dataframe.loc[dataframe['Age'].isnull()].apply(f, args=(dataframe, ), axis = 1)
   


# In[ ]:


#Lets get ready to bucketize the last data
#overall Age distribution 

group_labels = ['distplot']
fig = ff.create_distplot([train['Age'].values], ['Age'])
py.iplot(fig, filename='Basic Distplot')


# In[ ]:


#last step lets bucketize Age:

for dataframe in [train, test]:
    dataframe.loc[ dataframe['Age'] <= 10, 'Age']  = 0 #children
    dataframe.loc[(dataframe['Age'] > 10) & (dataframe['Age'] <= 15), 'Age'] = 1
    dataframe.loc[(dataframe['Age'] > 15) & (dataframe['Age'] <= 20), 'Age'] = 2
    dataframe.loc[(dataframe['Age'] > 20) & (dataframe['Age'] <= 30), 'Age'] = 3
    dataframe.loc[(dataframe['Age'] > 30) & (dataframe['Age'] <= 40), 'Age'] = 4
    dataframe.loc[(dataframe['Age'] > 40) & (dataframe['Age'] <= 50), 'Age'] = 5
    dataframe.loc[ dataframe['Age'] > 50, 'Age'] = 6 ;
    


# In[ ]:



group_labels = ['distplot']
fig = ff.create_distplot([train['Age'].values], ['Age'])
py.iplot(fig, filename='Basic Distplot')


# ## 2.8 Numeration of last variables.

# In[ ]:


#lets numerize the last values and plug this into some ML models
#create new column
for dataframe in [train, test]:
    dataframe['Female'] = dataframe.apply(lambda column: 1 if column['Sex'] == 'female' else 0, axis=1)
    dataframe.drop('Sex', inplace=True, axis= 1)


# In[ ]:


#last step lets bucketize Age:

for dataframe in [train, test]:
    dataframe.loc[ (dataframe['Title'] == ", Master."), 'Title']  = 0
    dataframe.loc[(dataframe['Title'] == ", Miss.") , 'Title'] = 1
    dataframe.loc[(dataframe['Title'] == ", Mr.") , 'Title'] = 2
    dataframe.loc[(dataframe['Title'] == ", Mrs.") , 'Title'] = 3
    dataframe.loc[(dataframe['Title'] == "Special") , 'Title'] = 4
    


# ## **3. Model selection and prediction**
# 
# Normalization.

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


# In[ ]:


#Split data into target feature and train data.
target = train.iloc[:,0]
train.drop('Survived', inplace=True, axis=1)

cols = train.columns


# In[ ]:


#Using max min scaler as our outliers are fixed with binning.
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)


# In[ ]:



from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

#K-Folds cross-validato
#Provides train/test indices to split data in train/test sets. 
#Split dataset into k consecutive folds (5 is default).
kf = KFold(n_splits = 5, random_state = 1)

#Set hyperparameters, lets iterate over a large range.
xgb_paramaters = {'max_depth' : [1, 2, 3, 4], 'learning_rate' : [0.05, 0.1, 0.15, 0.2], 'n_estimators' : [50, 100, 200, 300], 'n_jobs' : [-1], 'random_state' : [1]}
xgb = XGBClassifier()


# In[ ]:



from sklearn.metrics import classification_report


scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    #using Gridsearch to estimate the best combination of hyperparameter values.
    gs_xgb = GridSearchCV(xgb, xgb_paramaters, n_jobs = -1, cv = kf, scoring = 'roc_auc')
    gs_xgb.fit(train, target)
    

    print("Best parameters set found on development set:")
    print(gs_xgb.best_score_)
    print(gs_xgb.best_params_)
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = target, gs_xgb.predict(train)
    print(classification_report(y_true, y_pred))
    print()


# ### **4. Summary and data submission**

# In[ ]:


prediction = gs_xgb.predict(test)


# In[ ]:


index = pd.RangeIndex(start=892, stop=1310, step=1)


# In[ ]:



test_index = pd.DataFrame(test)
submission = pd.DataFrame({'Survived' : prediction}, index = index)
submission.to_csv('submission.csv', index_label = ['PassengerId'])


# In[ ]:



