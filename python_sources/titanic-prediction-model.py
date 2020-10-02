#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")
df_gender_submission=pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.shape


# In[ ]:


df_train.describe()


# We can see that there is NULL values in the age. Lets replace NULL values in the age with median value of this column

# In[ ]:


df_train['Age'].fillna(df_train['Age'].median(),inplace=True)


# ### Visualizations

# In[ ]:


df_train['Died'] = 1-df_train['Survived']


# In[ ]:


df_train.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar', figsize=(5, 5),
                                                          stacked=True, color=['b','r']);


# So it is clear from the above that more females were survided as compare to males. Now let's again visualize this as percentage

# In[ ]:


df_train.groupby('Sex').agg('mean')[['Survived','Died']].plot(kind='bar', figsize=(5, 5),
                                                          stacked=True, color=['g','r']);


# In[ ]:


fig = plt.figure(figsize=(5, 5))
sns.violinplot(x='Sex', y='Age', 
               hue='Survived', data=df_train, 
               split=True,
               palette={0: "r", 1: "g"}
              );


# From the above it is clear that
# 1. The less number of passengers who are young [age less than 10] are died it means that children travelling were saved first
# 2. Passengers whose age between 20-30 are died so it is irrespective of Sex of the passenger

# In[ ]:


figure = plt.figure(figsize=(20, 7))
plt.hist([df_train[df_train['Survived'] == 1]['Fare'], df_train[df_train['Survived'] == 0]['Fare']], 
         stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();


# Now it is clear from the above that passengers with low fare are less survived

# In[ ]:


plt.figure(figsize=(20, 7))
ax = plt.subplot()

ax.scatter(df_train[df_train['Survived'] == 1]['Age'], df_train[df_train['Survived'] == 1]['Fare'], 
           c='green', s=df_train[df_train['Survived'] == 1]['Fare'])
ax.scatter(df_train[df_train['Survived'] == 0]['Age'], df_train[df_train['Survived'] == 0]['Fare'], 
           c='red', s=df_train[df_train['Survived'] == 0]['Fare']);


# The size of the circles is proportional to the ticket fare.
# 
# On the x-axis, we have the ages and the y-axis, we consider the ticket fare.
# 
# We can observe different clusters:
# 
# 1. Large green dots between x=20 and x=45: adults with the largest ticket fares
# 2. Small red dots between x=10 and x=45, adults from lower classes on the boat
# 3. Small green dots between x=0 and x=7: these are the children that were saved

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
df_train.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(20, 7), ax = ax);


# This plot says that fare for class 1 was highest followed by class 2 and class 3

# ## Data Cleansing 

# In[ ]:


def merge_train_test_data():
    # reading train data
    train = pd.read_csv('../input/titanic/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/titanic/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    merged_data_frame = train.append(test)
    merged_data_frame.reset_index(inplace=True)
    merged_data_frame.drop(['index', 'PassengerId'], inplace=True, axis=1)
    
    return merged_data_frame


# In[ ]:


merged_df = merge_train_test_data()


# In[ ]:


merged_df.shape


# In[ ]:


merged_df.head()


# ### Name Column

# In[ ]:


merged_df['Name'][1].split(',')[1].strip().split('.')[0].strip()


# In[ ]:


merged_df['Title'] = merged_df['Name'].map(lambda str_name:str_name.split(',')[1].strip().split('.')[0].strip())


# In[ ]:


merged_df['Title'].unique()


# In[ ]:


title_dict={'Mr':'Mr',
           'Mrs':'Mrs',
           'Ms':'Mrs',
           'Mme':'Miss',
           'Miss':'Miss',
           'Mlle':'Miss',
           'Master':'Master',
            'Col':'Defence_Officer',
            'Major':'Defence_Officer',
            'Capt':'Defence_Officer',
            'Dr':'Dr',
            'Jonkheer':'Jonkheer',
            'Don':'Don',
            'Rev':'Rev',
            'Lady':'Lady',
            'Sir':'Sir',
            'the Countess':'the Countess',
            'Dona':'Dona'
           }


# In[ ]:


merged_df['Title'] = merged_df['Title'].map(title_dict)


# In[ ]:


merged_df['Title'].unique()


# In[ ]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(merged_df['Title'], drop_first=True,prefix='Title')
# Adding the results to the master dataframe
merged_df = pd.concat([merged_df, dummy1], axis=1)
merged_df.drop('Title',axis=1,inplace=True)


# In[ ]:


merged_df.drop('Name',axis=1,inplace=True)


# #### PClass Column

# In[ ]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(merged_df['Pclass'].astype('category'), drop_first=True,prefix='Pclass')
# Adding the results to the master dataframe
merged_df = pd.concat([merged_df, dummy1], axis=1)
merged_df.drop('Pclass',axis=1,inplace=True)


# ### Sex Column

# In[ ]:


sex_dict = {'male':1,
           'female':0}
merged_df['Sex'] = merged_df['Sex'].map(sex_dict)


# ### Cabin Column

# In[ ]:


merged_df['Cabin'] = merged_df['Cabin'].str[0]


# In[ ]:


merged_df['Cabin'].unique()


# Lets assign a character U-unassigned for the records which have NULL values

# In[ ]:


merged_df['Cabin'].fillna('U',inplace=True)


# In[ ]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(merged_df['Cabin'], drop_first=True,prefix='Cabin')
# Adding the results to the master dataframe
merged_df = pd.concat([merged_df, dummy1], axis=1)
merged_df.drop('Cabin',axis=1,inplace=True)


# ### Embarked Column

# In[ ]:


merged_df.groupby('Embarked').count()['Sex']


# In[ ]:


merged_df['Embarked'].isnull().sum()


# In[ ]:


#Replacing the value of Embarked with 'S' based on the frequency
merged_df['Embarked'].fillna('S',inplace=True)


# In[ ]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(merged_df['Embarked'], drop_first=True,prefix='Embarked')
# Adding the results to the master dataframe
merged_df = pd.concat([merged_df, dummy1], axis=1)
merged_df.drop('Embarked',axis=1,inplace=True)


# ### Age Column

# In[ ]:


merged_df['Age'].isnull().sum()


# As the pepople with different age were travelling together lets replace the age as per their age group

# In[ ]:


def age_groups(num_age):
    if num_age>=0 and num_age <=2:
        return 'infant'
    elif num_age>=3 and num_age<=15:
        return 'kids'
    else:
        return 'adult'


# In[ ]:


merged_df['age_group']=merged_df['Age'].map(lambda age:age_groups(age))


# In[ ]:


age_group=merged_df.groupby('age_group').median()['Age'].reset_index()


# In[ ]:


age_group


# In[ ]:


age_group[age_group['age_group']=='adult']['Age'][0]


# In[ ]:


def age(row):
    if row['age_group']=='adult':
        return age_group[age_group['age_group']=='adult']['Age'][0]
    elif row['age_group']=='infant':
        return age_group[age_group['age_group']=='infant']['Age'][0]
    else:
        return age_group[age_group['age_group']=='kids']['Age'][0]


# In[ ]:


merged_df['Age']=merged_df.apply(lambda row: age(row) if np.isnan(row['Age']) else row['Age'], axis=1)


# In[ ]:


merged_df.head()


# In[ ]:


merged_df.drop('age_group',axis=1,inplace=True)


# ### SibSp and Parch Columns

# In[ ]:


merged_df['traveller_cnt'] = merged_df['SibSp'] + merged_df['Parch']+1


# In[ ]:





# ### Ticket Column

# In[ ]:


merged_df.drop('Ticket',axis=1,inplace=True)


# ### Getting the Train and Test Dataset

# In[ ]:


def get_train_test_target():
    targets = pd.read_csv('../input/titanic/train.csv', usecols=['Survived'])['Survived'].values
    train = merged_df.iloc[:891]
    test = merged_df.iloc[891:]
    
    return train, test, targets


# In[ ]:


train, test, targets = get_train_test_target()


# In[ ]:


train.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# ### Random Forest

# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
rf = rf.fit(train, targets)


# In[ ]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = rf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


# In[ ]:


features.plot(kind='barh', figsize=(25, 25))


# In[ ]:


from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(rf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)


# In[ ]:


test_reduced = model.transform(test)
print(test_reduced.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
#gboost = GradientBoostingClassifier()
#models = [logreg, logreg_cv, rf, gboost]
models = [logreg, rf]


# In[ ]:


from sklearn.model_selection import cross_val_score
def compute_score(rf, X, y, scoring='accuracy'):
    xval = cross_val_score(rf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# In[ ]:


for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(rf=model, X=train_reduced, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1,
                               n_jobs=-1
                              )

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 50, 
                  'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)


# ## Making Final Submisson File

# In[ ]:


test.head()


# In[ ]:


predictions = model.predict(test)


# In[ ]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(predictions)


# In[ ]:


y_pred_1.head()


# In[ ]:


# Renaming the column 
y_pred_1= y_pred_1.rename(columns={ 0 : 'Survived'})


# In[ ]:


df_gender_submission['Survived'] = y_pred_1['Survived'] 


# In[ ]:


df_gender_submission.head()

