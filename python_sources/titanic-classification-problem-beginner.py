#!/usr/bin/env python
# coding: utf-8

# # Titanic - Clasification Problem

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span><ul class="toc-item"><li><span><a href="#Import-Libraries-and-Data" data-toc-modified-id="Import-Libraries-and-Data-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Import Libraries and Data</a></span></li><li><span><a href="#Check-dataframes" data-toc-modified-id="Check-dataframes-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Check dataframes</a></span></li><li><span><a href="#Checking-missing-values" data-toc-modified-id="Checking-missing-values-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Checking missing values</a></span></li><li><span><a href="#Summary-of-Variables-and-what-to-do-with-each-one" data-toc-modified-id="Summary-of-Variables-and-what-to-do-with-each-one-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Summary of Variables and what to do with each one</a></span></li><li><span><a href="#Drop-useless-variables" data-toc-modified-id="Drop-useless-variables-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Drop useless variables</a></span></li></ul></li><li><span><a href="#Data-Visualization" data-toc-modified-id="Data-Visualization-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Visualization</a></span><ul class="toc-item"><li><span><a href="#First-look-visualizations" data-toc-modified-id="First-look-visualizations-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>First look visualizations</a></span><ul class="toc-item"><li><span><a href="#Distrubution-of-each-variable" data-toc-modified-id="Distrubution-of-each-variable-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Distrubution of each variable</a></span></li><li><span><a href="#Age-distribution" data-toc-modified-id="Age-distribution-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Age distribution</a></span></li></ul></li><li><span><a href="#Categorical-Variables" data-toc-modified-id="Categorical-Variables-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Categorical Variables</a></span><ul class="toc-item"><li><span><a href="#Sex" data-toc-modified-id="Sex-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Sex</a></span></li><li><span><a href="#Pclass" data-toc-modified-id="Pclass-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Pclass</a></span></li><li><span><a href="#Embarked" data-toc-modified-id="Embarked-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Embarked</a></span></li><li><span><a href="#Survival-rates" data-toc-modified-id="Survival-rates-2.2.4"><span class="toc-item-num">2.2.4&nbsp;&nbsp;</span>Survival rates</a></span></li></ul></li><li><span><a href="#Numerical-variables" data-toc-modified-id="Numerical-variables-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Numerical variables</a></span><ul class="toc-item"><li><span><a href="#Age" data-toc-modified-id="Age-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Age</a></span></li><li><span><a href="#Fare" data-toc-modified-id="Fare-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Fare</a></span></li></ul></li><li><span><a href="#Correlations" data-toc-modified-id="Correlations-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Correlations</a></span></li></ul></li><li><span><a href="#Data-manipulation" data-toc-modified-id="Data-manipulation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data manipulation</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Create-variable:-Familysize" data-toc-modified-id="Create-variable:-Familysize-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>Create variable: Familysize</a></span></li><li><span><a href="#Create-variable:-Alone" data-toc-modified-id="Create-variable:-Alone-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>Create variable: Alone</a></span></li><li><span><a href="#Fill-Embarked-values-(df_train)" data-toc-modified-id="Fill-Embarked-values-(df_train)-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>Fill Embarked values (df_train)</a></span></li><li><span><a href="#Fill-Fare-value-(df_test)" data-toc-modified-id="Fill-Fare-value-(df_test)-3.0.4"><span class="toc-item-num">3.0.4&nbsp;&nbsp;</span>Fill Fare value (df_test)</a></span></li><li><span><a href="#Fill-Age-values" data-toc-modified-id="Fill-Age-values-3.0.5"><span class="toc-item-num">3.0.5&nbsp;&nbsp;</span>Fill Age values</a></span></li><li><span><a href="#Extract-Title-from-Name" data-toc-modified-id="Extract-Title-from-Name-3.0.6"><span class="toc-item-num">3.0.6&nbsp;&nbsp;</span>Extract Title from Name</a></span></li><li><span><a href="#Handle-Title" data-toc-modified-id="Handle-Title-3.0.7"><span class="toc-item-num">3.0.7&nbsp;&nbsp;</span>Handle Title</a></span></li></ul></li><li><span><a href="#Encoding-variables-(Dummies)" data-toc-modified-id="Encoding-variables-(Dummies)-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Encoding variables (Dummies)</a></span></li><li><span><a href="#Scaling-Age-and-Fare-variables" data-toc-modified-id="Scaling-Age-and-Fare-variables-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Scaling Age and Fare variables</a></span></li></ul></li><li><span><a href="#Algorithms" data-toc-modified-id="Algorithms-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Algorithms</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#KNN" data-toc-modified-id="KNN-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>KNN</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Random Forest</a></span></li><li><span><a href="#XGBOOST" data-toc-modified-id="XGBOOST-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>XGBOOST</a></span></li></ul></li><li><span><a href="#Results" data-toc-modified-id="Results-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Results</a></span></li><li><span><a href="#Ensemble" data-toc-modified-id="Ensemble-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Ensemble</a></span></li><li><span><a href="#Submision" data-toc-modified-id="Submision-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Submision</a></span></li><li><span><a href="#To-do-list" data-toc-modified-id="To-do-list-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>To do list</a></span></li></ul></div>

# ## Introduction

# ### Import Libraries and Data

# In[ ]:


# Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

sns.set(style="white", font_scale=1.2)


# In[ ]:


# Load dataframes

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')


# ### Check dataframes

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


# Info to see rows, columns and missing values

df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.describe(include='O')


# ### Checking missing values

# In[ ]:


# Heatmap of missing values in train and test dataframes

fig, axes = plt.subplots(1, 2, figsize=(12,4))
# Train
sns.heatmap(df_train.isnull(), cmap='viridis', cbar=False, yticklabels=False, ax=axes[0])
axes[0].set_title('TRAIN DF')

#Test
sns.heatmap(df_test.isnull(), cmap='viridis', cbar=False, yticklabels=False, ax=axes[1])
axes[1].set_title('TEST DF')


# In[ ]:


# Function to check missing values in each dataframe

def check_missing_values(df, df_name=None):
    print(f'{df_name} - Missing values:')
    print('-'*30)
    columns = df.columns

    for column in columns:
        count_missing_values = df[column].isnull().sum()
        missing_values = (count_missing_values / len(df[column])) * 100
    
        if missing_values !=0:
            print(f'{column} --> {count_missing_values} values | {missing_values:.2f}%')


# In[ ]:


check_missing_values(df_train, 'TRAIN')


# In[ ]:


check_missing_values(df_test, 'TEST')


# In a first view we see that:
# 
# - Cabin contain too many values in the train and test dataframes, so we'll delete that variable.
# 
# - There are many Age values missing, we'll deal with them later. Around 20% in each dataframe
# 
# - 2 Embarked  missing in TRAIN DF
# 
# - 1 Fare missing in TEST DF

# ### Summary of Variables and what to do with each one
# 
# * **PassengerId**: Unique identification of the passenger. -> _Delete_
# * **Survived**: Survival (0 = No, 1 = Yes). -> _Ready_
# * **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd). -> _Encode (categorical)_
# * **Name**: Name of the passenger. -> _Still don't know_
# * **Sex**: Sex. -> _Encode (categorical)_
# * **Age**: Age in years. -> _Fill missing values in an easy way and maybe group in intervals_
# * **SibSp**: # of siblings / spouses aboard the Titanic. -> _Ready_
# * **Parch**: # of parents / children aboard the Titanic. -> _Ready_
# * **Ticket**: Ticket number. -> _Delete?_
# * **Fare**: Passenger fare. -> _Maybe group in intervals_
# * **Cabin**: Cabin number. -> _Delete_
# * **Embarked**: Port of Embarkation. _Encode (categorical)_

# ### Drop useless variables

# In[ ]:


# Drop PasserngerId, Cabin and Ticket
# Drop Name also (maybe latter I can fix this and try to use Name)
df_train.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)

submission = pd.DataFrame()
submission['PassengerId'] = df_test['PassengerId']
df_test.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)


# In[ ]:


print('df_train shape:',df_train.shape)
print('df_test shape:',df_test.shape)


# ## Data Visualization

# In[ ]:


df_train.head()


# ### First look visualizations

# #### Distrubution of each variable

# In[ ]:


df_train.hist(bins=15, figsize=(10, 7))
plt.tight_layout()


# - Age and Fare on different scale

# #### Age distribution

# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(20,5))

sns.distplot(df_train['Age'].dropna(), kde=False, bins=30, ax=axes[0])
axes[0].set_title('Age Distribution overall')

sns.distplot(df_train[df_train['Sex']=='male']['Age'].dropna(),
             kde=False, color='blue', bins=30, ax=axes[1])
axes[1].set_title('Age Distribution (Male)')

sns.distplot(df_train[df_train['Sex']=='female']['Age'].dropna(),
             kde=False, color='orange', bins=30, ax=axes[2])
axes[2].set_title('Age Distribution (Female)')

sns.kdeplot(df_train[df_train['Sex']=='male']['Age'].dropna(),
            color='blue', ax=axes[3])
sns.kdeplot(df_train[df_train['Sex']=='female']['Age'].dropna(),
            color='orange', ax=axes[3])


# - The distributions by sex are similar
# 
# - There are extreme values (outliers?)

# ### Categorical Variables

# #### Sex

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.countplot(x='Sex', data=df_train, ax=axes[0])
axes[0].set_title('Number of males and females')

sns.countplot(x='Sex', hue='Survived', data=df_train, ax=axes[1], palette='Set3')
axes[1].set_title('Survival by sex')
axes[1].set_ylabel('')


# - There are more males than females
# 
# - Males tend to die, Females tend to survive

# #### Pclass

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(16,5))

sns.countplot(x='Pclass', data=df_train, ax=axes[0], palette='Set1')
axes[0].set_title('Number of people in each Pclass')

sns.countplot(x='Pclass', hue='Sex', data=df_train, ax=axes[1])
axes[1].set_title('Sex by Pclass')
axes[1].set_ylabel('')

sns.countplot(x='Pclass', hue='Survived', data=df_train, ax=axes[2], palette='Set3')
axes[2].set_title('Survival by Pclass')
axes[2].set_ylabel('')

plt.tight_layout()


# - More people in third class
# - Higher ratio Survive:Die in third class
# - More men than women die indepentedly of the class

# #### Embarked

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(16,5))

sns.countplot(x='Embarked', data=df_train, ax=axes[0], palette='Set1')
axes[0].set_title('Number of people in each Embarkation')

sns.countplot(x='Embarked', hue='Sex', data=df_train, ax=axes[1])
axes[1].set_title('Sex by Embarcation')
axes[1].set_ylabel('')

sns.countplot(x='Embarked', hue='Survived', data=df_train, ax=axes[2], palette='Set3')
axes[2].set_title('Survival by Embarcation')
axes[2].set_ylabel('')

plt.tight_layout()


# #### Survival rates

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(14,5))

sns.pointplot(x ='Sex', y="Survived", data=df_train, ax=axes[0])
axes[0].set_title('Survival by Sex')

sns.pointplot(x ='Pclass', y="Survived", data=df_train, ax=axes[1])
axes[1].set_title('Survival by Pclass')
axes[1].set_ylabel('')

sns.pointplot(x ='Embarked', y="Survived", data=df_train, ax=axes[2])
axes[2].set_title('Survival by Embarkation')
axes[2].set_ylabel('')

for ax in axes:
    ax.set_yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()


# ### Numerical variables

# #### Age

# In[ ]:


sns.boxplot(x='Survived', y='Age', data=df_train, palette='Set3')
plt.title('Survival by Age')


# #### Fare

# In[ ]:


sns.boxplot(x='Survived', y='Fare', data=df_train)
plt.title('Survival by Fare')


# ### Correlations

# In[ ]:


df_train.corr()['Survived']


# In[ ]:


plt.figure(figsize=(8,8))

sns.heatmap(df_train.corr(), annot=True, cmap='magma', square=True,
            linecolor="white", linewidths=0.1)
plt.title('Correlations between variables')


# - Pclass and Fare are the most correlated, but not much.

# ## Data manipulation

# - Create **Family size** (Family = SibSp + Parch) and **Alone** if doesn't have family members
# 
# - Fill all the missing values of **Age** in both dataframes (with mean based on Sex and Pclass) -> Maybe use some algorithm to predict them, in a future project.
# 
# - Fill 2 values of **Emarked** from df_train with the most common one or check in relation with other variables
# 
# - Fill 1 value of **Fare** from df_test

# In[ ]:


check_missing_values(df_train, 'DF TRAIN')


# In[ ]:


check_missing_values(df_test, 'DF TEST')


# #### Create variable: Familysize

# In[ ]:


df_train['Familysize'] = df_train['SibSp'] + df_train['Parch']

df_test['Familysize'] = df_test['SibSp'] + df_test['Parch']


# #### Create variable: Alone

# In[ ]:


# 1 if is alone, 0 if has family members
df_train['Alone'] = df_train['Familysize'].apply(lambda x: 1 if x == 0 else 0)

df_test['Alone'] = df_test['Familysize'].apply(lambda x: 1 if x == 0 else 0)


# #### Fill Embarked values (df_train)

# In[ ]:


df_train[df_train['Embarked'].isnull()]


# We can see that both missing values share the variables: Fare, Pclass, Sex, Survived, Alone.

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(16,8))

sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df_train, ax=axes[0,0])

sns.boxplot(x="Embarked", y="Fare", hue="Sex", data=df_train, ax=axes[0,1])

sns.boxplot(x="Embarked", y="Fare", hue="Survived", data=df_train, ax=axes[1,0])

sns.boxplot(x="Embarked", y="Fare", hue="Alone", data=df_train, ax=axes[1,1])

plt.tight_layout()


# Based on the median values of the plots seems likely to be 'C' > 'S', definitely is not Q. I'll go with C since Pclass, Sex and Survived point to that.

# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna('C')


# #### Fill Fare value (df_test)

# In[ ]:


df_test[df_test['Fare'].isnull()]


# In[ ]:


median_fare = df_test[(df_test['Pclass'] == 3) & (df_test['Embarked'] == 'S') & (df_test['Alone'] == 1)]['Fare'].median()

median_fare


# In[ ]:


df_test['Fare'] = df_test['Fare'].fillna(median_fare)


# #### Fill Age values

# In[ ]:


plt.figure(figsize=(12, 7))

testPlot = sns.boxplot(x='Pclass', y='Age', hue='Sex', data=df_train)

m1 = df_train.groupby(['Pclass', 'Sex'])['Age'].median().values
mL1 = [str(np.round(s, 2)) for s in m1]

ind = 0
for tick in range(len(testPlot.get_xticklabels())):
    testPlot.text(tick-.2, m1[ind+1]+1, mL1[ind+1],  horizontalalignment='center',  color='w', weight='semibold')
    testPlot.text(tick+.2, m1[ind]+1, mL1[ind], horizontalalignment='center', color='w', weight='semibold')
    ind += 2

# Display median values from: https://stackoverflow.com/questions/45475962/labeling-boxplot-with-median-values/45476485


# In[ ]:


# Get median value for Age based on Pclass and Sex (Not having survive/die in account, for now)

def get_age(cols):
    age = cols[0]
    pclass = cols[1]
    sex = cols[2]
    
    if pd.isnull(age):

        if pclass == 1:
            if sex == 'male':
                return 40
            else:
                return 35

        elif pclass == 2:
            if sex == 'male':
                return 30
            else:
                return 28

        else:
            if sex == 'male':
                return 25
            else:
                return 21.5
            
    else:
        return age


# In[ ]:


df_train['Age'] = df_train[['Age','Pclass', 'Sex']].apply(get_age, axis=1)

df_test['Age'] = df_test[['Age','Pclass', 'Sex']].apply(get_age, axis=1)


# #### Extract Title from Name

# In[ ]:


def get_title(name):
    for string in name.split():
        if '.' in string:
            return string[:-1]


# In[ ]:


df_train['Title'] = df_train['Name'].apply(lambda x: get_title(x))

df_test['Title'] = df_test['Name'].apply(lambda x: get_title(x))


# In[ ]:


df_train['Title'].value_counts()


# **Drop Name**

# In[ ]:


df_train.drop('Name', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)


# #### Handle Title

# In[ ]:


for dataframe in [df_train, df_test]:
    
    dataframe['Title'] = dataframe['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 
                                                 'Major', 'Rev', 'Sir', 'Dona', 'Countess', 'Jonkheer'], 'Other')

    dataframe['Title'] = dataframe['Title'].replace('Mlle', 'Miss')
    dataframe['Title'] = dataframe['Title'].replace('Ms', 'Miss')
    dataframe['Title'] = dataframe['Title'].replace('Mme', 'Mrs')


# ### Encoding variables (Dummies)

# **TRAIN DF**

# In[ ]:


sex = pd.get_dummies(df_train['Sex'], prefix='Sex', drop_first=True)
embarked = pd.get_dummies(df_train['Embarked'], prefix='Embarked', drop_first=True)
pclass = pd.get_dummies(df_train['Pclass'], prefix='Pclass', drop_first=True)
title = pd.get_dummies(df_train['Title'], prefix='Title', drop_first=True)

df_train.drop(['Sex', 'Embarked', 'Pclass', 'Title'], axis=1, inplace=True)

df_train = pd.concat([df_train, sex, embarked, pclass, title], axis=1)


# In[ ]:


print('df_train shape:',df_train.shape)
df_train.head()


# **TEST DF**

# In[ ]:


sex = pd.get_dummies(df_test['Sex'], prefix='Sex', drop_first=True)
embarked = pd.get_dummies(df_test['Embarked'], prefix='Embarked',drop_first=True)
pclass = pd.get_dummies(df_test['Pclass'], prefix='Pclass',drop_first=True)
title = pd.get_dummies(df_test['Title'], prefix='Title', drop_first=True)

df_test.drop(['Sex', 'Embarked', 'Pclass', 'Title'], axis=1, inplace=True)

df_test = pd.concat([df_test, sex, embarked, pclass, title], axis=1)


# In[ ]:


print('df_test shape:',df_test.shape)
df_train.head()


# ### Scaling Age and Fare variables

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train[['Age', 'Fare']] = scaler.fit_transform(df_train[['Age', 'Fare']])

df_test[['Age', 'Fare']] = scaler.transform(df_test[['Age', 'Fare']])


# In[ ]:


df_train.corr()['Survived'].sort_values()[:-1]


# In[ ]:


df_train.corr()['Survived'].sort_values()[:-1].plot.bar()


# ## Algorithms

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer


# In[ ]:


X = df_train.drop('Survived', axis=1)
y = df_train['Survived']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)


# In[ ]:


# Dictionary with each prediction
predictions = {}


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# **Base Line**

# In[ ]:


logreg = LogisticRegression(random_state=121)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)


# **Tunned parameters**

# In[ ]:


logreg = LogisticRegression(random_state=121)

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,18],
    'solver': ['liblinear','saga']}


# In[ ]:


model = GridSearchCV(logreg, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

model.fit(X_train, y_train)

print('Best Params:', model.best_params_)


# Best Params: {'C': 0.9, 'penalty': 'l1', 'solver': 'liblinear'}

# In[ ]:


best_lr = LogisticRegression(C=0.9, penalty='l1', solver='liblinear')
best_lr.fit(X_train, y_train)
y_pred = best_lr.predict(X_test)


# In[ ]:


print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print('-'*55)
print(classification_report(y_test, y_pred))
print('-'*55)
print(confusion_matrix(y_test, y_pred))


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# **Choosing best K value**

# In[ ]:


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    
    pred_i = knn.predict(X_test)
    
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


# Plot Error rate vs Number of neighbors
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate,
         color='blue', ls='--',
         marker='o', markerfacecolor='red', markersize=10)
plt.xlabel('Neighbors')
plt.ylabel('Error rate')
plt.title('Error rate vs Number of neighbors')


# I tried with 3, 5 and 25 neighbors and **25** was the best on the submission.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# In[ ]:


print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print('-'*55)
print(classification_report(y_test, y_pred))
print('-'*55)
print(confusion_matrix(y_test, y_pred))


# ### Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# **Base Line**

# In[ ]:


rf = RandomForestClassifier(random_state=121)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# **Tunned parameters**

# In[ ]:


#rf = RandomForestClassifier(random_state=121)
#param_grid = {
#    'criterion':['giny', 'entropy'],
#    'n_estimators':[50, 100, 500, 750, 1000],
#    'max_depth':[5, 8, 15, 25, 30],
#    'min_samples_split':[2, 5, 10, 15, 100],
#    'min_samples_leaf':[1, 5, 10]}


# In[ ]:


#model = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)

#model.fit(X_train, y_train)

#print('Best Params:', model.best_params_)


# Best Params: {'criterion': 'entropy', 'max_depth': 15, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 50}

# In[ ]:


best_rf = RandomForestClassifier(random_state=121, criterion='entropy', max_depth=15, min_samples_leaf=5, min_samples_split=2, n_estimators=50)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print('-'*55)
print(classification_report(y_test, y_pred))
print('-'*55)
print(confusion_matrix(y_test, y_pred))


# ### XGBOOST

# In[ ]:


from xgboost import XGBClassifier


# **Base line**

# In[ ]:


xgb = XGBClassifier(random_state=121)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# **Tuned parameters**

# In[ ]:





# ## Results

# In[ ]:


classifiers = [('Logistic Regression', best_lr),
               ('KNN', knn),
               ('Random Forest', best_rf),
               ('Xgboost', xgb)]

for name_clf, clf in classifiers:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name_clf} accuracy: {round(acc, 3)}%')


# ## Ensemble

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


vc = VotingClassifier(estimators=classifiers)

vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)

acc_vc = accuracy_score(y_test, y_pred)

print(f'Ensembler Accuracy: {round(acc_vc, 3)}%')


# ## Submision

# Fit the ensembler with the full dataset, to make prediction in the test dataset

# In[ ]:


vc.fit(X, y)
prediction = vc.predict(df_test)

submission['Survived'] = prediction

submission.to_csv('Submission.csv', index=False)


# ## To do list
# 
# - Group age/fare in intervals
# 
# - Hyperparameters KNN and XGBOOST

# In[ ]:




