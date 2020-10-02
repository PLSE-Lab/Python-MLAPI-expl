#!/usr/bin/env python
# coding: utf-8

# # First Kernel
# # EDA and Predictive Modeling of the titanic dataset

# In[1]:


# Data Analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Other Imports
import re
import warnings

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
# Machine Learning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


# Loading the datasets
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# * survival:    Survival 
# * PassengerId: Unique Id of a passenger. 
# * pclass:    Ticket class     
# * sex:    Sex     
# * Age:    Age in years     
# * sibsp:    # of siblings / spouses aboard the Titanic     
# * parch:    # of parents / children aboard the Titanic     
# * ticket:    Ticket number     
# * fare:    Passenger fare     
# * cabin:    Cabin number     
# * embarked:    Port of Embarkation
# * Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# * Continous: Age, Fare. Discrete: SibSp, Parch

# In[ ]:


train_df.info()


# In[ ]:


# Null values train dataset
train_df.isna().sum()


# In[ ]:


# Null values test dataset
test_df.isna().sum()


# In[6]:


train_df['source'] = 'train'
test_df['source'] = 'test'


# In[7]:


dataset = pd.concat([train_df, test_df], ignore_index=True)


# In[8]:


dataset.isnull().sum()


# In[ ]:


train_df.isnull().sum()


# **Types of features**
# * Categorical Features = Pclass, Sex, Sibsp, Parch, Cabin, Embarked
# * Numerical Features = Age, Fare
# * Others = Name, Ticket, PassengerID

# # Univariate Analysis (Numerical Variables)

# In[9]:


plt.figure(figsize=(10,4))
plt.subplot(121)
sns.distplot(train_df['Age'].dropna())
plt.title('Age Distribution')

plt.subplot(122)
sns.distplot(train_df['Fare'].dropna())
plt.title('Fare Distribution')


# # Univariate Analysis (Categorical Variables)

# In[10]:


train_df['Survived'].value_counts()


# In[11]:


train_df.Parch.value_counts()


# In[12]:


train_df.SibSp.value_counts()/891*100


# In[ ]:


train_df.Pclass.value_counts()/train_df.shape[0]*100


# # Bivariate Analysis

# * Categorical Variables = Pclass, Sex, Sibsp, Parch, Cabin, Embarked
# * Numerical Variables = Age, Fare
# * Others = Name, Ticket, PassengerID

# __**Categorical and Numerical**__

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived')
grid.map(sns.distplot, 'Age', bins=20, kde=False, color='green')


# In[ ]:


grid_pivot1 = train_df.pivot_table(columns='Survived', values='Age', aggfunc='mean')


# In[ ]:


grid_pivot1


# In[ ]:


sns.pointplot(x=train_df['Survived'].dropna(), y=train_df['Age'].dropna())


# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Sex')
grid.map(sns.distplot, 'Age', bins=20, kde=False, color='green')


# In[ ]:


# sns.pointplot(x=train_df['Survived'].fillna(-1), y=train_df['Age'].fillna(-1), hue=train_df['Sex'].fillna(-1)) 


# In[ ]:


grid_pivot2 = train_df.pivot_table(index='Sex', columns='Survived', values='Age', aggfunc='mean')
grid_pivot2


# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass')
grid.map(sns.distplot, 'Age', bins=20, kde=False,rug=True, color='red')


# In[ ]:


grid_pivot3 = train_df.pivot_table(index='Pclass',columns='Survived', values='Age')


# In[ ]:


grid_pivot3


# * Pclass 1 has the highest survival rate
# * Pclass 3 has most passengers however majority of them didn't survive
# * Infants of Pclass 2 & 3 were mostly survived

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='SibSp')
grid.map(sns.distplot,'Age', bins=20, kde=False,rug=True, color='pink')


# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Parch')
grid.map(sns.distplot,'Age', bins=20, kde=False,rug=True, color='green')


# * Most people were travelling alone
# * Very less chance of survival if with family

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Embarked')
grid.map(sns.distplot,'Age', bins=20, kde=False,rug=True, color='black')


# * Most people embarked from C
# 

# __Categorical Features__

# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep')
grid.add_legend()


# In[ ]:


train_df.pivot_table(index=['Embarked','Pclass'], columns='Sex', values='Survived')


# __Categorical and Numerical__

# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived')
grid.map(sns.barplot, 'Sex', 'Fare', ci=None)


# __Data Wrangling__ <br>
# Correcting, Completing and Creating Goals <br>
# Here we will start modifying out dataset

# __Creating__ new features extracting from existing features

# In[13]:


def extract_titles(name):
    tit = re.findall(' ([A-Za-z]+)\.', name)
    return tit[0]


# In[14]:


dataset['Title'] = dataset['Name'].apply(lambda x: extract_titles(x))


# In[ ]:


# train_df['Title'] = train_df['Name'].apply(lambda x: extract_titles(x))

# test_df['Title'] = test_df['Name'].apply(lambda x: extract_titles(x))


# In[15]:


dataset[dataset['source'] == 'train'].isnull().sum()


# In[16]:


dataset[dataset['source'] == 'test'].isnull().sum()


# In[17]:


dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr','Major', 'Lady', 'Sir','Col', 'Capt', 'Countess',
       'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace(['Ms', 'Mlle'], 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme','Mrs')


# In[18]:


dataset[dataset['source'] == 'train'].Title.value_counts()


# In[19]:


dataset[dataset['source'] == 'train'].pivot_table(index='Title', values='Survived')


# In[20]:


dataset[dataset['source'] == 'train'].pivot_table(index='Title', columns='Survived', values='Age')


# __Dropping features__

# In[ ]:


# dataset[dataset['source'] == 'train'] = dataset[dataset['source'] == 'train'].drop(['Name','PassengerId'], axis=1)

# dataset[dataset['source'] == 'train'].isnull().sum()

# train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

# test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)


# __Converting categorical features__

# In[21]:


dataset[dataset['source'] == 'train'].head()


# In[22]:


sex_dummy = pd.get_dummies(dataset['Sex'])


# In[23]:


sex_dummy.shape


# In[24]:


dataset.shape


# In[25]:


# train_df = pd.concat([train_df, sex_dummy], sort=False)
dataset = dataset.join(sex_dummy)


# In[26]:


dataset.head()


# In[ ]:


grid = sns.FacetGrid(dataset[dataset['source'] == 'train'], row='Pclass', col='Sex')
grid.map(sns.distplot, 'Age', bins=20, kde=False)


# In[ ]:


# train_df['Sex'] = train_df['Sex'].astype(int)


# In[27]:


grid_pivot = dataset[dataset['source'] == 'train'].pivot_table(index='Pclass',columns='Sex', values='Age', aggfunc='median')


# In[28]:


grid_pivot


# In[ ]:


# def fage(x):
#     age_med = grid_pivot.loc[x['Pclass'], x['Sex']]
#     return age_med


# In[ ]:


# dataset[dataset['source'] == 'train']['Age'].isna().sum()


# In[ ]:


# dataset.isna().sum()


# In[ ]:


# dataset['Age'].fillna(dataset[dataset['Age'].isnull()].apply(fage, axis=1), inplace=True)


# In[33]:


guess_ages = np.zeros((2,3))
guess_ages

for i in range(0, 2):
    for j in range(0, 3):
        guess_df = dataset[(dataset['Sex'] == i) &                               (dataset['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_df.median()
        
        # Convert random age float to nearest .5 age
        #guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                'Age'] = guess_ages[i,j]


# In[34]:


dataset['AgeBand'] = pd.cut(dataset['Age'], 5)


# In[35]:


dataset.AgeBand.value_counts()


# In[36]:


dataset['AgeBand'] = dataset['AgeBand'].astype(str)


# In[37]:


dataset.loc[dataset['source'] == 'train'].pivot_table(index='AgeBand', values='Survived')


# In[ ]:





# In[ ]:


# train_df.head()


# In[ ]:


# train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
# train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
# train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
# train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
# train_df.loc[(train_df['Age'] > 64), 'Age'] = 4


# In[ ]:


# train_df.head()


# In[38]:


def ageclass(x):
    if x <= 16:
        return 0
    elif x > 16 and x <= 32:
        return 1
    elif x > 32 and x <= 48:
        return 2
    elif x > 48 and x <= 64:
        return 3
    else:
        x > 64
        return 4


# In[39]:


dataset['AgeClass'] = dataset['Age'].apply(ageclass)


# In[40]:


dataset.head()


# __Create new features by combining existing ones__

# In[41]:


dataset['Family_Size']  = dataset['Parch'] + dataset['SibSp'] + 1


# In[42]:


dataset.pivot_table(index='Family_Size', values='Survived').sort_values(by='Survived',ascending=False )


# In[43]:


def isalone(x):
    if x == 1:
        return 1
    else:
        return 0


# In[44]:


dataset['IsAlone'] = dataset['Family_Size'].apply(isalone)


# In[45]:


dataset[dataset['source']=='train'].pivot_table(index='IsAlone', values='Survived')


# In[ ]:


dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


embarked_dummy = pd.get_dummies(dataset['Embarked'])


# In[ ]:


dataset = dataset.join(embarked_dummy)


# In[ ]:


dataset.head()


# In[ ]:


dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)


# In[ ]:


le = LabelEncoder()


# In[ ]:


dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)


# In[ ]:


dataset['FareClass'] = le.fit_transform(dataset['FareBand'])


# In[ ]:


dataset.pivot_table(index='FareClass', values='Survived').sort_values(by='Survived', ascending=False)


# In[ ]:


# def fareclass(x):
#     if x <= 7.896 :
#         return 0
#     elif x > 7.896 and x <= 14.454:
#         return 1
#     elif x > 14.454 and x <= 31.275 :
#         return 2
#     elif x > 31.275 and x <= 512.329:
#         return 3


# In[ ]:


# dataset['FareClass'] = dataset['Fare'].apply(fareclass)


# In[ ]:


# dataset['FareClass'] = dataset['Fare'].apply(fareclass)

# dataset['FareClass'] = dataset['FareClass'].astype(int)

# test_df['FareClass'] = test_df['Fare'].apply(fareclass).astype(int)


# In[ ]:


dataset.keys()


# In[ ]:


dataset['Title'] = le.fit_transform(dataset['Title'])


# In[ ]:


dataset['Age*Class'] = dataset['AgeClass'] * dataset['Pclass']


# In[ ]:


dataset.head()


# In[ ]:


drop_these = 'Age Cabin Embarked Fare Name Parch Ticket Sex SibSp AgeBand Family_Size FareBand'.split(' ')


# In[ ]:


drop_these


# In[ ]:


dataset = dataset.drop(drop_these, axis=1)


# In[ ]:


dataset.head()


# In[ ]:


train_cleaned = dataset[dataset['source'] == 'train']
test_cleaned = dataset[dataset['source'] == 'test']


# In[ ]:


train_cleaned['Survived'] = train_cleaned['Survived'].astype(int)


# In[ ]:


train_cleaned.keys()


# In[ ]:


'Pclass', 'Survived', 'source', 'Title', 'female','male', 'AgeClass', 'IsAlone', 'C', 'Q', 'S', 'FareClass', 'Age*Class'


# # Model Predict

# In[ ]:


def predict_model(dtrain, dtest, predictor, outcome, model):
    model.fit(dtrain[predictor], dtrain[outcome])
    dtrain_pred = model.predict(dtest[predictor])
    score = model.score(dtrain[predictor], dtrain[outcome])*100
    return score, dtrain_pred


# In[ ]:


predictors_var = ['Pclass','Title', 'female','male', 'AgeClass', 'IsAlone', 'C', 'Q', 'S', 'FareClass', 'Age*Class']
outcome_var = 'Survived'
# model_name = logreg
traindf = train_cleaned
testdf = test_cleaned


# __Logistic Regression__

# In[ ]:


logreg = LogisticRegression()


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, logreg)


# In[ ]:


coef1 = pd.Series(logreg.coef_[0], predictors_var).sort_values()


# In[ ]:


coef1.sort_values(ascending=False)


# __Support Vector Machine__

# In[ ]:


svc = SVC()


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, svc)


# In[ ]:


coef2 = pd.Series(svc.coef_[0], predictors_var).sort_values()

coef2.sort_values(ascending=False)


# __K Neighbour Classifier__

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, knn)


# __GaussianNB__

# In[ ]:


gaussian = GaussianNB()


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, gaussian)


# __Decision Tree__

# In[ ]:


decision_tree = DecisionTreeClassifier()


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, decision_tree)


# In[ ]:


coef4 = pd.Series(decision_tree.feature_importances_, predictors_var).sort_values()

coef4.sort_values(ascending=False)


# __Random Forest__

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, random_forest)


# In[ ]:


predict_model(traindf, testdf, predictors_var, outcome_var, random_forest)[1]


# In[ ]:


results = predict_model(traindf, testdf, predictors_var, outcome_var, random_forest)[1]


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": results
    })


# In[ ]:


submission.to_csv('submission_updated.csv', index=False)

