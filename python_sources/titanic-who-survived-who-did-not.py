#!/usr/bin/env python
# coding: utf-8

# In[59]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# visualization libraries
import seaborn as sns
sns.set(style='darkgrid')
import warnings
warnings.filterwarnings('ignore')

# Machine learning Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Machine Learning performance metrics 
from sklearn.metrics import classification_report 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## **1. The Data**

# In[60]:


# Importing the train data
train = pd.read_csv('../input/train.csv')

# Importing the test data
test = pd.read_csv('../input/test.csv')

# save PassengerId for final submission
passengerId = test['PassengerId']

# merge the train and test data
df = train.append(test, ignore_index=True)

# create indexes to separate data later on
train_idx = len(train)
test_idx = len(df) - len(test)


# In[61]:


print('First 5 samples of the full Data')
df.head()


# ## **2. Descriptive Statistics**

# In[62]:


print('Full Data stats: ')
print('Number of records: {}\nNumber of features: {}'.format(df.shape[0],df.shape[1] ))
print('Are there missing values: {}, How many: {}'.format(df.isnull().any().any(),df.isnull().sum().sum()))
print('\n')
print(('*')*40)
print('\n')
print('Full Data info:')
print(df.info())


# From the descriptive statistics, we see that there are 1309 entries and 12 columns(features). Also, we notice that the Age, Cabin, Embarked, Fare, and Survived  columns have less than the total entries.
# This dataset will need to be cleaned! 

# ## **3. Exploratory Data Analysis**

# ### *Sex vs Survived*

# In[63]:


print('Ratio of Passengers onboard the Titanic:')
print(df.Sex.value_counts())
print('\n')
print(('*')*40)
print('\n')
print('Number of Passengers onboard the Titanic that survived:')
print(len(df[df.Survived == 1]))
print('\n')
print(('*')*40)
print('\n')
print('Ratio of the sex of passengers onboard the Titanic that Survived:' )
print(df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().round(2))
plt.figure(figsize=(10,6))
g= sns.countplot(data = df, x = 'Survived', hue= 'Sex')
g.set_xticklabels(["No", "Yes"]);
plt.ylabel('No of Passengers');


# #### ***Insights***:
# * Of the 1309 passengers onboard the titanic. 843 were men while 466 were women
# * 342 of these passengeres survived with 74% of them being women while 19%  men
# * This may be because an importance was put on rescuing women (and chiledren) first
#    
# #### ***Conclusion***:
# * There's a correlation between the Sex feature and the Target column, Survived 

# ### *Pclass vs Survived*

# In[64]:


g = sns.catplot(x="Pclass", hue="Sex", col="Survived", 
                data=df, kind="count",height=6, aspect=.7);
g.set_ylabels('No. of Passengers');
print('Percentage of Passengers survival by class: ')
print('\n')
print(pd.crosstab(index=[df.Pclass, df.Sex], 
            columns = df.Survived, normalize = True).round(3))


# #### ***Insights***:
# * Passengers, women mostly in 1st class survived more than the other passenger classes
# * Passengers, men especially, in 3rd class recorded the most deaths
# * This could be also related to fare price as the most expensive class had the best chance of surviving
#    
# #### ***Conclusion***:
# * There's a correlation between the Pclass feature and the Target, Survived 

# ### *Age vs Survived*

# In[65]:


g = sns.FacetGrid(df, col='Survived', height=5, aspect=.6)
g = g.map(plt.hist, 'Age', bins=10, color = 'm')
print('Average Age of the Passengers on board the Titanic by passenger class:')
print('\n')
print(pd.crosstab(index= df['Pclass'], 
            columns = df['Sex'],
            values = df['Age'], 
            aggfunc=np.mean).round())


# #### ***Insights***:
# * Most of the passengers onboard were between 15 - 35
# * Adults were in the 1st class while young people occupied the 3rd class
# * A large number of young people  (15-25) didnt survive
# * A lot of Infants survived (again, emphasis were put on women and children)
# * Older passengers (80 years and above) survived
#    
# #### ***Conclusion***:
# * There's a correlation between the Age feature and the Target, Survived 

# ### *Embarked vs Survived*

# In[66]:


print('Survival percentage of Passengers based off where they embarked')
print('\n')
print(pd.crosstab(index = df.Embarked, 
            columns = df.Survived, 
            margins = True, margins_name = 'Total', normalize = 'index').round(2))


# #### ***Insights***:
# * Passengers who paid higher ticket fares(Embarking at gate C) had a better survival rate, 55%
# * Q and S had the highest mortality rate, 61% & 66%
#    
# #### ***Conclusion***:
# * There's a correlation between the Embarked feature and the Target, Survived 

# > ### *Missing Data*
# * From the descriptive statistics carried out, we know that the data contains missing values
# * This can be dealth with by either dropping the features containing the missing values or for numerical features, imputing an average value for the missing values or for categorical features using the most frequent value
# * Lets investigate more

# In[69]:


missing = pd.DataFrame(df.isnull().sum()).rename(columns = {0:'missing'})
missing


# #### ***Insights***:
# * 263 records of the Age feature is missing. 
# * The Cabin feature is missing 1014 records
# * Survived feature is missing 418 records
# * The Embarked and fare features are small enough to be negligible.
# 
# #### ***Conclusion***:
# * The  Age data  missing is small enough for reasonable replacement with some form of imputation
# * We would use the mean age/class to impute this missing data
# * The Cabin feature is missing too many records. It will be dropped
# * The missing data in the Embarked and fare feature will be dropped
# * Since the survived feature is out target value, we will analyze this during the model selection stage
# 

# ### *Cleaning the  Missing Train Data*

# In[70]:


# For the Age column, we use the average age per persenger class 
first_class = round(df[df['Pclass'] == 1]['Age'].mean())
second_class = round(df[df['Pclass'] == 2]['Age'].mean())
third_class = round(df[df['Pclass'] == 3]['Age'].mean())

# to vreate a function 
def impute_age(col):
    '''
    Creating a function that replaces the missing 
    age values in each class by the average age
    value of the class
    
    '''
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return first_class
        elif Pclass == 2:
            return second_class
        else:
            return third_class
    else:
        return Age
    
# and impute the average age with the function created to fill the missing values
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis = 1)

# For the  Cabin column, we simply drop the entire column because of the large number of missing values
df.drop('Cabin', axis= 1, inplace= True)

# for the Embarked feature, we find most frequent Embarked value
most_embarked = df.Embarked.value_counts().index[0]

# and fill the missing values with most_embarked value
df.Embarked = df.Embarked.fillna(most_embarked)

# for the fare column, we fill the missing value with median fare
df.Fare = df.Fare.fillna(df.Fare.median())


# ### *Checking if there are still  Missing Train Data*

# In[71]:


# Visualizing if any missing values still persists besides the survived column
sns.heatmap(df.isnull(), cbar= False, yticklabels= False);


# Great! We can now go on to feature engineering

# ## **4. Feature Engineering**

# 
# From the EDA carried out, the features we would be focusing on are:
# * **Age**
# * **Pclass**
# * **Embarked**
# * **Sex**
# * **Fare**
# * **Parch**
# * **SibSp**
# 
# And our target variable will be the **Survived** feature

# ### *Feature Selection*

# In[72]:


# Convert the male and female categories to integers
df.Sex = df.Sex.map({"male": 0, "female":1})

# Create dummy variables for  the categorical features
pclass_dummies = pd.get_dummies(df.Pclass, prefix="Pclass")
embarked_dummies = pd.get_dummies(df.Embarked, prefix="Embarked")

# concatenate dummy columns with the data
df_dummies = pd.concat([df, pclass_dummies, embarked_dummies], axis=1)

# drop categorical features
df_dummies.drop(['Pclass', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

print('After encoding our data:')
df_dummies.head()


# ## **5. Modeling Using Random Forrests**

# ### *Model Selection*

# In[73]:


# create train and test data
train = df_dummies[ :train_idx]
test = df_dummies[test_idx: ]

# convert Survived back to int
train.Survived = train.Survived.astype(int)

# create X and y for data and target values 
X = train.drop('Survived', axis=1).values 
y = train.Survived.values

# create array for test set
X_test = test.drop('Survived', axis=1).values


# ### *Hyperparameters*

# In[74]:


# create param grid object 
rf_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

# instantiate a Random Forest classifier
clf = RandomForestClassifier()

# fit the model 
clf_cv = GridSearchCV(estimator=clf, param_grid=rf_params, cv=5, iid= False) 
clf_cv.fit(X, y)

print("Best score: {}".format(clf_cv.best_score_))


# ## **6. Submission**

# In[75]:


predictions = clf_cv.predict(X_test)
submission = pd.DataFrame({"PassengerId": passengerId, "Survived": predictions})
submission.to_csv('gender_submission.csv', index=False)

