#!/usr/bin/env python
# coding: utf-8

# ### Titanic Challenge
# 
#  
# **Data Dictionary:**
# - **Name** = Name of the passenger
# - **Pclass** = Ticket class
# - **Sex** = Sex of the passenger
# - **Age** = Age in years
# - **SibSp** = Number of siblings and spouses aboard
# - **Parch** = Number of parents and children aboard
# - **Ticket** = Ticket number
# - **Fare** = Passenger fare
# - **Cabin** = Cabin number
# - **Embarked** = where the passenger boarded the Titanic (C = Cherbourg, Q = Queenstown, S = Southampton)
# - **boat** = Number of life boats
# - **body** = the body identification number for victims.
# - **home.des** = passenger's home/destination.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
print('Done')


# In[ ]:


df = pd.read_csv ('../input/titanic.csv')


# ## Explore the data and Look for patterns

# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# ## Preprocessing and Data Transformation

# In[ ]:


#Replace the null value for sibsp and parch  with zero

df['parch'] = df['parch'].fillna(0)
df['sibsp'] = df['sibsp'].fillna(0)
df.head(2)


# In[ ]:


#fill the null value for age with mean
df['age'].fillna((df['age'].mean()), inplace=True)
print('Done')


# In[ ]:


#convert sex to numeric
gender_int = {'male': 0, 'female': 1}

df['sex'] = df['sex'].map(gender_int)
df.head(5)


# In[ ]:


#convert embarked to numeric
embk_int = {'C': 1, 'Q': 2, 'S': 3}

df['embarked'] = df['embarked'].map(embk_int)
df.head(5)


# In[ ]:


#create indicator for cabin
df['cabin1'] = np.where(df['cabin'].isnull(), 0, 1)
df.head(2)


# In[ ]:


#drop the cabin with object datatype and rename the new numeric colum
df.drop(['cabin'], axis=1, inplace=True)
df.rename(columns={'cabin1': 'cabin'}, inplace=True)
df.head(2)


# ## Look for patterns

# In[ ]:


df.pivot_table('survived', index='sex', columns='embarked', aggfunc='count')


# Embarked:  0 = Cherbourg, 1 = Queenstown, 2 = Southampton <br>
# The table shows higher survival for Southhampton port. 

# In[ ]:


df.pivot_table('survived', index='cabin', columns='embarked', aggfunc='count')


# The table shows higher survival for those without cabin

# ### Survival rate for men and women

# In[ ]:


men = df.loc[df.sex == False]["survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# In[ ]:


women = df.loc[df.sex == True]["survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# The survival rate for women is higher than men. 

# In[ ]:


mean_survived = df.groupby('survived').mean()
mean_survived


# In[ ]:


#add a new column category next to the age group. and convert continuous ages into categorical groups. 
#Child:1,Young:2,Adult:3
category = pd.cut(df.age,bins=[-1,16,36,99],labels=[1, 2, 3])
df.insert(5,'age_group',category)
df.head(2)


# # Hot Encoding

# In[ ]:


# hot encode categorical values 
cat_columns = ['embarked', 'pclass', 'cabin', 'sex', 'age_group']
df = pd.get_dummies(df, prefix_sep="_", columns=cat_columns)


# In[ ]:


#drop unnecessary variables
df.drop(['name', 'home.dest', 'boat', 'body', 'ticket', 'age'], axis=1, inplace=True)
df.head()


# In[ ]:


df.info()


# In[ ]:


#convert the data type to category
df['survived'] = df['survived'].astype('category', copy=False)


# In[ ]:


df.isnull().sum()


# In[ ]:


#There are one to three null values in some columns. Drop all rows with null value
df = df.dropna(how='any',axis=0) 


# In[ ]:


df.isnull().sum()


# #### With no null value the dataset is ready for training

# ## Prepare for Training

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#split the data for training, test, and validation (60,20,20)
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=30)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=30)

print(len(X), len(y_train), len(y_val), len(y_test))


# In[ ]:


for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset) / len(y), 1))


# ## Fit and evaluate a basic model using 5-fold Cross-Validation

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


# In[ ]:


RFC = RandomForestClassifier()
print(cross_validate(RFC, X_train, y_train.values, cv=5))


# ## Hyperparameter Tuning 

# In[ ]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# In[ ]:


rfc = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}

cv = GridSearchCV(rfc, parameters, cv=5)
cv.fit(X_train, y_train.values.ravel())

print_results(cv)


# ## Train the Model

# ### Fit the best model from last result to training set

# In[ ]:


rfc1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rfc1.fit(X_train, y_train.values.ravel())

rfc2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
rfc2.fit(X_train, y_train.values.ravel())

rfc3 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)
rfc3.fit(X_train, y_train.values.ravel())


# ### Validate the model

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[ ]:


# evaluate on validation test
for mdl in [rfc1, rfc2, rfc3]:
    y_pred = mdl.predict(X_val)
    accuracy = round(accuracy_score(y_val, y_pred), 3)
    precision = round(precision_score(y_val, y_pred), 3)
    recall = round(recall_score(y_val, y_pred), 3)
    print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth,
                                                                         mdl.n_estimators,
                                                                         accuracy,
                                                                         precision,
                                                                         recall))


# ## Besed on previous evaluations the following model is so far the best model to predict the survived passengers (with 81% accuracy).

# In[ ]:


y_pred = rfc1.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)
print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(rfc1.max_depth,
                                                                     rfc1.n_estimators,
                                                                     accuracy,
                                                                     precision,
                                                                     recall))

