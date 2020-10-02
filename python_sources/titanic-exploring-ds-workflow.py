#!/usr/bin/env python
# coding: utf-8

# This work is a walkthrough of the most basic Data Science workflow practices.
# 
# In this notebook we will:
# - Clean the dataset and feature engineer,
# - Build predictive models and rank their performace,
# - Tune hyperparameters and cross-validate.
# 
# We will test out each practice, observe how it enhances prediction accuracy and make sense of them.

# In[ ]:


# import needed packages
import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load datasets
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# # 1. Get to know the dataset
# Lets get some general information about our dataset

# In[ ]:


train.describe()


# So about **38%** of the passengers survived the titanic in this dataset.
# Ages seem to range from **0.42** to **80**. 
# We can also see that there are missing data as the row **count** indicates.

# In[ ]:


# check for NaN values
train.isnull().sum()


# Column **Embarked** has only two missing values, which can be easily replaced by the most frequent value. Missing values in the column **Age** can potentionally be replaced with the mean age of all passangers. The **Cabin** column has many missing values and needs further investigation.

# # 2. Cleaning data
# First, lets try poorly cleaning our data and see how our models will perform.
# 
# Columns **Name**, **Cabin** and **PassengerId** don't seem to have much corelation to the survivabilty at first glance, let's just drop them.
# 
# We don't have to change our **test** dataset since we will do a more throughout cleaning later on, this is only to showcase how important data engineering is and how it can affect our prediction accuracy.

# In[ ]:


# make a copy of our dataset
train1 = train.copy()

# drop columns
train1.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=True)


# Dropping rows with NaN from this point on drops only maximum of 179 rows, which is kind of ok for starters.

# In[ ]:


# drop rows with NaN
train1.dropna(inplace=True)


# In[ ]:


# convert string columns to type int
# we will put this into a function since we will be using this later on
def string_to_int(dataset):
    string_cols = dataset.select_dtypes(['object']).columns
    dataset[string_cols].nunique()
    dataset[string_cols] = dataset[string_cols].astype('category').apply(lambda x: x.cat.codes)
    
# call function
string_to_int(train1)


# Let's check:

# In[ ]:


train1


# **Age** and **Fare** colums data need to be categorized so our Classifiers can work better. That will be for later though, since we want to try fitting badly cleaned data and see how our models will perform.

# # 3. Building models
# For this work, we will be using 3 ML classifiers: **DecisionTreeClassifier**, **RandomForestClassifier** and **AdaBoostClassifier**, ranking them by their accuracy scores while watching how this ranking chages as we clean our dataset more precisely.

# ## 3.1 Hyperparameter tuning and cross-validation
# 
# As rankings can be very close, lets do cross-validation and tune hyperparameters for each of them to maximize the fairness. For that, I've decided to use sklearn's **[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.fit)**.
# 
# Since **GridSearchCV** is an exhaustive search (it's really slow) and we plan on calling our **GridSearchCV** many times, let's first try narrowing down our hyperparameter ranges as much as possible.

# ### 3.1.1 Narrowing down hyperparameter ranges for each classifier

# #### Decision Tree Classifier

# In[ ]:


# splitting dataset into training and validation data
Xdata = train1.drop(columns = ['Survived'])
ydata = train1['Survived']
rd_seed = 333 # data splitted randomly with this seed
Xtrain, Xval, ytrain, yval = train_test_split(Xdata, ydata, test_size=0.25, random_state=rd_seed)


# In[ ]:


# chosen range of hyperparameters for Decision Tree Classifier
param_grid = {
    'max_depth': range(1,100), 
    'criterion': ['entropy', 'gini']
}
# using sklearn's ParameterGrid to make a grid of parameter combinations
param_comb = ParameterGrid(param_grid)


# In[ ]:


# iterate each parameter combination and fit model
val_acc = []
train_acc = []
for params in param_comb:
    dt = DecisionTreeClassifier(**params)
    dt.fit(Xtrain, ytrain)
    train_acc.append(metrics.accuracy_score(ytrain, dt.predict(Xtrain)))
    val_acc.append(metrics.accuracy_score(yval, dt.predict(Xval)))


# In[ ]:


# plotting how accurate our model is with each hyperparameter combination
plt.figure(figsize=(20,6))
plt.plot(train_acc,'or-')
plt.plot(val_acc,'ob-')
plt.xlabel('hyperparametr index')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])
best_params = param_comb[np.argmax(val_acc)]
best_params


# In[ ]:


print(param_comb[5]) # printing hyperparameters with index 5
print(param_comb[105]) # printing hyperparameters with index 105


# As we can see from our graph, comparing **gini** and **entropy** criterions can be beneficial.
# As for the **max_depth** hyperparameter however, the accuracy on validation dataset doesn't get better after the value 15. As a matter of fact, it seems to be heavily overfitting (when validation score doesn't get better and train score is close to 100%).

# #### Random Forest Classifier

# In[ ]:


# chosen range of hyperparameters for Random Forest Classifier
param_grid = {
    'max_depth': range(1, 30), 
    'n_estimators': range(1, 500, 25),
}
# using sklearn's ParameterGrid
param_comb = ParameterGrid(param_grid)


# In[ ]:


# iterate each parameter combination and fit model
val_acc = []
train_acc = []
for params in param_comb:
    dt = RandomForestClassifier(**params)
    dt.fit(Xtrain, ytrain)
    train_acc.append(metrics.accuracy_score(ytrain, dt.predict(Xtrain)))
    val_acc.append(metrics.accuracy_score(yval, dt.predict(Xval)))


# In[ ]:


# plotting how accurate our model is with each hyperparameter combination
plt.figure(figsize=(20,6))
plt.plot(train_acc,'or-')
plt.plot(val_acc,'ob-')
plt.xlabel('hyperparametr index')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])
best_params = param_comb[np.argmax(val_acc)]
best_params


# In[ ]:


print(param_comb[0]) # printing hyperparameters with index 0
print(param_comb[1]) # printing hyperparameters with index 1
print(param_comb[200]) # printing hyperparameters with index 1


# What seems to be causing large accuracy dips in our graph are low values of **n_estimators** and from a certain value (in this case **25**), it's slowly converging to the maximum accuracy it can achieve with other given hyperparameters (in this case **max_depth**).
# As for **max_depth**, the graph indicates that it's value of **10** and greater is overfitting our model.

# #### AdaBoost Classifier

# In[ ]:


# chosen range of hyperparameters for AdaBoost Classifier
param_grid = {
    'n_estimators': range(1, 1000, 25),
    'algorithm': ['SAMME', 'SAMME.R'],
}
# using sklearn's ParameterGrid
param_comb = ParameterGrid(param_grid)


# In[ ]:


# iterate each parameter combination and fit model
val_acc = []
train_acc = []
for params in param_comb:
    dt = AdaBoostClassifier(**params)
    dt.fit(Xtrain, ytrain)
    train_acc.append(metrics.accuracy_score(ytrain, dt.predict(Xtrain)))
    val_acc.append(metrics.accuracy_score(yval, dt.predict(Xval)))


# In[ ]:


# plotting how accurate our model is with each hyperparameter combination
plt.figure(figsize=(20,6))
plt.plot(train_acc,'or-')
plt.plot(val_acc,'ob-')
plt.xlabel('hyperparametr index')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])
best_params = param_comb[np.argmax(val_acc)]
best_params


# In[ ]:


print(param_comb[0]) # printing hyperparameters with index 0
print(param_comb[50]) # printing hyperparameters with index 1
print(param_comb[75]) # printing hyperparameters with index 75


# In contraty to the **Decision Tree Classifier**, the hyperparamemter **n_estimators** with a higher value is overfitting our model (starting at around **250** for the SAMME algorithm). The same happens for the **SAMME.R** algorithm in a more severe manner.

# ### 3.1.2 GridSearchCV
# <a id='gridsearchcv'></a>
# Using our aproximations of relevant hyperparameters from the previous section, we can now define our function with **GridSearchCV** more optimally.
# 
# We will also put it in a function to call later on.

# In[ ]:


def rank_classifiers(data):
    rd_seed = 333
    # specify the data we want to predict
    Xdata = data.drop(columns = ['Survived'])
    ydata = data['Survived']
    # ready Classifiers and their distinctive range of parameters for
    # hyperparameter tuning with cross-validation using GridSearchCV
    pipeline = Pipeline([
        ('clf', DecisionTreeClassifier()) # placeholder classifier
    ])
    # narrowed down hyperparameters
    parameters = [
        {
            'clf': (DecisionTreeClassifier(),),
            'clf__max_depth': range(1, 15), 
            'clf__criterion': ['gini', 'entropy'],
        }, {
            'clf': (RandomForestClassifier(),),
            'clf__max_depth': range(1, 10), 
            'clf__n_estimators': range(25, 500, 25),
        }, {
            'clf': (AdaBoostClassifier(),),
            'clf__n_estimators': range(1, 250, 10),
            'clf__algorithm': ['SAMME', 'SAMME.R'],
        }
    ]
    # run GridSearchCV with training data to determine the best 
    # classifier with the best parameters while cross-validating them
    # thus maximizing fairness of score rankings
    clf = GridSearchCV(pipeline, parameters, cv=5, iid=False, n_jobs=-1)
    clf.fit(Xdata, ydata)
    print('accuracy score (train): {0:.6f}'.format(clf.best_score_))
    # now lets see how well it predicts our testing data
    return clf


# In[ ]:


clf = rank_classifiers(train1)


# In[ ]:


print(clf.best_params_)


# And there is our winner: **RandomForestClassifier**,
# With parameters: **max_depth = 1** and **n_estimators= 275**, with an accuracy score of: **0.94** for training data and **0.75** for testing data.
# 
# Let's see what happens if we clean our data a little better this time.

# # 4. Cleaning data try 2
# 
# ## 4.1 Missing data

# In[ ]:


# print sum of missing values for each column
for dataset in [train, test]:
    print(dataset.isna().sum())


# **Cabin:**
# After further investigation, it seems that the letter in the cabin number represents a deck. We can thus extract each letter and create a new feature for every passanger.

# In[ ]:


import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

for dataset in [train ,test]:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    # we can now drop the cabin feature
    dataset.drop(columns=['Cabin'], inplace=True)


# **Age**: Generate random ages. Generated ages will be around the mean of ages, with the maximum difference of stadart deviation (std).
# 
# Another idea is to fill the missing values using another ML model. We will do that later and see if it has any impact on the prediction of survivability.

# In[ ]:


for dataset in [train, test]:
    mean = dataset["Age"].mean()
    std = dataset["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # create an array of random numbers given the range and length
    rand_ages_for_nan = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in the column age with random values generated
    ages_slice = dataset["Age"].copy()
    ages_slice[np.isnan(ages_slice)] = rand_ages_for_nan
    dataset["Age"] = ages_slice.astype(int)


# **Embarked**: This column has only two missing values, which we can just fill with the most common value.

# In[ ]:


# describe data to find the most frequent one
for dataset in [train, test]:
    print(dataset.Embarked.describe())


# In[ ]:


# fill missing values with the frequent value
for dataset in [train, test]:
    dataset.Embarked.fillna('S', inplace=True)


# There is only one missing value left in the **Fare** column of test data so we will just use the mean for that.

# In[ ]:


test.Fare.fillna(test.Fare.mean(), inplace=True)


# In[ ]:


# final check of missing values
for dataset in [train, test]:
    print(dataset.isna().sum())


# ## 4.2 Creating new features based on other features

# Firsty, lets drop the column **PassengerId** since it does not corelate with survivability for obvious reasons.
# 
# Note: We will only drop the **PassengerId** column for train dataset, since we will be training our model on this dataset. We will keep the **PassengerId** on the test dataset since the submission requires it.

# In[ ]:


train.drop(columns=['PassengerId'], inplace=True)


# **Name**: We can extract **Titles** from the **Name** column and create a new feature.

# In[ ]:


for dataset in [train, test]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-z]+\.)', expand=False)
    # print added Titles
    print(dataset.Title.value_counts())


# Apart from the **title**, which we saved as a new feature, there doesn't seem to be any more value in the **name** column, thus we can drop that.

# In[ ]:


for dataset in [train, test]:
    dataset.drop(columns=['Name'], inplace=True)


# **Sibsp and Parch**: These features together mean the number of **Relatives** on board. We can also create another feature called **Not_alone** out of this aswell.

# In[ ]:


for dataset in [train, test]:
    # create relatives feature by summing up sibsp and parch
    dataset['Relatives'] = dataset.SibSp + dataset.Parch
    # create not_alone feature, where it is 1 if # of relatives is 0
    dataset.loc[dataset['Relatives'] > 0, 'Not_alone'] = 0
    dataset.loc[dataset['Relatives'] == 0, 'Not_alone'] = 1


# ## 4.3 Converting string features
# Lets see what columns are of type *string*. Keep in mind that in **pandas**, type *string* is in the class *object*.
# 
# We can use our predefined function from before.

# In[ ]:


for dataset in [train, test]:
    string_to_int(dataset)


# ## 4.4 Categorizing continuous data

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# As we can see, most of our features are categorized (small number of unique values). The only features left to categorize are: **Age**, **Ticket** and **Fare**.

# Let's categorize **Age** as follows:
# - Infants(0): x <= 1,
# - Children(1): 1 < x <= 13,
# - Youth(2): 13 < x <= 24,
# - Young_adults(3): 24 < x <= 35,
# - Adults(4): 35 < x <= 48,
# - Older_adults(5) 48 < x <= 65,
# - Seniors(6) 65 < x.

# In[ ]:


for dataset in [train, test]:
    dataset.loc[dataset.Age > 65, 'Age_cat'] = 6
    dataset.loc[dataset.Age <= 65, 'Age_cat'] = 5
    dataset.loc[dataset.Age <= 48, 'Age_cat'] = 4
    dataset.loc[dataset.Age <= 35, 'Age_cat'] = 3
    dataset.loc[dataset.Age <= 24, 'Age_cat'] = 2
    dataset.loc[dataset.Age <= 13, 'Age_cat'] = 1
    dataset.loc[dataset.Age <= 1, 'Age_cat'] = 0
    # drop age column
    dataset.drop(columns=['Age'], inplace=True)


# For **Fare** feature, we cannot easily categorize values with our real-world knowledge. For this case, we will use **[pandas.qcut](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.qcut.html)**, which discretizes variables into equal-sized buckets.

# In[ ]:


for dataset in [train, test]:
    dataset.Fare = pd.qcut(dataset.Fare, 6, labels=[0,1,2,3,4,5]).astype(int)


# **Ticket** feature has too many unique values, creating too many categories. It would be hard to make it a useful feature so we will drop it

# In[ ]:


for dataset in [train, test]:
    dataset.drop(columns=['Ticket'], inplace=True)


# Final check:

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# # 5. Correlation matrix
# We can plot the correlation matrix to see how relevant are our new features.

# In[ ]:


plt.subplots(figsize = (15,10))
sns.heatmap(train.corr(), annot=True,cmap="RdYlGn_r")
plt.title("Feature Correlations", fontsize = 18)


# # 6. Train classifiers on better cleaned data
# We can now use our function **[rank_classifiers](#gridsearchcv)** with **GridSearchCV**, which we have defined earlier in this work.

# In[ ]:


clf = rank_classifiers(train)


# In[ ]:


print(clf.best_params_)


# Our scores are significantly higher this time at **0.830**. Best classifier is still **RandomForestClassifier** with parameters: **max_depth = 5** and **n_estimators = 375**.

# # 7. Hyperparameter importance
# The **RandomForestClassifier** model allows us to plot out the importance of hyperparameters. 

# In[ ]:


importances = pd.DataFrame({'feature':train.drop(columns=['Survived']).columns,'importance':np.round(clf.best_params_['clf'].feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()


# # 8. Submission
# All that is left to do is to fit our model on the test data and submit our predictions.

# In[ ]:


# extract PassengerId
id_col = test.PassengerId
# predict survival
test = test.drop(columns=['PassengerId'])


# In[ ]:


# create submission dataframe
submission = pd.DataFrame({
    'PassengerId': id_col.values,
    'Survived': clf.predict(test)
})


# In[ ]:


# save submission
submission.to_csv('submission.csv', index=False)

