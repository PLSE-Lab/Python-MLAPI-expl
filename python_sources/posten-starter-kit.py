#!/usr/bin/env python
# coding: utf-8

# # Posten hackathon starter kit
# This kernel is meant as a starting off point for the Posten hackathon on `17.12.18`. 

# # Titanic data set preparations
# 
# Start by importing the libraries we will use and read the data sets from the disk.
# 
# - The train data set is the part of the data set for which we know if a passenger survived
# - The task is to predict which passengers in the test data set who survived
# 
# It's easy to solve this using google, but a lot more fun to solve it using machine-learning.
# 
# This notebook contains a demo for a simple machine learning models and visualizations that you can use to predict a result. Feel free to fork the notebook and improve it by making the machine learning models consider more attributes of the passengers.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
from matplotlib import cm

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Investigating the data sets
# 
# The data has been loaded as a pandas dataframe, full [API documentation](http://pandas.pydata.org/pandas-docs/stable/)
# 
# Here's a short demo of what pandas can do.

# In[ ]:


train.info() # basic information about the data set, such as how many values that are null and how much memory the data occupies


# We can see there are 891 rows (passengers) in the data set, and we have 12 attributes to work with.  Some are numeric and others are strings ( `object`).  Not all the attributes are complete, e.g. `Age` where we only have 714 non-null values.

# Machine learning algorithms works best on numeric data, so let's try to convert some of the strings in the data set.  For instance, let's try to rewrite the data for Sex so that 0 means male and 1 means female:

# In[ ]:


train['Sex'] = (train['Sex'] == 'female').astype(int)


# # Checking out correlations of the numeric variables

# In[ ]:


plt.figure(figsize=(18, 12)) # make the plot 18 by 12 inches
sns.heatmap(train.corr(), cmap=cm.coolwarm) # plot it


# Studying the correlation matrix is a good way to find out which variables in the data that can help us predict whether a passenger survived.  Keep in mind that a strong negative correlation (aka the dark blue squares) are still strongly correlated.  Values close to 0 are the ones we don't care too much about.
# 
# By looking at the Survived column, we can see that it correlates well with Sex, meaning we can tell a lot about whether a person survived but looking at their gender.  We'll use this to our advantage later.

# Let's look at that discovery in another way:

# In[ ]:


gender_by_survived = train.groupby(['Sex', 'Survived']) # group it
# count the groups, could also use other aggregates
gender_by_survived = gender_by_survived.size()
# unstack it to make the survived or not counts into columns and the gender group into rows
gender_by_survived = gender_by_survived.unstack()
gender_by_survived.plot.bar(figsize=(18, 12))


# Here we can see why the correlation is strong:  Women are much more likely to have survived than men.

# # Making a simple model
# 
# It definitely looks like the gender is very predictive for whether someone survived. So we should use it in a model.
# 
# We will use scikit-learn to demonstrate, you can see the [supervised learning](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) section of the documentation for detailed information about how this works.

# In[ ]:


import sklearn.tree
import sklearn.linear_model


# In[ ]:


tree = sklearn.tree.DecisionTreeClassifier()
predictors = ['Sex']


# The decision tree is the simplest possible model. It can work with data in almost all formats. But it's not necessarily the best model. We'll use it to demonstrate the scikit-learn API.
# 
# To train a scikit-learn model, call the `.fit()` method with the attributes you want the tree to do learning on, and the answers you want it to predict for that set of attributes:

# In[ ]:


tree.fit(train[predictors], train['Survived'])


# Now we can use the model to make predictions. Let's see if it gets everything right:

# In[ ]:


train['prediction'] = tree.predict(train[predictors])
train[['Survived', 'prediction']].head()


# Looks good for those few values, the prediction matched survived.  We can check the entire dataset in a single call like this:

# In[ ]:


tree.score(train[predictors], train['Survived'])


# # But don't do that
# 
# Does that mean we're done?
# 
# Short answer: no
# 
# The reason why it's getting so many right is that the model is simply remembering every data point in the data set. To see how well it works, we actually need to show it some data it hasn't already used for learning. To do that, we need to split the data set before we train. Here's one way to do that:

# In[ ]:


train_on = train.iloc[:800] # train on the first 800 samples
validate_on = train.iloc[800:] # save the rest for validation
tree = sklearn.tree.DecisionTreeClassifier() # make a new model
tree.fit(train_on[predictors], train_on['Survived']) # train it only on part of the data set
tree.score(validate_on[predictors], validate_on['Survived']) # check using data the model hasn't seen


# Still very good, and shows that we've picked out some pretty important attributes.
# 
# But this is a very tedious way to check how good the model is doing, so here's a better one: **cross validations**

# In[ ]:


import sklearn.model_selection

sklearn.model_selection.cross_val_score(tree, train[predictors], train['Survived'], cv=5)


# What we just did, using the `cv=5` parameter was to ask for 5 cross validation. For each cross validation, scikit-learn splits the data set into 5 pieces, trains the model on 4 of them and validates on the last. That's why we got 5 scores out here. We can see that the model does better on some of the cross validations than others. But we have a pretty good indication that we can get about 75% of the data set correct.

# # Submit to kaggle
# 
# Let's submit our prediction to kaggle. We'll do that by predicting on the test data set, for which we don't know the answer, and write it to a CSV file.
# 
# To do the prediction, we need to make the exact same changes to the `test` data set as we've done to the `train` data set, or our model won't be able to make sense of it.

# In[ ]:


test['Sex'] = (test.Sex == 'female').astype(int)

# Fit the model to the training data set
tree.fit(train[predictors], train['Survived'])
test['Survived'] = tree.predict(test[predictors])

test[['PassengerId', 'Survived']].head()


# Now we've written 'Survived' attribute for the test set, it's time to submit to kaggle. We can do that by writing a CSV file to the current working directory, containing _only_ the PassengerId and Survived column:

# In[ ]:


test[['PassengerId', 'Survived']].to_csv('predictions.csv', header=True, index=False)


# You can't actually see the file yet, you'll have to commit the notebook and run it first.  When you've done that you can see the output and submit it to the competition.  However, there's a limit to how often you can do this, so please use the train dataset and cross validations to check if your model looks improved.

# # Where to next?
# 
# I've already submitted this, so I know it'll give a decent score, but not a great one.  There's a lot of things we can do to improve on it, but that's what I'm hoping you will do today.  Here are some ideas:
# 
# - See if another classifier will do better.  Decision trees are great because they're simple, not because they give the best results.
# - We've just looked at a single attribute, `Sex`, that's wasting a lot of data.  See if you can find other attributes which improve the model
# - There is a lot of data in the strings, see if you can extract something useful from those.  Perhaps the title in the `Name` column can be used for something?
# 
# Can you get to 80%?  It's certainly possible, but it'll take a lot of work and some smart problem solving.
# 
# **Good luck, and have fun**
