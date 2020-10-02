#!/usr/bin/env python
# coding: utf-8

# # A Solution to Bank Telemarketing Predictions
# 
# ## Introduction
# 
# This notebook was completed as a response to
# <a href="https://www.kaggle.com/c/predicting-bank-telemarketing/overview">this</a> Kaggle competition
# 
# ### Goal
# 
# The focus of this Kaggle competition is to target clients through telemarketing to sell long-term deposits. The data, which was collected from 2008 to 2013, contains demographic and personal information about each client. Our goal is to correctly guess whether or not a client will buy long-term deposits given this information 
# 
# ### Data
# 
# #### Necessary Import Statements

# In[ ]:


### Necessary Imports
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split


# - The data was given in the form of training, testing, and a sample sumbission (as an example).
#     - The training data has the column "duration," which indicates the duration of a call with the client. As we will not know the duration of a call **before** calling a client, this data is not included in our testing data. Because of this, we will remove it from the training set
#     - Below, we can see that the training data has a column labeled 'y' with the result. 1 indicates a success (the client purchased the deposit), and 0 indicates a failute (the client did not purchase the deposit)
#     
# #### Reading in the Data

# In[ ]:


samp = pd.read_csv('../input/predicting-bank-telemarketing/samp_submission.csv')
train = pd.read_csv('../input/predicting-bank-telemarketing/bank-train.csv')
test = pd.read_csv('../input/predicting-bank-telemarketing/bank-test.csv')
train.drop(columns = 'duration')


# In[ ]:


train.head()


# In[ ]:


test.head()


# #### Probbilistic Distributions
# 
# From the head of the training file, we can see that none of the first 5 clients purchased deposits. Here, I will look at the percentage of clients who declined the offer, and the percentage that accepted.
# - **Failure (0) :** 88.76%
# - **Success (1) :** 11.24%
# 
# Clearly, not many people liked the offer. We can use this information to infer that most of the time, a client is most likely to say no.

# In[ ]:


### This gives us the probability of each occurance
train['y'].value_counts(1)


# ## Random Guess
# 
# If you were to guess randomly using these distributions, this is how you would do it:
# - This attempt yielded me a "success score" of around 0.8. This seems pretty good, but it's only because it is very easy to guess a failure. If the failure rate is 0.88%, and I guess "failure" 88% of the time, then I would correctly theoretically predict 77.44% of the failures simply by guessing
# 
# Here is my code for determining the random predictions:

# In[ ]:


### In the sample data (which only has the client id), we randomly assign success values based off the probabilities shown above
samp.Predicted = np.random.choice(range(2), size = samp.shape[0], p = [train['y'].value_counts(1)[0], train['y'].value_counts(1)[1]])


# In[ ]:


samp.to_csv('first_test.csv', index = False)


# ## Decision Tree

# ### Removing NULL Rows
# - Rows with NULL values can screw up our decision tree. Here, will will remove any rows that have NULL values

# In[ ]:


print(train.columns)
train.dropna()


# ### Creating Dummy Variables
# - Some of our columns contain categorical variables, like job status and marital status. Unfortunately, the decision tree can't handle categorical variables
# - By creating dummy variables, we create a new column for each category in the original column
#     - For example, the **marital** column would be broken up into 3 new columns: Single, Married, and Divorced. A '1' in the Single column represents that the person is single, a '0' in the Married column represents that the person is not married, etc.
# - I added an extra column to the testing data. This is because there was no 'default_yes' value in the original default column. Because of this, I make default_yes a column of zeros

# In[ ]:


train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.head()


# In[ ]:


X = train.drop(columns = 'y')
X = X.drop(columns = 'id')
X = X.drop(columns = 'duration')
Y = train['y']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

test = test.drop(columns = 'id')
test = test.drop(columns = 'duration')
test['default_yes'] = 0


# ### Creating the tree
# - I created the tree with a maximum depth of 5. This is because with more than 20 features (columns), our tree would otherwise grow very large. Having a large tree not only slows down the algorithm and becomes confusing, but it can cause overfitting

# In[ ]:


## Fitting the tree to our testing data 
tree = DecisionTreeClassifier(max_depth = 5)
tree.fit(X_train,Y_train)


# ### Testing the Tree
# - Below we can see how well our model ran on the training and testing data we partitioned
# - About 90% isn't bad!

# In[ ]:


## Running the tree on our training and testing data
print("Training accuracy:", tree.score(X_train, Y_train))
print("Testing accuracy:", tree.score(X_test, Y_test))


# ### Most "Important" Features
# - We will sort the features based off of their **gain**, or the weight given to each feature
# - Features with higher gain more heavily impact the model

# In[ ]:


pd.DataFrame({'Gain': tree.feature_importances_}, index = X_train.columns).sort_values('Gain', ascending = False)


# ## Decision Tree with Bagging Classifier
# - Now that we've created our decision tree, we can run a bagging classifier on the tree
# - The classifier will run on a model n_estimators times. Each time the classifier runs, it selects a percentage of the original data points, with replacement. All of these attempts are then averaged together
# - For more information about bagging classifiers, look here: 
# - Our classifier improved our model from **90.51%** to **90.64%**

# In[ ]:


## Running bagging classifier on our original decision tree
bag_model = BaggingClassifier(base_estimator=tree, n_estimators=100,bootstrap=True)
bag_model = bag_model.fit(X_train,Y_train)
y_pred = bag_model.predict(X_test)
print("Training accuracy: ", bag_model.score(X_train,Y_train))
print("Testing accuracy: ", bag_model.score(X_test,Y_test))


# ### Most "Important" Features
# - We will sort the features based off of their **gain**, or the weight given to each feature
# - Features with higher gain more heavily impact the mode
# - We can see that these features are similar to, but not exactly the same as, the original decision tree

# In[ ]:



feature_importances = np.mean([
    tree.feature_importances_ for tree in bag_model.estimators_
], axis=0)

pd.DataFrame({'Gain': tree.feature_importances_}, index = X_train.columns).sort_values('Gain', ascending = False)


# ## Exporting Data
# - To export the data, we're running the bagging model on our ORIGINAL testing set. Then, we're saving the results to a csv file

# In[ ]:


predictions = pd.DataFrame(bag_model.predict(test))


# In[ ]:


samp['Predicted'] = predictions
samp.to_csv('second_test.csv', index=False)

