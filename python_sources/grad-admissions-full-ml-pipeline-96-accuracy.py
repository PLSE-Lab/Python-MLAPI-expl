#!/usr/bin/env python
# coding: utf-8

# # Graduate Admissions Data:
# 
# ## An Exploration & Endeavor To Predict Acceptance Probabilities
# 
# What do the chances for getting accepted into grad school look like for a prospective student?
# 
# In this notebook, we will attempt to aswer this question by:
# 
# - Implementing data visualization techniques
# - Utilizing Data Cleaning / Features Engineering
# - Predicting outcomes with a machine learning model
# 
# Lets get to it!

# ## 1) Standard Imports and Loading the Dataset

# Obvious start, we'll begin with some standard imports for working with data and load the dataset itself:

# In[ ]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


# Now loading in data with Pandas
df = pd.read_csv('../input/Admission_Predict.csv')

print('dataframe shape: {}'.format(df.shape))
df.head(3)


# In[ ]:


# Now getting a better look at what each column represents
df.info()


# So we're off to a good and clear start. Here is what we can tell so far:
# 
# - __Data Shape:__ By looking at the first few entries and dataframe shape, we can see that we are working with 400 entries, each one with 9 features.
# - __Feature Data Types:__ All of our features are _numerical_, meaning they are quantified measurements which we can perform mathematical operations on without doing any kind of special conversion.
# - __Amount of Data:__ As we just mentioned, we have only 400 entries, and this is important to take note. It means that when it comes time to implement some machine learning model, _we should avoid neural networks_. Why? Neural networks only start to work well with large datasets (at least a few thousand entries), and our 400 entries now won't cut it. We will have to work with more traditional techniques to see good results
# 
# Great! Before we can go into some further data exploration and begin visualization techniques, lets do a little housekeeping:

# ## 2) Basic Data Cleaning

# We haven't looked too closely at the data yet to get involved in faeatures engineering, but we can already start with some items that will make our life easier. Right off the bat, this is what we can do:
# 
# - Get rid of the 'Serial No.' Column, as it only serves the purpose of identifying entries and would not contribute to data exploration/visualization/predicitons
# - Standardize the column names, for ease of access. We will go ahead and use *snake_case*. Also, uncommon acronyms will be replaced for their meaning (like 'LOR' for 'Letter of Recommendation'), but keep common ones, like GPA

# In[ ]:


# Dropping 'Serial No.'
df = df.drop(columns=['Serial No.'])


# In[ ]:


# Standardizing column names
df.columns = ['GRE_score', 'TOEFL_score', 'university_rating', 'statement_of_purpose', 'letter_of_recommendation', 'GPA', 'research', 'chance_of_admit']


# Again, lets take a look at our first three entries. We should now only see the _Serial No._ column gone and the columns renamed:

# In[ ]:


df.head(3)


# Awesome! Lets move on to some further exploration and visualization

# ## 3) Data Exploration / Visualization

# ### Acceptance Distributions

# First things first, lets take a quick look at the distribution of acceptance probabilites across our dataset. For this, we will ceate a __histogram__

# In[ ]:


df['chance_of_admit'].hist(figsize=(10,4))


# Here we can see that it _roughly_ follows a normal (gaussian ditribution), skewed to the left.
# 
# We can confirm our observation by using a _density plot_

# In[ ]:


df['chance_of_admit'].plot(kind='density', subplots=True, figsize=(10, 4));


# What does this mean? It means that for most of our data entries, our distribution leans to most of them having higher probabilities of grad school acceptance.
# 
# This could pose a problem in our analysis and any future predictions because of a possible __bias__ in a test set. Our bias could potentially be that we give candidates a higher probability score than what it actually should be.
# 
# However, given the current dataset, we cannot extrapolate to what a healthy and realistic distribution would look like. We will continue working with the probabilites at hand, but also keep this potential bias problem on our minds.

# ### Correlations
# 
# Since we're dealing with only a few number of features, lets implement a _correlation matrix_. A correlation matrix is great to get a high level view of our features and any possible relationships between them:

# In[ ]:


# Creating a correlation matrix:
corr_matrix = df.corr()
# Plotting heatmap
sns.heatmap(corr_matrix);


# The heatmap allows us to note the stronger correlations by lighter-colored tiles. 
# 
# We can see that *chance_of_admit_*, *GRE_score*, *TOEFL_score*, and *GPA*, are closely related. If one of these move in a certain direction (increase or decrease in value), it is very likely that the other features will follow in that direction.
# 
# Higher University scores also seem to influence *chance_of_admit*, but not as much as GPAs and test scores.
# 
# Interestingly, we note that *research* seems to be weakly correlated with our *chance_of_admit*. Its important to remember that this is binary value, with 1 representing experience in research, while 0 represents no experience. Soon, we will take a closer look and see how they add up.

# ### Quantitative Relationships w/ Scatter Plots
# 
# We can spend all day looking deeper into certain data relationships, but we'll keep our current analysis only down to a few.
# 
# The strongest relationship we noted in the correlation matrix for *chance_of_admit* was with *GPA*. Lets start there.
# 
# We will use a _jointplot_, which will mainly display a _scatterplot_ for our correlation, but also contain a distribution of each feature above its respected axis:

# In[ ]:


sns.jointplot(x='chance_of_admit', y='GPA', data=df, kind='scatter');


# As noted before,we see the correlation that as *GPA* increases, so does *chance_of_admit*. We also can see the distribution for columns. *GPA* seems to follow a more standard Gaussian distribution than *chance_of_admit*, and it is not as skewed.
# 
# But this one seems __quite obvious__. What about that tricker relationship, regarding *research*? Lets take a look at it now

# ### Quantitative-Cateogrical Analysis with Research and Acceptance

# We just used a scatter plot to visualize the relationship between *GPA* and *chance_of_admit*. But how about research? Here we implement the same scatterplot visualization, but we discriminate them by research status

# In[ ]:


sns.lmplot('chance_of_admit', 'GPA', data=df, hue='research', fit_reg=False);


# _Interesting!_ Based on our new findings, we can see that __almost all candidates with higher chances of admission have participated in research work__.
# 
# This is a great example of needing to use various visualization tools to get a better understanding of our data. If we only had relied on the correlation matrix, we would have concluded that research would not have mattered for grad school acceptance.

# We've achieved our basic goal of data exploration and visualization. Its always a good idea to look further and do more visualizations, but we will stop for now and move on to seeing how we can predict acceptance probabilities for candidates

# ## 4) Building a Machine Learning Model for Acceptance Predictions
# 
# Time for everyone's favorite part, machine learning! 
# 
# Before we go crazy throwing all kinds of ml models at the problem, let's break down what exactly we are trying to do here:
# 
# __Given all features, given *chance_of_admit*, predict the *percentage* of *chance_of_admit*__
# 
# From our problem statement, we can identidy the following:
# 
# - Predicting based on an already given label is a _supervised learning_ problem
# - Predicting based on a percentage can be modeled as a _regression_ problem
# 
# This makes our initial endeavor into ML for this dataset straightforward: _We will begin with a simple linear regression model to predict chance of admission._ 

# ### A Note on Picking ML Models

# Some might scoff at picking to use such a simple model to predict acceptance rates.
# 
# While linear regression is one of the simplest models available for implementation, it is also quite effective and has a fast iteration cycle. Part of our job as machine learning engineers and data scientists is to work with _iteration_.
# 
# You could begin with a more complicated model, such as a _Support Vector Machine_, but configuration is not as straight forward, and you won't have a universal baseline of performance.
# 
# Linear Regression is so widely used, that it serves as a stronger baseline of how effectively you can make predictions. With time, we can iterate and implement more complex models and analyze the improvements that may come with them.

# ### Importing sklearn ||  Seperating our Feature Inputs and Target
# 
# To implement linear regression, we will import the existing function from scikit learn, one of the landmark libraries for data science programming in Python. We will also go ahead and import *metrics*, which will allow us to quantify our predictions based on their actual results.

# In[ ]:


# taking care of our ML imports
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# It is common to denote our features by **X**, and our target variable as __y__. We will follow this common notation here when creating our training set

# In[ ]:


# Seperating our features and our target.
train_features = list(set(df.columns) - set(['chance_of_admit']))

train_X = df[train_features]
train_y = df['chance_of_admit']


# ### Initializing and Training our Regression Model
# 
# Now that our training set has been properly allocated, we can actually create an instance of our model and begin training!

# In[ ]:


# Create LinearRegression instance
linear_regression = LinearRegression()

# Begin training! This is also knows as 'fitting' the model to our data
linear_regression.fit(train_X, train_y)


# ### Analyzing Coefficients
# 
# Now that it has been trained, we can see how heavily each feature influenced our linear regression model. This will be good insight to se understand whats happening under the hood:

# In[ ]:


pd.DataFrame(linear_regression.coef_, train_X.columns, columns=['Coefficient'])


# As we can see, GPA was the most influential factor for our linear regression model, with research coming in second.
# 
# Its also interesting to see that GPA outweighs all the other features, by almost as much as 5 times compared to research.
# 
# Now that the model has been fitted, lets evaluate it with our test set

# ## 5) Evaluating our Model

# Evaluation is crucial to measure the performance of our regression model. For this we need two things:
# 
# - A test set
# - A criterion, also known as a loss function
# 
# For our test set, we will load that in as it has already been prepared. For the criterion, we will use the __MAE (Mean Absolute Error) Loss Function__.

# ### Test Set Loading & Formatting

# In[ ]:


# Loading in Test set
test_df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

test_df.head(3)


# For the sake of consistency, we will convert our column names for the test dataframe and drop *Serial No.*.

# In[ ]:


test_df = test_df.drop(columns=['Serial No.'])
test_df.columns = ['GRE_score', 'TOEFL_score', 'university_rating', 'statement_of_purpose', 'letter_of_recommendation', 'GPA', 'research', 'chance_of_admit']


# In[ ]:


test_df.head(3)


# Now we've followed the same structure as our training data

# ### Test Formatting, Evaluation Score
# 
# We set up our test and run the model

# In[ ]:


test_X = test_df[train_features]
test_y = test_df['chance_of_admit']


# In[ ]:


y_pred = linear_regression.predict(test_X)


# Before using our actual loss function, lets take a manual look of how our predictions compare to the actual results

# In[ ]:


pd.DataFrame({'Prediction': y_pred, 'Actual': test_y}).head(10)


# We can see that our model is making reasonable assumptions about admission rates! While not perfect, this is a good first step with a linear regression model.
# 
# Let's finally take an accurate score with the __Mean Absolute Error__

# In[ ]:


print("Mean Absolute Error: {}".format(metrics.mean_absolute_error(test_y, y_pred)))


# ## Conclusion

# With this notebook, we created a full machine learning pipeline, from mapping out an initial dataframe to making reasonable predictions about a target variable.
# 
# We dealt with mostly straighforward data, with not many dimensions regarding our features. This allowed us to fit an incredibly effective but simple linear regression model that brought with it strong prediction results. To recap on some highlights/ lessons learned:
# 
# - Predictions for small datasets are not viable for deep learning techniques. Instead, we choose to go with traditional ML algorithms.
# - Data Cleaning/ Feature Engineering helps us in standardizing our dataset, and makes exploration/visuazization a much easier task
# - Never settle for one particular visualization. At first glance, research was not too important when looking at grad admission rates. After looking at a colored scatterplot, we discovered otherwise.
# - Machine Learning modelling should be an iterative process; start simple, and as needed work with more advanced algorithms and configurations.
