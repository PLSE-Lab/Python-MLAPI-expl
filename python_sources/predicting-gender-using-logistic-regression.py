#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# In this notebook we'll explore the following questions:
# 
# 1. How effective was the test-preparation course for the students' revision?
# 2. How does a student's parental education level affect their exam scores?
# 3. Can we predict a student's gender based on their exam scores and other attributes?

# # **Importing Packages & Data**

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()


# In[ ]:


print('Shape of dataframe:', df.shape)


# # **Data Cleaning & Feature Engineering**

# First we'll check if there are any missing entries that need to be dealt with. Luckily in this dataset there aren't.

# In[ ]:


df.isnull().sum()


# Next, there are a few changes we can make to help prepare the data for analysis, and to make it slightly more readable:
# 
# 1. Rename columns to a simpler form.
# 2. Remove unnecessary information from 'race' column.
# 3. Create an average score column to use in later analysis.

# In[ ]:


# 1
df.columns = ['gender', 'race', 'parent_education', 'lunch', 'test_prep', 'math_score', 'reading_score', 'writing_score']

# 2
df['race'] = df.race.apply(lambda x: x[-1])

# 3
df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3

df.head()


# The last piece of the dataset that needs cleaning is the 'parent_education' column.

# In[ ]:


# count of each parent_education entry
df.groupby(['parent_education']).gender.count()


# Notice that we have entries labelled both 'high school' and 'some high school'. These entries can be grouped into the same category.
# 
# We'll also rename 'some college' entries to 'college'.

# In[ ]:


df['parent_education'] = df.parent_education.apply(lambda x: 'high school' if x == 'some high school' else ('college' if x == 'some college' else x))
df.head()


# # **Initial Analysis**

# **The data is now ready for us to perform some initial analysis.**

# First, we use scatterplots to see the correlation between each of the subject scores. We can also separate these plots by gender to see how they differ.

# In[ ]:


fig, axs = plt.subplots(figsize=(22,6), ncols=3)
fig.subplots_adjust(wspace=0.23)

sns.scatterplot(x='math_score', y='reading_score', hue='gender', data=df, ax=axs[0])
sns.scatterplot(x='math_score', y='writing_score', hue='gender', data=df, ax=axs[1])
sns.scatterplot(x='reading_score', y='writing_score', hue='gender', data=df, ax=axs[2])


# We see that men tend to have a slightly higher math score than reading or writing score, and women tend to be the opposite. 
# 
# We also see that no students performed extremely well in one subject and badly in another.

# Next we check the distributions of the scores achieved in each exam.

# In[ ]:


fig, axs = plt.subplots(figsize=(18,5), ncols=3)
fig.subplots_adjust(wspace=0.3)

sns.distplot(df.math_score, ax=axs[0])
sns.distplot(df.reading_score, ax=axs[1])
sns.distplot(df.writing_score, ax=axs[2])


# Below we use a boxplot to compare how much of an impact the test-preparation exam had for each subject.

# In[ ]:


fig, axs = plt.subplots(figsize=(15,6), ncols=3)
fig.subplots_adjust(wspace=0.5)

sns.boxplot(x='test_prep', y='math_score', data=df, ax=axs[0], fliersize=2)
sns.boxplot(x='test_prep', y='reading_score', data=df, ax=axs[1], fliersize=2)
sns.boxplot(x='test_prep', y='writing_score', data=df, ax=axs[2], fliersize=2)


# Overall it seems that the test-preparation exam was effective revision for every subject, most affecting the scores in the writing exam.

# Lastly, we use another boxplot to compare how a student's parental education level affects their average exam score.

# In[ ]:


fig, axs = plt.subplots(figsize=(9,7))

sns.boxplot(x='parent_education', y='avg_score', data=df, fliersize=0)
sns.swarmplot(x='parent_education', y='avg_score', data=df, color='0')


# # **Logistic Regression**

# **Now we'll try to predict a student's gender based on their exam scores and other attributes.**

# There are a few changes we must make to the data before we are ready to fit the model.

# In[ ]:


log_df = df.copy()
log_df.head()


# First, we'll convert the 'gender', 'lunch' and 'test_prep' columns into binary. 

# In[ ]:


log_df['gender'] = log_df.gender.apply(lambda x: 1 if x == 'male' else 0)
log_df['reduced_lunch'] = log_df.lunch.apply(lambda x: 1 if x == 'free/reduced' else 0)
log_df['test_prep'] = log_df.test_prep.apply(lambda x: 1 if x == 'completed' else 0)

# removing 'lunch' and 'avg_score' columns
log_df = log_df.drop(['lunch', 'avg_score'], axis=1)

log_df.head(3)


# Next we need to use one-hot encoding on the 'race' and 'parent_education' columns so that the data can effectively be fed into the model.

# In[ ]:


race_df = pd.get_dummies(log_df.race)
ed_df = pd.get_dummies(log_df.parent_education)
log_df = pd.concat([log_df, race_df, ed_df], axis=1)

log_df = log_df.drop(['race', 'parent_education'], axis=1)

log_df.columns = ['gender', 'test_prep', 'math_score', 'reading_score', 'writing_score', 
                  'reduced_lunch', 'race_A', 'race_B', 'race_C', 'race_D', 'race_E', 
                  'p_associates', 'p_bachelors', 'p_college', 'p_high_school', 'p_masters']

log_df.head(3)


# We'll also convert the scores into decimals so that their scale is in keeping with the rest of the data. This stops the regression model from assigning skewed weight towards the score variables.

# In[ ]:


scores = ['math_score', 'reading_score', 'writing_score']
for i in scores:
    log_df[i] = log_df[i]/100

log_df.head(3)


# Our target variable is 'gender', and the other features are our predictor variables.
# 
# We split the dataframe into training data (80%), and test data (20%).

# In[ ]:


predictors = list(log_df.columns)
predictors.remove('gender')

X_train, X_test, y_train, y_test = train_test_split(log_df[predictors], log_df['gender'], 
                                                    test_size=0.2, random_state=38)
print('Training data:', X_train.shape, '\nTest data:', X_test.shape)


# Now it's time to fit the model. We use scikit-learn's GridSearchCV function to find the best hyperparameters to use.
# 
# The model is trained using our training data, and then evaluated on the test data to obtain an accuracy score.

# In[ ]:


log = LogisticRegression()

# parameter space
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 20)
log_params = dict(C=C, penalty=penalty)

# grid search
log_clf = GridSearchCV(log, log_params, cv=5, verbose=0)
best_log_model = log_clf.fit(X_train, y_train)
log_score = best_log_model.score(X_test, y_test)

# tuned parameters
print('Best Penalty:', best_log_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_log_model.best_estimator_.get_params()['C'])
print('\nModel Accuracy:', log_score)


# Below is a table showing how many of the model's predictions were correct.

# In[ ]:


log_predictions = best_log_model.predict(X_test)
pd.crosstab(y_test, log_predictions, rownames=['Actual'], colnames=['Predicted'])


# Lastly, let's compare which of our features the model assigned most weight to. This will give us an idea of what features were most influential in our model's decisions.
# 
# The metric we'll use for this comparison is the magnitude of the feature's coefficient in our model, multiplied by the standard deviation of the data in the corresponding column.

# In[ ]:


logistic = LogisticRegression(penalty = best_log_model.best_estimator_.get_params()['penalty'], 
                              C = best_log_model.best_estimator_.get_params()['C'])
logistic.fit(X_train, y_train)

feature_importance = abs(np.std(X_train, 0) * list(logistic.coef_[0]))
n = len(feature_importance)

fig, axs = plt.subplots(figsize=(12,5))
sns.barplot(x = feature_importance.nlargest(n).index, 
            y = feature_importance.nlargest(n))
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
axs.set(xlabel='Feature', ylabel='Feature Importance')


# As expected, the exam scores were the most influential features, followed by the completion of the test-preparation course.
