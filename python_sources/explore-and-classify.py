#!/usr/bin/env python
# coding: utf-8

# # Simple Exploratory Data Analysis

# This is a kernel to show how to do basic EDA and classification for custom labels.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


data = pd.read_csv('../input/world-happiness-report-2019.csv')


# ## Overview of Data

# Each column of data has the next description.

# 1. <b>Country (region)</b> Name of the country.
# 2. <b>Ladder</b> is a measure of life satisfaction.
# 3. <b>SD of Ladder</b> Standard deviation of the ladder.
# 4. <b>Positive affect</b> Measure of positive emotion.
# 5. <b>Negative affect</b> Measure of negative emotion.
# 6. <b>Social support</b> The extent to which Social support contributed to the calculation of the Happiness Score.
# 7. <b>Freedom</b> The extent to which Freedom contributed to the calculation of the Happiness Score.
# 8. <b>Corruption</b> The extent to which Perception of Corruption contributes to Happiness Score.
# 9. <b>Generosity</b> The extent to which Generosity contributed to the calculation of the Happiness Score.
# 10. <b>Log of GDP per capita</b> The extent to which GDP contributes to the calculation of the Happiness Score.
# 11. <b>Healthy life expectancy</b> The extent to which Life expectancy contributed to the calculation of the Happiness Score.

# Data is sorted in a way the country on top of the dataframe is the happiest while the last is the opposite.

# In[ ]:


data.head(10)


# ## Cleaning

# Ladder is just another word for ranking so I removed it together with SD of ladder and renamed others. 

# In[ ]:


data = data.drop(['Ladder', 'SD of Ladder'], axis=1)

data = data.rename(columns={
    'Country (region)':'Country',
    'Positive affect':'Pos',
    'Negative affect':'Neg',
    'Log of GDP\nper capita':'GDP',
    'Healthy life\nexpectancy':'Life expectancy'
})


# In[ ]:


data.shape


# Let's see how many rows are null.

# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# There are a few ways when dealing with null samples. 
# 1. Drop Rows
# 2. Fill with mean values or similar computation (e.g. forward fill)
# 3. Use K-Nearest Neighbors
# 
# Because the whole data is small anyway, I chose to just drop them.

# In[ ]:


data = data[~data.isnull().any(axis=1)]
data.shape


# ## Exploration

# Next we see the general numeric description of each feature.

# In[ ]:


data.describe()


# The data looks quite weird because we can see that all features' descriptions are almost the same. Additionally, the max value of Life expectancy is 150!

# One of many things to do at the beginning of EDA is checking if any outliers exist among samples.

# In[ ]:


fig, ax = plt.subplots(2, 4, figsize=(16, 8))
plt.tight_layout()

for i, feature in enumerate(list(data)[1:]):
    sns.boxplot(x=feature, data=data, orient='v', ax=ax[int(i-4>=0)][i%4]);


# It does not seem like there exists any outliers so we don't have to drop any samples.

# After cleaning up the data a bit, we could try if any of features are correlated.

# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(data.corr(), cmap="Blues");


# In[ ]:


data.corr()


# It seems GDP and Life expectancy are the most correlated with the value of 0.847850, among others.

# I though for a while what I could do with the data above. What I did was to evenly distribute samples and add a new feature <b>Class</b> which each sample will have either 0, 1, or 2.

# <b>0</b> shows a country is <b>happy</b>, <b>1</b> is <b>neutral</b> and <b>2</b> is <b>sad</b> (or unhappy). The smaller the value, the happier. This way, I can see which other features are correlated to it.

# In[ ]:


size = int(data.shape[0]/3)
size


# In[ ]:


data['Class'] = 0
data.iloc[size:2*size]['Class'] = 1
data.iloc[2*size:]['Class'] = 2


# In[ ]:


# Happiest countries among each group

data.iloc[[0, size, 2*size], :]


# After adding the feature, I found a few things I could try and the first thing I did was to see how much Freedome behaves when classifying the Class.

# In[ ]:


def distplot(col, bins=10):
    
    fig, ax = plt.subplots(1,3,figsize=(16, 4))

    sns.distplot(data[data['Class']==0][col], bins=10, ax=ax[0])
    ax[0].set_title('Class 0')

    sns.distplot(data[data['Class']==1][col], bins=10, ax=ax[1])
    ax[1].set_title('Class 1')

    sns.distplot(data[data['Class']==2][col], bins=10, ax=ax[2])
    ax[2].set_title('Class 2')

    plt.show();


# In[ ]:


distplot('Freedom')


# It seems that the lower the freedom value is, the happier a country is. What?

# I'm not sure how the data is computed exactly because when I checked out the original data from the original site, the features were different so I think this data was aggregated in some way I have no idea.

# But EDA continues.

# Let's do the same with Corruption.

# In[ ]:


distplot('Corruption')


# Based on the plot, Corruption affects less to the happiness of a country than the Freedom because the line plots are less indicative that Class 1 and Class 2 looks quite similar.

# I chose three features <b>Social support</b>, <b>GDP</b>, and <b>Life expectancy</b> that are highly correlated to <b>Class</b> from the heatmap plot for better information gain.

# In[ ]:


distplot('Social support')


# In[ ]:


distplot('GDP')


# In[ ]:


distplot('Life expectancy')


# All of the above plots show one common thing. The countires with Class 0 have right-skewed distribution while Class 2 have left-skewed distribution. And of course, Class 1 countries show more symmetric distribution than others.

# Now that we've found out they are correlated to Class, we explore if they are correlated themselves.

# In[ ]:


def scatterplot(x, y):
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    sns.regplot(x, y, data=data[data['Class']==0], ax=ax[0])
    ax[0].set_title('Class 0', size=15)
    ax[0].set_xlabel(x, size=15)
    ax[0].set_ylabel(y, size=15)
    
    sns.regplot(x, y, data=data[data['Class']==1], ax=ax[1])
    ax[1].set_title('Class 1')
    ax[1].set_title('Class 1', size=15)
    ax[1].set_xlabel(x, size=15)
    ax[1].set_ylabel(y, size=15)
    
    sns.regplot(x, y, data=data[data['Class']==2], ax=ax[2])
    ax[2].set_title('Class 2')
    ax[2].set_title('Class 2', size=15)
    ax[2].set_xlabel(x, size=15)
    ax[2].set_ylabel(y, size=15)
    
    plt.show();


# In[ ]:


x = 'Social support'
y = 'GDP'
z = 'Life expectancy'


# In[ ]:


scatterplot(x, y)


# In[ ]:


scatterplot(y, z)


# In[ ]:


scatterplot(x, z)


# We can do the same thing as above but this time, using all Class.

# In[ ]:


fig = sns.pairplot(data=data[['GDP', 'Social support', 'Life expectancy']])

fig.fig.set_size_inches(12, 12);


# ---

# # Classification

# Since we've added Class feature, we could build a model which predicts based on other features.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# We only have 140 samples so it may not work very well. So I held out 5 samples from each class for validation set.

# In[ ]:


# Reset index since some samples were dropped before that a few numbers skip
data.index = np.arange(data.shape[0])


# In[ ]:


# Randomly choose testing samples
happy_idx = np.random.choice(np.arange(size), size=5, replace=False)
neutral_idx = np.random.choice(np.arange(size, 2*size), size=5, replace=False)
sad_idx = np.random.choice(np.arange(2*size, data.shape[0]), size=5, replace=False)

test_idx = list(happy_idx) + list(neutral_idx) + list(sad_idx)


# In[ ]:


test = data.iloc[test_idx]
test


# In[ ]:


train = data[~data.index.isin(test_idx)]

train.shape, test.shape


# In[ ]:


def split_data(dat):
    
    X = dat.loc[:, ['Social support', 'GDP', 'Life expectancy']]
    y = dat.loc[:, 'Class']
    
    return X, y


# In[ ]:


# Only use three features
X_train, y_train = split_data(train)
X_test, y_test = split_data(test)


# In[ ]:


# Set random_state for reproducibility
clf = DecisionTreeClassifier(random_state=123)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# Using the top 3 correlated features show poor result so I used all features again.

# In[ ]:


X_train, y_train = train.drop(['Class', 'Country'], axis=1), train.loc[:, 'Class']
X_test, y_test = test.drop(['Class', 'Country'], axis=1), test.loc[:, 'Class']


# In[ ]:


clf2 = DecisionTreeClassifier(random_state=123)

clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)


# It still quite works poorly but the score increased a little.

# If we sort the features by descending order based on its importance to Class, we have the following result.

# In[ ]:


feature_importances = np.stack((clf2.feature_importances_, list(X_train)), axis=1)
feature_importances = feature_importances[feature_importances.argsort(axis=0)[:, 0]][::-1]
feature_importances


# In many cases, 8 features for a machine learning model aren't considered too much dimension and not really necessary to reduce them. However, let's see how many we can eliminate and still maintain the score.

# In[ ]:


scores = []

for i in range(1, len(feature_importances)+1):
    
    features = feature_importances[:i, 1]

    clf = DecisionTreeClassifier(random_state=123)
    
    clf.fit(X_train.loc[:, features], y_train)
    
    scores.append(clf.score(X_test.loc[:, features], y_test))


# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(scores)
plt.xlabel('Numer of Features', size=15)
plt.ylabel('Scores', size=15)
plt.show();


# In[ ]:


feature_importances


# With the above graph, we could use 3, 4, 5, or 6 features and still have the same score. 

# Note that scores will vary as well as the number of features because training and validation set for models are chosen randomly and since our data is not big, random selection will affect the performance quite much.

# Because we are short in the number of samples and data is not time-series data, there are not much we can do to explore. But still, we could try something like visualizing proportions of Social support (or other features) by different classes. 

# If you find any errors and/or typos or have any suggestion, please let me know!
