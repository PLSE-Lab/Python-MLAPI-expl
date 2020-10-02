#!/usr/bin/env python
# coding: utf-8

# # Data Analysis

# In[ ]:


import pandas as pd
import numpy as np

from scipy.stats import ttest_ind

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


data_df = [train_df, test_df]


# In[ ]:


train_df.head()


# ## General analysis

# Checking data types of loaded data

# In[ ]:


train_df.dtypes


# Converting _object_ and _int64_ data type to _categorical_ for __crew__, __experiment__, __seat__, and __event__ columns. <br>
# These columns are categorical by the description of data on Kaggle. 

# In[ ]:


for df in data_df:
    df.crew = df.crew.astype('category')
    df.experiment = df.experiment.astype('category')
    df.seat = df.seat.astype('category')
    
train_df.event = train_df.event.astype('category')


# Checking for missing values

# In[ ]:


train_df.isnull().sum().sum(), test_df.isnull().sum().sum()


# ## Analyze by column

# ### Categorical columns

# In[ ]:


train_df[['event', 'crew']].groupby(['event', 'crew'], as_index=False).size()


# There is no significant relationship between __event__ and __crew__ columns. __Crew__ column is a candidate for deletion.

# In[ ]:


train_df.experiment.unique().categories, test_df.experiment.unique().categories


# __Experiment__ column is just a description column and it does not hold significant information. As it could not be used as a predictor it will be deleted.

# In[ ]:


train_df[['event', 'seat']].groupby(['event', 'seat'], as_index=False).size()


# There is no significant relationship between __event__ and __seat__ columns. __Seat__ column is a candidate for deletion.

# In[ ]:


#Dropping categorical columns
for df in data_df:
    df.drop(columns=['crew'], inplace = True)
    df.drop(columns=['experiment'], inplace = True)
    df.drop(columns=['seat'], inplace = True)


# ### Numerical columns

# Approximating probability density functions of numerical predictors by plotting histograms of data divided into bins.

# In[ ]:


for predictor in test_df.columns[1:]:
    g = sns.FacetGrid(train_df, col='event')
    g.map(plt.hist, predictor, bins=100)


# For such columns as __time__, __ecg__, __r__ and __gsr__ it is hard to see some pattern. And it would be hard to capture it if it exists. Therefore, we will delete these columns.

# In[ ]:


#Dropping numerical columns
for df in data_df:
    df.drop(columns=['time'], inplace = True)
    df.drop(columns=['ecg'], inplace = True)
    df.drop(columns=['r'], inplace = True)
    df.drop(columns=['gsr'], inplace = True)


# Predictors which remain can be better seen on the log-y scale.

# In[ ]:


for predictor in test_df.columns[1:]:
    g = sns.FacetGrid(train_df, col='event')
    g.set(yscale="log")
    g.map(plt.hist, predictor, bins=100)


# Building box plots of each predictor by the __event__. Plots visualize skewness of data by comparing mean, median and quartiles. 

# In[ ]:


for predictor in test_df.columns[1:]:
    train_df.boxplot(column=predictor, by='event', showmeans=True, showfliers=False)


# Some of the _predictors_ distributions look the same for all of the __event__ types, which means that they won't likely be good predictors for event classification. But there are some other _predictors_ that show different mean for different events, which probably can be used for classifying __events__. We will investigate this means differences further in the next section.

# ## Selecting predictors

# ### Check if feature predicts event

# Testing a hypothesis that values for different __events__ are drown from the same distribution for each predictor. 
# 
# Null hypothesis:
# - values of a predictor for event A and another event are taken from the same distribution (they have the same expected value) 
# 
# Alternative hypothesis:
# - values are taken from different distributions (different expected values)
# 
# We will reject the null hypothesis if the probability of the null hypothesis is less than 5%. scypi.stats.ttest_ind performs such testing.

# In[ ]:


#Helper function
def get_subset(event, predictor):
    return train_df.loc[train_df.event == event, predictor]

#Helper lists
all_predictors = test_df.columns[1:]
events = ['A', 'B', 'C', 'D']


# In[ ]:


distinctive_A_predictors = []

for predictor in all_predictors:
    p_vals = [ttest_ind(get_subset('A', predictor), get_subset(e, predictor), equal_var = False)[1] for e in events[1:]]
    is_distinct = [p < 0.05 for p in p_vals]
    if (all(is_distinct)):
        distinctive_A_predictors.append(predictor)

distinctive_A_predictors


# In[ ]:


distinctive_predictors = []

for predictor in all_predictors:
    p_vals = [ttest_ind(get_subset(events[e1], predictor), get_subset(events[e2], predictor), equal_var = False)[1] for e1 in range(3) for e2 in range(e1+1,4)]
    is_distinct = [p < 0.05 for p in p_vals]
    if (all(is_distinct)):
        distinctive_predictors.append(predictor)

distinctive_predictors


# *distinctive_predictors* contains only those predictors that have a statistically significant difference in expected values for all of the events. We will use only *distinctive_predictors* features in modeling.

# ### Check highly correlated features

# Building correlation matrix

# In[ ]:


corrmat = train_df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# Selecting the most correlated pairs of features

# In[ ]:


abs_corrmat = train_df[distinctive_predictors].corr().abs()
corrlist = abs_corrmat.unstack().sort_values(ascending=False).iloc[len(distinctive_predictors)::2]
corrlist[corrlist > 0.7]


# In[ ]:


uncorrelated_predictors = distinctive_predictors.copy()
uncorrelated_predictors.remove('eeg_p4')
uncorrelated_predictors.remove('eeg_fp2')


# Selecting final set of features

# In[ ]:


selected_predictors = uncorrelated_predictors


# # Model & predict

# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss


# ## Data preparation

# In[ ]:


X = train_df[selected_predictors]
Y = train_df.event
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=.2, random_state=35)
X_train.shape, Y_train.shape, X_val.shape, Y_val.shape 


# ## Modeling

# Chosen models:
# - Linear and Quadratic Discriminant Analysis 
# - Logistic Regression
# - Random Forrest

# In[ ]:


losses = pd.DataFrame(columns=['log_loss','model'])


# Quadratic Discriminant Analysis

# In[ ]:


qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, Y_train)
qda_prob = qda.predict_proba(X_val)
qda_loss = log_loss(Y_val, qda_prob, labels=qda.classes_)
losses.loc["QDA"] = [qda_loss, qda]

qda_loss


# Linear Discriminant Analysis

# In[ ]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
lda_prob = lda.predict_proba(X_val)
lda_loss = log_loss(Y_val, lda_prob, labels=lda.classes_)
losses.loc["LDA"] = [lda_loss, lda]

lda_loss


# Logistic Regression for different values of penalty.

# In[ ]:


logreg = LogisticRegression(multi_class = 'multinomial', solver='lbfgs', C=3, max_iter=500)
logreg.fit(X_train, Y_train)
logreg_prob = logreg.predict_proba(X_val)
logreg_loss = log_loss(Y_val, logreg_prob, labels=logreg.classes_)
losses.loc["Log Regression"] = [logreg_loss, logreg]

logreg_loss


# Random Forrest

# In[ ]:


rforest = RandomForestClassifier(n_estimators = 100 , class_weight="balanced")
rforest.fit(X_train, Y_train)
rforest_prob = rforest.predict_proba(X_val)
rforest_loss = log_loss(Y_val, rforest_prob, labels=rforest.classes_)
losses.loc["R-forest"] = [rforest_loss, rforest]
    
rforest_loss


# ## Analyzing obtained results

# In[ ]:


losses.head()


# In[ ]:


losses.log_loss.sort_values(ascending=True)


# In[ ]:


best_model = losses.model[losses.log_loss.argmin()]


# ## Saving results

# In[ ]:


test_res = best_model.predict_proba(test_df[selected_predictors])
res_df = pd.DataFrame({"A" : test_res[:,0], 
                       "B" : test_res[:,1], 
                       "C" : test_res[:,2], 
                       "D" : test_res[:,3]})
res_df.to_csv('submission.csv', index_label='id')

